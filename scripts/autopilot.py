#!/usr/bin/env python3
"""
Autopilot — 持续优化自动化循环
================================
自动执行: 读取状态 → 训练 → 评估 → 记录 → 决策下一轮

协议:
  - 每轮只做一类主要变量改变
  - 保留上一轮可运行配置作为回退基线
  - 连续 N 轮无提升 → 切换策略分支
  - 结构化输出每轮 round report

用法:
    # 从指定基线配置启动
    python scripts/autopilot.py --baseline configs/exp032_cosine_fiters8.yaml

    # 运行A/B对比 (两个config同时训练)
    python scripts/autopilot.py --baseline configs/exp032_cosine_fiters8.yaml \
        --candidate configs/exp033_room0_compressed.yaml

    # 从已有 round history 继续
    python scripts/autopilot.py --baseline configs/exp032_cosine_fiters8.yaml \
        --history output/autopilot/history.json

    # 使用特定GPU
    python scripts/autopilot.py --baseline configs/exp032_cosine_fiters8.yaml \
        --gpus 0,1

    # dry-run: 只生成下一个配置，不训练
    python scripts/autopilot.py --baseline configs/exp032_cosine_fiters8.yaml \
        --dry-run
"""

import os
import sys
import json
import copy
import time
import yaml
import shutil
import signal
import argparse
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

sys.path.insert(0, str(Path(__file__).parent.parent))


# ==============================================================================
#  Metric Parsing
# ==============================================================================

def parse_training_log(log_path: str) -> Dict[str, Any]:
    """Parse a training log file for the best validation metrics.

    Searches for the validation output lines and extracts metrics.
    Returns the best epoch metrics based on rot_mean.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return {}

    text = log_path.read_text()
    best_metrics = {}
    best_rot = float('inf')
    all_epochs = []

    # Parse validation lines: [Val E{epoch}]  rot=... (med ...)  trans=...  <1°=...  joint@0.1°/5.3mm=...
    val_pattern = re.compile(
        r'\[Val E(\d+)\]\s+'
        r'rot=([\d.]+)°\s+\(med ([\d.]+)°\)\s+'
        r'trans=([\d.]+)mm\s+'
        r'<1°=([\d.]+)%'
        r'(?:\s+joint@0\.1°/5\.3mm=([\d.]+)%)?'
        r'(?:\s+joint@1°/50mm=([\d.]+)%)?'
    )
    for m in val_pattern.finditer(text):
        epoch = int(m.group(1))
        metrics = {
            'epoch': epoch,
            'rot_mean': float(m.group(2)),
            'rot_median': float(m.group(3)),
            'trans_mean': float(m.group(4)),
            'pct_1deg': float(m.group(5)),
            'joint_01deg_53mm': float(m.group(6)) if m.group(6) else None,
            'joint_1deg_50mm': float(m.group(7)) if m.group(7) else None,
        }
        all_epochs.append(metrics)
        if metrics['rot_mean'] < best_rot:
            best_rot = metrics['rot_mean']
            best_metrics = metrics

    # Also parse ★ New best lines
    star_pattern = re.compile(
        r'★ New best: rot=([\d.]+)°'
        r'(?:\s+joint@0\.1°/5\.3mm=([\d.]+)%)?'
    )
    last_star = None
    for m in star_pattern.finditer(text):
        last_star = {
            'rot_mean': float(m.group(1)),
            'joint_01deg_53mm': float(m.group(2)) if m.group(2) else None,
        }

    # Check for OOM or NaN
    has_oom = 'CUDA out of memory' in text or 'OutOfMemoryError' in text
    has_nan = 'NaN' in text or 'nan' in text.split('loss=')[-1][:20] if 'loss=' in text else False

    return {
        'best': best_metrics,
        'last_star': last_star,
        'all_epochs': all_epochs,
        'num_epochs_completed': len(all_epochs),
        'has_oom': has_oom,
        'has_nan': has_nan,
    }


def parse_eval_results(json_path: str) -> Dict[str, Any]:
    """Parse evaluation results JSON from eval_iterative.py."""
    p = Path(json_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ==============================================================================
#  Config Management
# ==============================================================================

def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def config_diff(base: Dict, new: Dict, prefix: str = '') -> List[str]:
    """Return a list of human-readable differences between two configs."""
    diffs = []
    all_keys = set(list(base.keys()) + list(new.keys()))
    for k in sorted(all_keys):
        full_key = f"{prefix}.{k}" if prefix else k
        v_base = base.get(k)
        v_new = new.get(k)
        if isinstance(v_base, dict) and isinstance(v_new, dict):
            diffs.extend(config_diff(v_base, v_new, full_key))
        elif v_base != v_new:
            diffs.append(f"{full_key}: {v_base} → {v_new}")
    return diffs


def apply_change(config: Dict, key_path: str, value: Any) -> Dict:
    """Apply a single parameter change to a config dict.

    key_path: dot-separated path, e.g. 'training.lr' or 'model.fine_iters'
    """
    cfg = copy.deepcopy(config)
    keys = key_path.split('.')
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
    return cfg


# ==============================================================================
#  Round Management
# ==============================================================================

class RoundHistory:
    """Track autopilot rounds and their results."""

    def __init__(self, history_path: str = 'output/autopilot/history.json'):
        self.path = Path(history_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rounds: List[Dict] = []
        if self.path.exists():
            with open(self.path) as f:
                self.rounds = json.load(f)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.rounds, f, indent=2, ensure_ascii=False)

    @property
    def next_round_id(self) -> int:
        return len(self.rounds) + 1

    @property
    def best_round(self) -> Optional[Dict]:
        """Return round with best rot_mean."""
        completed = [r for r in self.rounds if r.get('status') == 'completed']
        if not completed:
            return None
        return min(completed, key=lambda r: r.get('metrics', {}).get('rot_mean', float('inf')))

    @property
    def last_round(self) -> Optional[Dict]:
        return self.rounds[-1] if self.rounds else None

    def consecutive_no_improve(self) -> int:
        """Count consecutive rounds with no improvement over best."""
        if not self.rounds:
            return 0
        best = self.best_round
        if not best:
            return 0
        best_rot = best.get('metrics', {}).get('rot_mean', float('inf'))
        count = 0
        for r in reversed(self.rounds):
            if r.get('status') != 'completed':
                continue
            r_rot = r.get('metrics', {}).get('rot_mean', float('inf'))
            if r_rot < best_rot:
                break
            count += 1
        return max(0, count - 1)  # Subtract the best round itself

    def add_round(self, round_data: Dict):
        self.rounds.append(round_data)
        self.save()

    def update_last(self, **kwargs):
        if self.rounds:
            self.rounds[-1].update(kwargs)
            self.save()


# ==============================================================================
#  Experiment Runner
# ==============================================================================

def run_training(
    config_path: str,
    gpu_id: int = 0,
    warmstart: Optional[str] = None,
    resume: Optional[str] = None,
    log_path: Optional[str] = None,
) -> subprocess.Popen:
    """Launch a training run as a subprocess.

    Returns the Popen object for monitoring.
    """
    cmd = [
        sys.executable, 'scripts/train_ms_flow.py',
        '--config', config_path,
    ]
    if warmstart:
        cmd.extend(['--warmstart', warmstart])
    if resume:
        cmd.extend(['--resume', resume])

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    log_file = Path(log_path) if log_path else None
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fout = open(log_file, 'w')
    else:
        fout = None

    print(f"  [GPU {gpu_id}] Launching: {' '.join(cmd)}")
    if log_file:
        print(f"  [GPU {gpu_id}] Log: {log_file}")

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=fout or subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).parent.parent),
    )
    return proc


def run_eval(
    config_path: str,
    checkpoint_path: str,
    gpu_id: int = 0,
    iters: List[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run eval_iterative.py and return parsed results."""
    if iters is None:
        iters = [1, 3, 5]

    cmd = [
        sys.executable, 'scripts/eval_iterative.py',
        '--config', config_path,
        '--checkpoint', checkpoint_path,
        '--iters', *[str(i) for i in iters],
    ]
    if output_path:
        cmd.extend(['--output', output_path])

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )

    if result.returncode != 0:
        print(f"  [Eval] Failed:\n{result.stderr[:500]}")
        return {}

    if output_path and Path(output_path).exists():
        return parse_eval_results(output_path)

    return {'stdout': result.stdout}


def wait_for_training(
    proc: subprocess.Popen,
    log_path: Optional[str] = None,
    poll_interval: int = 30,
) -> int:
    """Wait for training subprocess to complete, printing periodic status."""
    start = time.time()
    while proc.poll() is None:
        elapsed = (time.time() - start) / 60
        # Print periodic status from log
        if log_path and Path(log_path).exists():
            log_text = Path(log_path).read_text()
            # Find last [Val E...] or [Train E...]
            val_lines = [l for l in log_text.split('\n') if '[Val E' in l]
            if val_lines:
                print(f"  [{elapsed:.0f}min] {val_lines[-1].strip()}")
        time.sleep(poll_interval)

    elapsed = (time.time() - start) / 60
    print(f"  Training completed in {elapsed:.0f}min (exit code: {proc.returncode})")
    return proc.returncode


# ==============================================================================
#  Strategy Engine
# ==============================================================================

# Ordered list of single-variable changes to try
DEFAULT_STRATEGY_QUEUE = [
    # PHASE 1: Training hyperparams (safest, fastest to verify)
    {'name': 'lr_warmup', 'key': 'training.lr', 'values': [3e-5, 7e-5, 1e-4]},
    {'name': 'fine_iters_12', 'key': 'model.fine_iters', 'values': [12]},
    {'name': 'gamma_085', 'key': 'training.loss.gamma', 'values': [0.85]},
    {'name': 'gamma_09', 'key': 'training.loss.gamma', 'values': [0.9]},
    {'name': 'val_outer_iters_7', 'key': 'training.val_outer_iters', 'values': [7]},
    {'name': 'val_outer_iters_10', 'key': 'training.val_outer_iters', 'values': [10]},
    {'name': 'outer_iters_5', 'key': 'training.outer_iters', 'values': [5]},
    {'name': 'trans_weight_20', 'key': 'training.loss.trans_weight', 'values': [20.0]},

    # PHASE 2: Model architecture (medium risk)
    {'name': 'pe_concat', 'key': 'model.positional_encoding', 'values': [True],
     'extra': {'model.pe_mode': 'concat', 'model.pe_dim': 32}},
    {'name': 'irls_3', 'key': 'model.irls_iters', 'values': [3]},
    {'name': 'learnable_temp', 'key': 'model.learnable_temperature', 'values': [True]},
    {'name': 'dir_confidence', 'key': 'model.directional_confidence', 'values': [True]},
    {'name': 'ms_consistency', 'key': 'model.multiscale_consistency', 'values': [True]},
    {'name': 'adaptive_damp', 'key': 'model.adaptive_damping', 'values': [True]},
    {'name': 'local_radius_6', 'key': 'model.local_radius', 'values': [6]},
    {'name': 'corr_dilations', 'key': 'model.corr_dilations', 'values': [[1, 2, 4]]},

    # PHASE 3: Noise curriculum (medium risk)
    {'name': 'curriculum_slow',
     'key': 'training.noise_curriculum.enabled', 'values': [True],
     'extra': {'training.noise_curriculum.warmup_epochs': 40,
               'training.noise_curriculum.noise_rot_min': 2.0,
               'training.noise_curriculum.noise_trans_min': 0.05}},

    # PHASE 4: Deeper architecture changes
    {'name': 'deep_flow_head', 'key': 'model.deep_flow_head', 'values': [True]},
    {'name': 'cross_scale_ctx', 'key': 'model.cross_scale_context', 'values': [True]},
    {'name': 'geo_upsample_2x', 'key': 'model.geometry_upsample', 'values': [2]},
    {'name': 'pixel_stride_2', 'key': 'model.pixel_stride', 'values': [2]},
    {'name': 'dino_all_scales', 'key': 'model.dino_all_scales', 'values': [True]},

    # PHASE 5: Plan-based improvements (from plans/full.md Phase 1)
    {'name': 'localizability_prior',
     'key': 'model.localizability_prior', 'values': [True],
     'extra': {'training.loss.localizability_weight': 0.01}},
    {'name': 'flow_consistency_005',
     'key': 'training.loss.flow_consistency_weight', 'values': [0.05]},
    {'name': 'flow_consistency_010',
     'key': 'training.loss.flow_consistency_weight', 'values': [0.10]},
    {'name': 'per_scale_lr',
     'key': 'training.per_scale_lr.enabled', 'values': [True],
     'extra': {'training.per_scale_lr.coarse_scale': 0.5,
               'training.per_scale_lr.mid_scale': 0.75,
               'training.per_scale_lr.fine_scale': 1.0}},
    {'name': 'depth_pe_8',
     'key': 'model.depth_pe_dim', 'values': [8],
     'extra': {'model.positional_encoding': True, 'model.pe_mode': 'concat', 'model.pe_dim': 32}},
    {'name': 'loc_prior_flow_cons_combined',
     'key': 'model.localizability_prior', 'values': [True],
     'extra': {'training.loss.localizability_weight': 0.01,
               'training.loss.flow_consistency_weight': 0.05}},
]


def get_next_strategy(
    history: RoundHistory,
    strategy_queue: List[Dict] = None,
) -> Optional[Dict]:
    """Pick the next untried strategy from the queue."""
    if strategy_queue is None:
        strategy_queue = DEFAULT_STRATEGY_QUEUE

    tried_names = {r.get('strategy_name') for r in history.rounds}
    for s in strategy_queue:
        if s['name'] not in tried_names:
            return s
    return None


def generate_round_config(
    baseline_config: Dict,
    strategy: Dict,
    round_id: int,
    output_dir_base: str = 'output/autopilot',
) -> Tuple[Dict, str]:
    """Generate a new config from baseline + strategy change.

    Returns (new_config, config_path).
    """
    cfg = copy.deepcopy(baseline_config)

    # Apply main change
    key = strategy['key']
    value = strategy['values'][0]  # Use first value
    keys = key.split('.')
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value

    # Apply extra changes if any
    for extra_key, extra_val in strategy.get('extra', {}).items():
        ek = extra_key.split('.')
        d = cfg
        for k in ek[:-1]:
            d = d.setdefault(k, {})
        d[ek[-1]] = extra_val

    # Update experiment metadata
    exp_name = f"autopilot_r{round_id:03d}_{strategy['name']}"
    cfg['exp_name'] = exp_name
    cfg['output_dir'] = f"{output_dir_base}/{exp_name}"

    config_path = f"configs/autopilot/{exp_name}.yaml"
    return cfg, config_path


# ==============================================================================
#  Round Report
# ==============================================================================

def format_round_report(round_data: Dict, prev_best: Optional[Dict] = None) -> str:
    """Format a structured round report per the protocol."""
    r = round_data
    m = r.get('metrics', {})
    lines = [
        f"{'=' * 60}",
        f"Round {r['round_id']}: {r.get('strategy_name', 'baseline')}",
        f"{'=' * 60}",
        f"  Config: {r.get('config_path', 'N/A')}",
        f"  Changes: {'; '.join(r.get('changes', ['baseline']))}",
        f"  Status: {r.get('status', 'unknown')}",
        "",
        f"  Results:",
        f"    rot_mean:           {m.get('rot_mean', 'N/A'):.4f}°" if isinstance(m.get('rot_mean'), (int, float)) else f"    rot_mean:           N/A",
        f"    rot_median:         {m.get('rot_median', 'N/A'):.4f}°" if isinstance(m.get('rot_median'), (int, float)) else f"    rot_median:         N/A",
        f"    trans_mean:         {m.get('trans_mean', 'N/A'):.1f}mm" if isinstance(m.get('trans_mean'), (int, float)) else f"    trans_mean:         N/A",
        f"    joint@0.1°/5.3mm:  {m.get('joint_01deg_53mm', 'N/A'):.1f}%" if isinstance(m.get('joint_01deg_53mm'), (int, float)) else f"    joint@0.1°/5.3mm:  N/A",
        f"    joint@1°/50mm:     {m.get('joint_1deg_50mm', 'N/A'):.1f}%" if isinstance(m.get('joint_1deg_50mm'), (int, float)) else f"    joint@1°/50mm:     N/A",
        f"    pct_1deg:           {m.get('pct_1deg', 'N/A'):.1f}%" if isinstance(m.get('pct_1deg'), (int, float)) else f"    pct_1deg:           N/A",
    ]

    if r.get('has_oom'):
        lines.append("    ⚠ OOM detected")
    if r.get('has_nan'):
        lines.append("    ⚠ NaN detected")

    # Delta from previous best
    if prev_best and m:
        pb = prev_best.get('metrics', {})
        if pb.get('rot_mean') is not None and m.get('rot_mean') is not None:
            delta_rot = m['rot_mean'] - pb['rot_mean']
            delta_sign = '+' if delta_rot >= 0 else ''
            lines.append(f"\n  vs Previous Best (Round {prev_best.get('round_id', '?')}):")
            lines.append(f"    Δ rot_mean:          {delta_sign}{delta_rot:.4f}°")
        if pb.get('joint_01deg_53mm') is not None and m.get('joint_01deg_53mm') is not None:
            delta_j = m['joint_01deg_53mm'] - pb['joint_01deg_53mm']
            delta_sign = '+' if delta_j >= 0 else ''
            lines.append(f"    Δ joint@0.1°/5.3mm:  {delta_sign}{delta_j:.1f}%")

    lines.append("")
    return '\n'.join(lines)


# ==============================================================================
#  Main Autopilot Loop
# ==============================================================================

def autopilot_loop(args):
    """Main autopilot execution loop."""
    history = RoundHistory(args.history)
    baseline_config = load_config(args.baseline)
    baseline_name = Path(args.baseline).stem

    # Parse GPU list
    gpus = [int(g) for g in args.gpus.split(',')]
    primary_gpu = gpus[0]

    print(f"{'=' * 60}")
    print(f"  ICLPose Autopilot")
    print(f"  Baseline: {args.baseline}")
    print(f"  GPUs: {gpus}")
    print(f"  History: {args.history}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Max no-improve: {args.max_no_improve}")
    print(f"{'=' * 60}\n")

    # Strategy queue — filter already tried
    strategy_queue = DEFAULT_STRATEGY_QUEUE.copy()
    if args.strategies:
        # Filter to only user-specified strategies
        strategy_queue = [s for s in strategy_queue if s['name'] in args.strategies]

    # Load custom strategies from file if provided
    if args.strategy_file:
        with open(args.strategy_file) as f:
            custom = yaml.safe_load(f)
        strategy_queue = custom.get('strategies', strategy_queue)

    # Determine current baseline (rolling best)
    current_baseline = baseline_config
    current_baseline_path = args.baseline
    current_best_checkpoint = args.warmstart

    if history.best_round:
        br = history.best_round
        print(f"[History] Best round: R{br['round_id']} "
              f"(rot={br['metrics'].get('rot_mean', '?')}°)")
        if br.get('best_checkpoint') and Path(br['best_checkpoint']).exists():
            current_best_checkpoint = br['best_checkpoint']
        if br.get('config_path') and Path(br['config_path']).exists():
            current_baseline = load_config(br['config_path'])
            current_baseline_path = br['config_path']
    else:
        print("[History] No previous rounds. Starting fresh.")

    no_improve_count = history.consecutive_no_improve()
    print(f"[History] Consecutive no-improve: {no_improve_count}")

    for round_num in range(args.max_rounds):
        round_id = history.next_round_id

        # Check stop conditions
        if no_improve_count >= args.max_no_improve:
            print(f"\n[STOP] {no_improve_count} consecutive rounds with no improvement. "
                  f"Requesting human decision.")
            break

        # Get next strategy
        strategy = get_next_strategy(history, strategy_queue)
        if strategy is None:
            print("\n[STOP] All strategies exhausted.")
            break

        print(f"\n{'━' * 60}")
        print(f"  Round {round_id}: {strategy['name']}")
        print(f"{'━' * 60}")

        # Generate config
        new_config, config_path = generate_round_config(
            current_baseline, strategy, round_id,
            output_dir_base=args.output_dir,
        )
        changes = config_diff(current_baseline, new_config)
        print(f"  Changes: {changes}")

        # Save config
        save_config(new_config, config_path)

        # Build round data
        round_data = {
            'round_id': round_id,
            'strategy_name': strategy['name'],
            'config_path': config_path,
            'baseline_path': current_baseline_path,
            'changes': changes,
            'warmstart': current_best_checkpoint,
            'gpu': primary_gpu,
            'started_at': datetime.now().isoformat(),
            'status': 'running',
        }
        history.add_round(round_data)

        if args.dry_run:
            print(f"  [DRY RUN] Config saved to {config_path}")
            print(f"  Changes: {changes}")
            history.update_last(status='dry_run')
            continue

        # Launch training
        exp_name = new_config['exp_name']
        log_path = f"{args.output_dir}/{exp_name}/train.log"

        proc = run_training(
            config_path=config_path,
            gpu_id=primary_gpu,
            warmstart=current_best_checkpoint,
            log_path=log_path,
        )

        # Wait for completion
        exit_code = wait_for_training(proc, log_path=log_path)

        # Parse results
        parsed = parse_training_log(log_path)
        best_metrics = parsed.get('best', {})

        # Find best checkpoint
        ckpt_dir = Path(args.output_dir) / exp_name / 'checkpoints'
        best_ckpt = str(ckpt_dir / 'best.pth') if (ckpt_dir / 'best.pth').exists() else None

        # Update round
        status = 'completed'
        if exit_code != 0:
            status = 'failed'
        if parsed.get('has_oom'):
            status = 'oom'
        if parsed.get('has_nan'):
            status = 'nan_diverged'

        history.update_last(
            status=status,
            exit_code=exit_code,
            metrics=best_metrics,
            best_checkpoint=best_ckpt,
            has_oom=parsed.get('has_oom', False),
            has_nan=parsed.get('has_nan', False),
            completed_at=datetime.now().isoformat(),
            num_epochs=parsed.get('num_epochs_completed', 0),
        )

        # Run evaluation if training succeeded
        if status == 'completed' and best_ckpt:
            eval_output = f"{args.output_dir}/{exp_name}/eval_results.json"
            eval_results = run_eval(
                config_path=config_path,
                checkpoint_path=best_ckpt,
                gpu_id=primary_gpu,
                iters=[1, 3, 5],
                output_path=eval_output,
            )
            history.update_last(eval_results=eval_results)

        # Print report
        report = format_round_report(
            history.last_round,
            prev_best=history.best_round,
        )
        print(report)

        # Save report to file
        report_path = Path(args.output_dir) / exp_name / 'round_report.txt'
        report_path.write_text(report)

        # Update rolling baseline if improved
        prev_best = history.best_round
        if prev_best and prev_best['round_id'] == round_id:
            print(f"  ✓ Round {round_id} is new best! Updating baseline.")
            current_baseline = new_config
            current_baseline_path = config_path
            current_best_checkpoint = best_ckpt
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"  ✗ No improvement ({no_improve_count}/{args.max_no_improve})")

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"  Autopilot Summary")
    print(f"{'=' * 60}")
    print(f"  Total rounds: {len(history.rounds)}")
    if history.best_round:
        br = history.best_round
        m = br.get('metrics', {})
        print(f"  Best round: R{br['round_id']} ({br.get('strategy_name', 'N/A')})")
        print(f"    rot_mean:           {m.get('rot_mean', 'N/A')}")
        print(f"    joint@0.1°/5.3mm:  {m.get('joint_01deg_53mm', 'N/A')}")
        print(f"    Config: {br.get('config_path', 'N/A')}")
        print(f"    Checkpoint: {br.get('best_checkpoint', 'N/A')}")
    print()


# ==============================================================================
#  Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ICLPose Autopilot — Continuous Optimization Loop',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--baseline', type=str, required=True,
                        help='Baseline config YAML to start from')
    parser.add_argument('--warmstart', type=str, default=None,
                        help='Initial checkpoint for warmstart')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU IDs (default: 0)')
    parser.add_argument('--history', type=str,
                        default='output/autopilot/history.json',
                        help='Path to round history JSON')
    parser.add_argument('--output-dir', type=str,
                        default='output/autopilot',
                        help='Base output directory for autopilot experiments')
    parser.add_argument('--max-rounds', type=int, default=50,
                        help='Maximum number of rounds to run')
    parser.add_argument('--max-no-improve', type=int, default=3,
                        help='Stop after N consecutive rounds with no improvement')
    parser.add_argument('--strategies', type=str, nargs='*', default=None,
                        help='Only try these strategy names')
    parser.add_argument('--strategy-file', type=str, default=None,
                        help='Custom strategy definitions YAML file')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate configs without running training')

    args = parser.parse_args()

    if not Path(args.baseline).exists():
        print(f"Error: Baseline config not found: {args.baseline}")
        sys.exit(1)

    autopilot_loop(args)


if __name__ == '__main__':
    main()
