#!/usr/bin/env python3
"""
训练进度监控脚本
持续监测 exp010 / exp011 的训练日志，报告最新指标和收敛趋势。
当检测到接近收敛且平移精度未达标时发出提示。

用法:
    python scripts/monitor_training.py [--interval 120] [--trans_target 0.02]
"""
import re
import time
import argparse
from pathlib import Path
from datetime import datetime


def parse_val_line(line):
    """解析 Val: 行"""
    m = re.search(r'rot:.*?→\s*([\d.]+)°.*?mean\s*([\d.]+)°.*?<1°=(\d+\.?\d*)%.*?<5°=(\d+\.?\d*)%', line)
    if m:
        return {
            'rot_median': float(m.group(1)),
            'rot_mean': float(m.group(2)),
            'lt1': float(m.group(3)),
            'lt5': float(m.group(4)),
        }
    return None


def parse_seq2_line(line):
    """解析 Seq2: 行"""
    m = re.search(r'rot:.*?→\s*([\d.]+)°.*?mean\s*([\d.]+)°.*?<1°=(\d+\.?\d*)%.*?<5°=(\d+\.?\d*)%.*?trans=([\d.]+)m', line)
    if m:
        return {
            'rot_median': float(m.group(1)),
            'rot_mean': float(m.group(2)),
            'lt1': float(m.group(3)),
            'lt5': float(m.group(4)),
            'trans': float(m.group(5)),
        }
    return None


def parse_epoch_line(line):
    """解析 Epoch 行"""
    m = re.search(r'Epoch\s+(\d+)/(\d+)', line)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_train_line(line):
    """解析 Train: 行, 提取trans"""
    m = re.search(r'trans:\s*[\d.]+ → ([\d.]+)m', line)
    if m:
        return float(m.group(1))
    return None


def parse_best_line(line):
    """解析 Best model saved 行"""
    m = re.search(r'val_rot=([\d.]+)°', line)
    if m:
        return float(m.group(1))
    return None


def analyze_log(log_path):
    """分析完整训练日志"""
    if not log_path.exists():
        return None
    
    lines = log_path.read_text().strip().split('\n')
    
    result = {
        'current_epoch': 0,
        'total_epochs': 0,
        'val_history': [],
        'seq2_history': [],
        'train_trans_history': [],
        'best_rot': float('inf'),
        'best_trans': float('inf'),
    }
    
    current_epoch = 0
    for line in lines:
        # Epoch
        ep, total = parse_epoch_line(line)
        if ep is not None:
            current_epoch = ep
            result['current_epoch'] = ep
            result['total_epochs'] = total
        
        # Val
        val = parse_val_line(line)
        if val and 'Val:' in line:
            val['epoch'] = current_epoch
            result['val_history'].append(val)
            if val['rot_median'] < result['best_rot']:
                result['best_rot'] = val['rot_median']
        
        # Seq2
        if 'Seq2:' in line:
            seq2 = parse_seq2_line(line)
            if seq2:
                seq2['epoch'] = current_epoch
                result['seq2_history'].append(seq2)
                if seq2['trans'] < result['best_trans']:
                    result['best_trans'] = seq2['trans']
        
        # Train trans
        if 'Train:' in line:
            train_trans = parse_train_line(line)
            if train_trans is not None:
                result['train_trans_history'].append((current_epoch, train_trans))
    
    return result


def check_convergence(val_history, window=5):
    """检查是否接近收敛 (最近 window 个验证结果的标准差 < 阈值)"""
    if len(val_history) < window:
        return False, float('inf')
    
    recent = [v['rot_median'] for v in val_history[-window:]]
    import statistics
    std = statistics.stdev(recent)
    mean = statistics.mean(recent)
    # 收敛条件: 标准差 < 均值的 10%
    return std < mean * 0.10, std


def print_report(exp_name, result, trans_target):
    """打印单个实验的报告"""
    print(f"\n{'='*60}")
    print(f"  {exp_name}  |  Epoch {result['current_epoch']}/{result['total_epochs']}")
    print(f"{'='*60}")
    
    # 最近5个Val结果
    print("\n  最近验证结果:")
    for v in result['val_history'][-5:]:
        marker = " ★" if v['rot_median'] == result['best_rot'] else ""
        print(f"    epoch {v['epoch']:3d}: rot={v['rot_median']:.2f}° (mean {v['rot_mean']:.2f}°)  "
              f"<1°={v['lt1']:.1f}%  <5°={v['lt5']:.1f}%{marker}")
    
    # Seq2结果
    if result['seq2_history']:
        print("\n  Seq2 测试结果:")
        for s in result['seq2_history']:
            trans_status = "✓" if s['trans'] <= trans_target else "✗"
            print(f"    epoch {s['epoch']:3d}: rot={s['rot_median']:.2f}° trans={s['trans']:.4f}m "
                  f"[{trans_status} target={trans_target:.3f}m]  <5°={s['lt5']:.1f}%")
    
    # 训练集平移趋势
    if result['train_trans_history']:
        recent_trans = [t for _, t in result['train_trans_history'][-10:]]
        print(f"\n  训练集trans趋势(最近10): {min(recent_trans):.4f}~{max(recent_trans):.4f}m")
    
    # 收敛分析
    converged, std = check_convergence(result['val_history'])
    best = result['best_rot']
    print(f"\n  最佳rotation: {best:.2f}°")
    if result['best_trans'] < float('inf'):
        print(f"  最佳translation: {result['best_trans']:.4f}m (目标: {trans_target:.3f}m)")
        if result['best_trans'] > trans_target:
            gap = result['best_trans'] / trans_target
            print(f"  ⚠ 平移精度差距: {gap:.1f}x 目标值")
    
    if converged:
        print(f"\n  ⟹ 检测到收敛趋势 (val std={std:.3f}°)")
        if result['best_trans'] > trans_target:
            print(f"  ⟹ ⚠⚠⚠ 平移精度未达标！建议启动 task-driven 特征变换实验 ⚠⚠⚠")
    else:
        progress = result['current_epoch'] / result['total_epochs'] * 100 if result['total_epochs'] > 0 else 0
        print(f"\n  训练进度: {progress:.0f}%  val变化: std={std:.3f}°  (仍在优化中)")


def main():
    parser = argparse.ArgumentParser(description='训练进度监控')
    parser.add_argument('--interval', type=int, default=120, help='监控间隔(秒)')
    parser.add_argument('--trans_target', type=float, default=0.02, help='平移精度目标(米)')
    parser.add_argument('--once', action='store_true', help='只报告一次，不循环')
    args = parser.parse_args()
    
    base = Path(__file__).parent.parent / 'output' / 'corr_pose'
    experiments = {
        'exp010_sim_mixed': base / 'exp010_sim_mixed' / 'train_log.txt',
        'exp011_finetune_sim': base / 'exp011_finetune_sim' / 'train_log.txt',
    }
    
    print(f"训练监控启动 | 目标trans: {args.trans_target:.3f}m | 间隔: {args.interval}s")
    
    while True:
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n{'#'*60}")
        print(f"  监控报告  {now}")
        print(f"{'#'*60}")
        
        any_converged_but_miss = False
        
        for name, log_path in experiments.items():
            result = analyze_log(log_path)
            if result is None:
                print(f"\n  {name}: 日志文件不存在")
                continue
            print_report(name, result, args.trans_target)
            
            converged, _ = check_convergence(result['val_history'])
            if converged and result['best_trans'] > args.trans_target:
                any_converged_but_miss = True
        
        if any_converged_but_miss:
            print(f"\n{'!'*60}")
            print(f"  ⚠ 建议: 至少一个实验已收敛但平移未达标 ({args.trans_target*100:.0f}cm)")
            print(f"  ⚠ 可以开始实现 task-driven 特征变换 (PoseAwareUpsampler)")
            print(f"  ⚠ 推荐配置: 768d→64d, 35x46→140x184, 联合训练")
            print(f"{'!'*60}")
        
        if args.once:
            break
        
        print(f"\n下次检查: {args.interval}秒后...")
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
