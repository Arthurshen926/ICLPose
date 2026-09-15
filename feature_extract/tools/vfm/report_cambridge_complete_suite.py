"""Summarize only five completed official-test full-method evaluations."""
import argparse
import json
from pathlib import Path
from feature_extract.vfm.localization_goal_maplet.lineage import file_sha256

SCENES = ['GreatCourt', 'KingsCollege', 'OldHospital', 'ShopFacade', 'StMarysChurch']


def report(root):
    paths = [root / scene / 'full_test_evaluation/metrics.json' for scene in SCENES]
    missing = [scene for scene, p in zip(SCENES, paths) if not p.is_file()]
    if missing:
        raise FileNotFoundError('Full-method evaluations still pending: ' + ', '.join(missing))
    rows = [json.loads(p.read_text()) for p in paths]
    lines = ['# Cambridge 五场景完整 v414 方法，新高斯先验', '',
             '全部官方 test；失败保留在分母。召回同时限制平移和旋转误差。种子1/2对应原求解器260901/260902，地图与模型共享。', '',
             '| 场景 | 种子 | 平移均值/中位数 m | 旋转均值/中位数 ° | R10/1° % | R25/2° % | R50/5° % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for scene, row in zip(SCENES, rows):
        if row['scene'] != scene or not row['all_queries_in_denominator']:
            raise ValueError('Evaluation contract differs: ' + scene)
        for seed in ['1', '2']:
            m = row['reports'][seed]['all']
            if m['images'] != row['official_test_queries']:
                raise ValueError('Incomplete official denominator: ' + scene)
            rec = m['recall_percent']
            lines.append(f"| {scene} | {seed} | {m['translation_mean_m']:.6f}/{m['translation_median_m']:.6f} | "
                         f"{m['rotation_mean_deg']:.6f}/{m['rotation_median_deg']:.6f} | "
                         f"{rec['0.1m_1deg']:.4f} | {rec['0.25m_2deg']:.4f} | {rec['0.5m_5deg']:.4f} |")
    lines += ['', '每场景地图、投影和坐标头重新拟合；通用选择器沿用原 StMary 权重。'
              'StMary 历史查询曾用于多轮方法分析，本表不作全新盲测声明。'
              '用户已确认五份高斯先验均仅用官方 train 图像重建；记录为用户来源声明。', '',
              '表格不混入旧 PLY 的 v417 结果，也不包含 v415 简化方法。']
    for scene,row in zip(SCENES,rows):
        if row.get('protocol_amendment'):
            lines += ['', scene + '：表面坐标头采用仅用训练校准子集的 p90 约束标量校准；网络权重不变，独立验证门槛不变。其余四场景保留原校准，因此本表不是所有场景校准策略完全一致的消融实验。']
    (root / 'complete_suite_metrics.md').write_text('\n'.join(lines) + '\n')
    (root / 'SUITE_COMPLETE.json').write_text(json.dumps(dict(
        five_scene_validation_complete=True, all_queries_in_denominator=True,
        method='All accepted v414 ancestors', sources={str(p): file_sha256(p) for p in paths}), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('output/cambridge_full_v418'))
    report(p.parse_args().root)


if __name__ == '__main__':
    main()
