"""Audit completed overlap/LoD controls and render diagnostic figures (GT-only plots)."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    b=Path('output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1')
    r=b/'overlap_lod_v402'; figures=r/'figures'; figures.mkdir(exist_ok=True)
    splits=['seq10','shard0','shard1','shard2','shard3']
    baseline=json.load(open(b/'regularized_stage_precision_v357/metrics.json'))['stage_plain_reg_all']
    folders=['flat_fixed','lod_fixed','flat_fixed_final','lod_fixed_final','flat_fixed_consolidated','lod_fixed_consolidated','flat_seed2','lod_seed2','flat_seed2_final','lod_seed2_final','flat_seed2_consolidated','lod_seed2_consolidated']
    summary={}; files={}; audits=[]
    for folder in folders:
        for arm,v in json.load(open(r/folder/'metrics.json')).items():
            assert v['names']==baseline['names'] and len(v['names'])==438 and len(set(v['names']))==438
            summary[folder+'/'+arm]=v['all']
            for split in splits:
                p=r/folder/f'{split}_{arm}.npz'; files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
                with np.load(p) as f:
                    poses=f['pose_w2c']; finite=np.isfinite(poses).all(axis=(1,2)); rot=poses[finite,:3,:3]
                    assert np.allclose(rot@rot.transpose(0,2,1),np.eye(3),atol=1e-5)
                    assert np.allclose(np.linalg.det(rot),1,atol=1e-5)
                    assert np.allclose(poses[finite,3],np.array([0,0,0,1]),atol=1e-6)
                    if arm.endswith('consolidated'):
                        with np.load(b/'regularized_stage_precision_v357'/f'{split}_stage_plain_reg_all.npz') as base:
                            assert np.array_equal(f['names'],base['names'])
                            assert np.array_equal(poses[~f['accepted']],base['pose_w2c'][~f['accepted']])
                audits.append(str(p))
    assert len(summary)==24 and len(audits)==120
    (r/'summary.json').write_text(json.dumps(dict(baseline=baseline['all'],outcomes=summary),indent=2))
    (r/'completion_audit.json').write_text(json.dumps(dict(outcome_groups=24,pose_artifacts=120,queries_per_group=438,names_unique_and_identical=True,finite_rotations_valid=True,consolidated_fallback_exact=True,sha256=files),indent=2))
    def fmt(x):return '∞' if not math.isfinite(x) else f'{x:.3f}'
    lines=['|输出|平移均值/中位数 m|旋转均值/中位数 °|10 cm / 1° %|25 cm / 2° %|50 cm / 5° %|1 m / 10° %|无效位姿 %|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in [('strong baseline',baseline['all']),*summary.items()]:
        rec=m['recall_percent'];lines.append('|'+name+'|'+fmt(m['translation_mean_m'])+'/'+fmt(m['translation_median_m'])+'|'+fmt(m['rotation_mean_deg'])+'/'+fmt(m['rotation_median_deg'])+'|'+'|'.join(f'{rec[k]:.2f}' for k in ['0.1m_1deg','0.25m_2deg','0.5m_5deg','1m_10deg'])+f"|{100*m['invalid_pose_rate']:.2f}|")
    (r/'metrics_table.md').write_text('\n'.join(lines)+'\n')
    diag=json.load(open(r/'diagnostics_equal_count/ideal_support.json'))
    attr=json.load(open(r/'diagnostics_equal_count/token_attrition.json'))['tokens']
    keys=['positive_in_region','positive_in_top32','positive_in_distinct4','independent_positive','mnn_positive','joint_positive']
    counts=[sum(t['physical_proxy_available'] for t in attr)]+[sum(any(a[k] for a in t['regions']) for t in attr) for k in keys]
    fig,axs=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    axs[0].bar(range(7),counts,color=['#2563eb']*4+['#ea580c']*3)
    axs[0].set_xticks(range(7),['Map proxy','Regions','Top 32','4 cells','Null + argmax','MNN','Joint'],rotation=30,ha='right')
    axs[0].set_ylabel('Tower query tokens retaining a positional proxy');axs[0].set_title('Prior matcher: token attrition (19 sampled)')
    for i,c in enumerate(counts):axs[0].text(i,c+.12,str(c),ha='center')
    axs[0].set_ylim(0,12)
    for key,v in diag['results'].items():
        noise=v['noise_coarse_pixels'];xs=sorted(float(x) for x in noise if float(x)>0);ys=[noise[str(x)]['metrics']['translation_median_m'] for x in xs]
        axs[1].plot(xs,ys,'o-',label=f"{key.replace('_',' ')} ({v['anchors']})")
    axs[1].set_yscale('log');axs[1].set_xlabel('Synthetic Gaussian noise, coarse-image pixels');axs[1].set_ylabel('Median translation error (m)');axs[1].legend(fontsize=8);axs[1].set_title('GT-projected correspondences: sensitivity only');axs[1].grid(alpha=.2)
    fig.savefig(figures/'tower_attrition_and_observability.png',dpi=190);fig.savefig(figures/'tower_attrition_and_observability.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained');x=np.arange(3);width=.18
    for j,(folder,arm,label) in enumerate([('flat_fixed_final','mnn_refined','Flat + MNN'),('lod_fixed_final','mnn_refined','LoD + MNN'),('flat_fixed_final','learned_refined','Flat + learned'),('lod_fixed_final','learned_refined','LoD + learned')]):
        m=summary[folder+'/'+arm]['recall_percent'];ax.bar(x+(j-1.5)*width,[m[k] for k in ['0.1m_1deg','0.25m_2deg','0.5m_5deg']],width,label=label)
    ax.set_xticks(x,['10 cm / 1°','25 cm / 2°','50 cm / 5°']);ax.set_ylabel('Recall (%)');ax.set_ylim(0,100);ax.legend(ncol=2);ax.set_title('Four-factor control after identical refinement; seed 1, 438 queries')
    fig.savefig(figures/'four_factor_refined_recall.png',dpi=190);fig.savefig(figures/'four_factor_refined_recall.pdf');plt.close(fig)
    print(json.dumps(dict(groups=len(summary),pose_files=len(files),figures=str(figures))))


if __name__=='__main__':main()
