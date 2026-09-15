"""Export fixed-seed tables and scientific figures from the sealed reports."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from feature_extract.tools.vfm.report_cambridge_core_benchmark import SCENES,ARMS
from feature_extract.tools.vfm.cambridge_core_benchmark import ROOT

def fmt(x,n=3):return '∞' if not np.isfinite(x) else f'{x:.{n}f}'

def main():
    reports={s:json.load(open(ROOT/f'report_seed{s}.json')) for s in [260901,260902]};rows=[]
    for seed,report in reports.items():
        for scene in SCENES:
            for arm in ARMS:
                m=report['scenes'][scene]['metrics'][arm];row=dict(seed=seed,scene=scene,arm=arm,**{k:v for k,v in m.items() if k!='recall_percent'},**m['recall_percent']);rows.append(row)
    with open(ROOT/'all_metrics.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    keys=['0.1m_1deg','0.25m_2deg','0.5m_5deg'];arms=['pooled_multistart','regional','regional_agreement'];labels=['Global pool, four starts','Regional proposals','Regional + scale agreement'];colors=['#88939e','#4478ad','#228779'];fig,axes=plt.subplots(1,3,figsize=(14,4.6),sharey=True)
    for ax,key,title in zip(axes,keys,['0.10 m / 1°','0.25 m / 2°','0.50 m / 5°']):
        for j,(arm,label,color) in enumerate(zip(arms,labels,colors)):
            values=np.array([[reports[s]['scenes'][scene]['metrics'][arm]['recall_percent'][key] for scene in SCENES] for s in reports]);means=values.mean(0);ax.bar(np.arange(5)+(j-1)*.24,means,.23,label=label,color=color,yerr=np.abs(values-means),error_kw=dict(lw=.8,capsize=2))
        ax.set_xticks(np.arange(5),['Great\nCourt','Kings\nCollege','Old\nHospital','Shop\nFacade','StMarys\nChurch']);ax.set_title(title);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Official-test joint recall (%)');handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.02));fig.suptitle('Cambridge portable-core assay: fixed map budget and parameters',fontsize=13);fig.text(.5,.105,'Bars: mean of two fixed seeds. Whiskers: seed range, not confidence intervals. Not the full v414 cascade.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.14,1,.95]);fig.savefig(ROOT/'five_scene_core_recall.png',dpi=180);fig.savefig(ROOT/'five_scene_core_recall.pdf');plt.close(fig)
    lines=['完整封存的两个种子、五个场景、五个受控端点已评估。下表固定展示 seed260901 的 `regional_agreement`，不按场景挑选赢家。','', '| 场景 | 平移均值/中位 m | 旋转均值/中位 ° | R10/1° % | R25/2° % | R50/5° % | 无效位姿 % |','|---|---:|---:|---:|---:|---:|---:|']
    for scene in SCENES:
        m=reports[260901]['scenes'][scene]['metrics']['regional_agreement'];rec=m['recall_percent'];lines.append(f"| {scene} | {fmt(m['translation_mean_m'])} / {fmt(m['translation_median_m'])} | {fmt(m['rotation_mean_deg'])} / {fmt(m['rotation_median_deg'])} | {rec[keys[0]]:.2f} | {rec[keys[1]]:.2f} | {rec[keys[2]]:.2f} | {m['invalid_pose_rate']*100:.2f} |")
    lines+=['','### 跨场景配对机制效果','','下表是两个种子的平均召回差，单位 pp。`区域−全局多起点` 控制了四次求解/LM 机会；`一致性−区域` 隔离局部优化与一致性接受的组合收益。所有场景均原样保留。','','| 场景 | 区域−全局多起点 R10/R25/R50 | 一致性−区域 R10/R25/R50 |','|---|---:|---:|']
    deltas={}
    for scene in SCENES:
        items=[];deltas[scene]={}
        for pair in ['regional_vs_pooled_multistart','regional_agreement_vs_regional']:
            v=np.array([[reports[s]['scenes'][scene]['paired_comparisons'][pair][key]['delta_pp'] for key in keys] for s in reports]);items.append(' / '.join(f'{x:+.2f}' for x in v.mean(0)));deltas[scene][pair]={'seed_values_pp':v.tolist(),'mean_pp':v.mean(0).tolist()}
        lines.append('| '+scene+' | '+' | '.join(items)+' |')
    lines+=['','### 场景等权宏平均','','| 端点 | seed260901 R10/R25/R50 % | seed260902 R10/R25/R50 % |','|---|---:|---:|']
    for arm in ARMS:lines.append('| '+arm+' | '+' | '.join(' / '.join(f"{reports[s]['macro_scene_recall_percent'][arm][k]:.2f}" for k in keys) for s in reports)+' |')
    lines+=['','全量 50 行场景×种子×端点表见 `output/cambridge_core_v415/all_metrics.csv`；逐查询误差、预测、封存哈希、配对改善/损害率和描述性 bootstrap 区间见两份 `report_seed*.json` 及各场景 `errors_seed*.npz`。图像加权结果另存于报告 JSON，不能替代上述场景等权结果。','','![五场景核心召回](/root/ICLPose/output/cambridge_core_v415/five_scene_core_recall.png)']
    path=Path('docs/vfm/g25_cambridge_five_scene_core_validation_20260914.md');s=path.read_text().replace('状态：实验执行中，最终数值仅在全部预测封存后填写。','状态：两种子五场景完整实验已完成。');s=s.replace('待全部预测完成后自动汇总。','\n'.join(lines));path.write_text(s);(ROOT/'mechanism_deltas.json').write_text(json.dumps(deltas,indent=2));print('\n'.join(lines[:9]))
if __name__=='__main__':main()
