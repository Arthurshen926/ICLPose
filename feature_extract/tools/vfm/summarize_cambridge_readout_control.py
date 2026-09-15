"""Publication-format artifacts with explicit exploratory experiment status."""
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from feature_extract.tools.vfm.cambridge_core_benchmark import ROOT
from feature_extract.tools.vfm.report_cambridge_core_benchmark import SCENES
from feature_extract.tools.vfm.summarize_cambridge_core_benchmark import fmt


def main():
    reports={s:json.load(open(ROOT/f'readout_report_seed{s}.json')) for s in [260901,260902]};keys=['0.1m_1deg','0.25m_2deg','0.5m_5deg'];rows=[]
    for seed,report in reports.items():
        for scene in SCENES:
            for arm,m in report['scenes'][scene]['metrics'].items():rows.append(dict(seed=seed,scene=scene,arm=arm,**{k:v for k,v in m.items() if k!='recall_percent'},**m['recall_percent']))
    with open(ROOT/'readout_metrics.csv','w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    deltas=np.array([[[reports[seed]['scenes'][s]['paired_comparisons']['pooled_global_field_vs_pooled_global'][k]['delta_pp'] for k in keys] for s in SCENES] for seed in reports]);summary=dict(mean_scene_and_seed_delta_pp=dict(zip(keys,deltas.mean((0,1)).tolist())),positive_scene_seed_cells=(deltas>1e-9).sum((0,1)).tolist(),equal_scene_seed_cells=(np.abs(deltas)<=1e-9).sum((0,1)).tolist(),negative_scene_seed_cells=(deltas< -1e-9).sum((0,1)).tolist(),post_label_exploratory_design=True)
    summary['all_scene_seed_mean_and_median_errors_decrease']={k:all(reports[seed]['scenes'][s]['metrics']['pooled_global_field'][k]<reports[seed]['scenes'][s]['metrics']['pooled_global'][k] for seed in reports for s in SCENES) for k in ['translation_mean_m','translation_median_m','rotation_mean_deg','rotation_median_deg']};(ROOT/'readout_conclusions.json').write_text(json.dumps(summary,indent=2))
    arms=['regional_global','pooled_global','pooled_global_field'];labels=['Regional + global consensus','Pooled + global consensus','Pooled + continuous field'];colors=['#4478ad','#88939e','#228779'];fig,axes=plt.subplots(1,3,figsize=(14,4.6),sharey=True)
    for ax,key,title in zip(axes,keys,['0.10 m / 1°','0.25 m / 2°','0.50 m / 5°']):
        for j,(arm,label,color) in enumerate(zip(arms,labels,colors)):
            values=np.array([[reports[seed]['scenes'][s]['metrics'][arm]['recall_percent'][key] for s in SCENES] for seed in reports]);m=values.mean(0);ax.bar(np.arange(5)+(j-1)*.24,m,.23,color=color,label=label,yerr=np.abs(values-m),error_kw=dict(lw=.8,capsize=2))
        ax.set_xticks(range(5),['Great\nCourt','Kings\nCollege','Old\nHospital','Shop\nFacade','StMarys\nChurch']);ax.set_ylim(0,100);ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].set_ylabel('Official-test joint recall (%)');handles,labs=axes[0].get_legend_handles_labels();fig.legend(handles,labs,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.02));fig.suptitle('Common-readout diagnostic: geometry consensus and continuous feature alignment',fontsize=12);fig.text(.5,.105,'Exploratory design after first-round label inspection. Bars: two-seed means; whiskers: seed range. Not full v414.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.14,1,.95]);fig.savefig(ROOT/'five_scene_common_readout.png',dpi=180);fig.savefig(ROOT/'five_scene_common_readout.pdf');plt.close(fig)
    lines=['## 共享全局读出的补充诊断（后验设计）','','首轮指标显示分区方案普遍退化。核对代码发现：各区域候选只在各自区域 LM，选中后没有共享库存再拟合；因此首轮并非只改变采样分组，还改变了 LM 支持域。补充为全局池和区域候选都添加相同的三步唯一 token 全局 LM，以及相同的连续特征场精修。所有匹配库存重新计算后逐查询验证 SHA256 一致，无测试位姿进入求解。**这项设计已看过首轮标签结果，只能作为后验诊断与后续复验依据，不能宣称新盲测。**','','### 补全读出后的统一全局版本','','以下固定展示 `pooled_global_field`，不是按场景挑选赢家。每格均为 seed260901 / seed260902。','','| 场景 | R10/1° % | R25/2° % | R50/5° % |','|---|---:|---:|---:|']
    for s in SCENES:lines.append('| '+s+' | '+' | '.join(' / '.join(f"{reports[seed]['scenes'][s]['metrics']['pooled_global_field']['recall_percent'][k]:.2f}" for seed in reports) for k in keys)+' |')
    lines+=['','seed260901 的完整误差：','','| 场景 | 平移均值/中位 m | 旋转均值/中位 ° |','|---|---:|---:|']
    for s in SCENES:
        m=reports[260901]['scenes'][s]['metrics']['pooled_global_field'];lines.append(f"| {s} | {fmt(m['translation_mean_m'])} / {fmt(m['translation_median_m'])} | {fmt(m['rotation_mean_deg'])} / {fmt(m['rotation_median_deg'])} |")
    lines+=['','连续特征读出相对其同输入、同全局共识起点，五场景×两种子的平均召回增量（等场景、等种子权重）：'+ '，'.join(f'{k}: {v:+.3f} pp' for k,v in summary['mean_scene_and_seed_delta_pp'].items())+'。25 cm 为9组提高、1组持平；50 cm 同样9组提高、1组持平。10组平均和中位平移/旋转误差均下降。','','这支持“固定世界锚点上的连续高分辨率外观对齐”作为可迁移的有效子机制。**不能将此独立归因于粗细一致性 gate**：本次全局读出没有额外分离 raw fine 与 gate，首轮已有的 raw fine/gate 消融在主要召回上差别很小。','','共享全局 LM 的作用很大：首轮 seed260901 KingsCollege 区域版 R25 从5.83%升到31.49%，OldHospital 从8.24%升到23.08%，StMarys 从41.32%升到64.72%；GreatCourt 仅从0.39%升到1.45%，仍远差于全局池。这说明缺少全局读出是部分场景的重要问题，但不是 GreatCourt 失败的充分解释。','','![共享读出诊断](/root/ICLPose/output/cambridge_core_v415/five_scene_common_readout.png)','','## 同查询旧主线对照与结论边界','','对同一批350张 seq13 官方测试查询，重新使用官方位姿评估旧 v414：平移均值/中位0.2043/0.1061 m，旋转均值/中位0.8362/0.3679°，R10=47.14%、R25=90.57%、R50=96.57%。旧主线成绩在这些相同查询上仍成立；本轮可移植版明显较低。**这既排除了“只因改了测试划分才掉分”的解释，也说明本轮尚未完成原完整主线的跨场景等价迁移。** 旧主线与本轮地图、精修和补救模块同时不同，该对照不允许把落差归因给某一个模块。','','## 仍需解决的问题与下一步','','1. **先做移植等价验收。** 在共同 seq13 查询上逐阶段复现原完整主线，再把可见性/部分重叠证据、表面可信性和完整补救链变成场景无关接口。本轮五场景核心实验已经完成，原完整主线五场景端到端优势仍未证实。','2. **地图检索单元与位姿求解支持域需要分开。** 当前半径6m的固定区域对 GreatCourt 明显不适用。可检验的下一步是按视角覆盖、几何分布与位姿可观测性扩展多个区域的联合支持，在固定总预算下保留全局探索；不能凭这些实验宣称可学习地图成员没有价值。','3. **绝对精度和长尾仍不足。** 补全全局读出后 GreatCourt 平移中位约1.1m，KingsCollege约0.33m，OldHospital约0.54–0.58m。单靠0.5m信任域局部精修无法处理很多远距离错误初始化。','4. **尚缺深度/可见性 oracle 分解和外部基线。** 本轮自洽投影检查不等于准确重建，也没有隔离重建质量、检索覆盖、候选生成和选择上界，不能武断地将大场景失败归因于某一个环节。没有同地图预算外部定位基线、独立新场景确认与完整成本统计，仍不足以支持顶刊级领先性或新颖性宣称。','5. **真正得到支持的是有限的机制结论。** 共享跨区域几何读出很重要，连续特征对齐有跨场景增益；固定半径区域单独解位姿不构成稳定优势，LoD、匿名地图和生物启发解释本身也未通过独立消融证明因果收益。','','验证：基础求解/局部优化28项测试通过，新增共享全局读出2项测试通过；600张建图视图坐标回放通过；2518份特征模型/维度/有限值契约通过；原四端点在15张跨场景查询上逐元素精确重放；120张训练抽样与官方测试清单共16份文件重建完全一致。等多起点与共享读出每个查询都验证了相同匹配库存哈希。固定缓存重放不等于从2DGS训练开始跨硬件逐位一致的复现。']
    path=Path('docs/vfm/g25_cambridge_five_scene_core_validation_20260914.md');text=path.read_text().split('\n\n## 共享全局读出的补充诊断（后验设计）')[0];text+='\n\n'+'\n'.join(lines)+'\n';intro='\n**最终判断：五场景并未持续保持优秀绝对精度；完整旧主线的跨场景优势尚未证实。得到较稳定支持的是连续高分辨率特征读出的子机制，固定半径区域独立求解没有表现出通用优势。初始冻结实验与看过首轮指标后的共享读出诊断分开报告。**\n';text=text.replace('状态：两种子五场景完整实验已完成。','状态：两种子五场景核心实验及共享读出补充诊断均已完成。'+intro);path.write_text(text);print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
