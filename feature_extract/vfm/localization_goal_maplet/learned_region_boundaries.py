"""Budgeted discrete physical-boundary learning; set value is a mapping proxy."""
import numpy as np


def boundary_value(quality,observed,choice,objective):
    chosen=quality[:,np.arange(len(choice)),choice]
    if objective=='set':return float(chosen.max(1).mean())
    if objective=='independent':return float((chosen.sum(0)/np.maximum(observed.sum(0),1)).sum())
    raise ValueError('unknown objective')


def optimize_boundaries(quality,observed,cost,budget,objective,max_rounds=50):
    n,centers,shapes=quality.shape;choice=np.ones(centers,dtype=int);active=np.flatnonzero(observed.any(0));trace=[]
    if cost[np.arange(centers),choice].sum()>budget:raise ValueError('initial map exceeds budget')
    for _ in range(max_rounds):
        value=boundary_value(quality,observed,choice,objective);used=int(cost[np.arange(centers),choice].sum())
        chosen=quality[:,np.arange(centers),choice];top=np.argsort(chosen,axis=1)[:,-3:];topv=np.take_along_axis(chosen,top,axis=1)
        mean=quality.sum(0)/np.maximum(observed.sum(0),1)[:,None]
        actions=[(int(i),a,int(cost[i,a]-cost[i,choice[i]])) for i in active for a in range(shapes) if a!=choice[i]]
        best_gain=1e-12;best=None
        def gain(acts):
            if objective=='independent':return sum(mean[i,a]-mean[i,choice[i]] for i,a,_ in acts)
            keep=np.ones(top.shape,bool)
            for i,_,_ in acts:keep &= top!=i
            rest=np.where(keep,topv,0).max(1)
            for i,a,_ in acts:rest=np.maximum(rest,quality[:,i,a])
            return float(rest.mean()-value)
        for action in actions:
            if used+action[2]<=budget:
                g=gain([action])
                if g>best_gain:best_gain,best=g,[action]
        # Paired exchanges allow an expansion even when the initial map fills capacity.
        shrink=[a for a in actions if a[2]<0]
        for expansion in [a for a in actions if a[2]>0]:
            for contraction in shrink:
                if expansion[0]==contraction[0] or used+expansion[2]+contraction[2]>budget:continue
                g=gain([expansion,contraction])
                if g>best_gain:best_gain,best=g,[expansion,contraction]
        if best is None:break
        for i,a,_ in best:choice[i]=a
        trace.append({'value':boundary_value(quality,observed,choice,objective),'references':int(cost[np.arange(centers),choice].sum())})
    return choice,trace
