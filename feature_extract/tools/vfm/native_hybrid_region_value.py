"""Set-conditional features and capped region additions relative to the plane frontend."""
import numpy as np
from feature_extract.tools.vfm.native_region_value_features import addition_features


def pooled_additions(original_tokens,original_prototypes,region_tokens,region_prototypes,region_scores,selected,limit=1024):
 existing=set(zip(map(int,original_tokens),map(int,original_prototypes)));pairs={}
 for r in selected:
  for t,p,s in zip(region_tokens[r],region_prototypes[r],region_scores[r]):
   key=(int(t),int(p))
   if key not in existing:pairs[key]=float(s)
 ordered=sorted(pairs,key=lambda k:(-pairs[k],k))[:limit]
 return np.asarray([k[0] for k in ordered],int),np.asarray([k[1] for k in ordered],int),np.asarray([pairs[k] for k in ordered],float)


def hybrid_features(original_tokens,original_prototypes,original_scores,rt,rp,rs,context,centers):
 bt,bp,bs=pooled_additions(original_tokens,original_prototypes,rt,rp,rs,list(range(8)))
 base_tokens=np.r_[original_tokens,bt];base_scores=np.r_[original_scores,bs];base_unique=np.unique(base_tokens);features=[]
 for c in range(8,16):
  ct,cp,cs=pooled_additions(original_tokens,original_prototypes,rt,rp,rs,[c]);nt,np_,ns=pooled_additions(original_tokens,original_prototypes,rt,rp,rs,list(range(8))+[c]);new_unique=np.unique(np.r_[original_tokens,nt])
  old=addition_features(base_tokens,base_scores,ct,cs,context[c],context[:8],centers[c],centers[:8])
  features.append(np.r_[old,len(np.unique(original_tokens))/2304,len(bt)/1024,(len(nt)-len(bt))/1024,len(np.setdiff1d(new_unique,base_unique))/2304,len(np.setdiff1d(base_unique,new_unique))/2304,float(len(nt)==1024)])
 return np.asarray(features)
