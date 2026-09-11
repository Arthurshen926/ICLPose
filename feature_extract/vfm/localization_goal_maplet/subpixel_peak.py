"""Bounded continuous maximum of a locally concave 3x3 similarity surface."""
import numpy as np

def quadratic_peak(similarity,step=4/3):
 s=np.asarray(similarity,float)
 if s.ndim!=2 or s.shape[1]!=9 or not np.isfinite(s).all() or step<=0:raise ValueError('finite Nx9 scores and positive step required')
 xy=np.array([(x,y) for y in [-step,0.,step] for x in [-step,0.,step]])
 tie=s.copy();tie[:,4]+=1e-7;best=np.argmax(tie,axis=1);discrete=xy[best];x,y=xy.T
 design=np.c_[x*x,x*y,y*y,x,y,np.ones(9)];coef=s@np.linalg.pinv(design).T
 h=np.zeros((len(s),2,2));h[:,0,0]=2*coef[:,0];h[:,1,1]=2*coef[:,2];h[:,0,1]=h[:,1,0]=coef[:,1]
 concave=np.linalg.eigvalsh(h)[:,-1]<-1e-6;proposal=discrete.copy()
 ids=np.flatnonzero(concave)
 if len(ids):proposal[ids]=-np.linalg.solve(h[ids],coef[ids,3:5])
 accepted=concave&np.isfinite(proposal).all(1)&(np.abs(proposal)<=step).all(1)
 return np.where(accepted[:,None],proposal,discrete),accepted,discrete
