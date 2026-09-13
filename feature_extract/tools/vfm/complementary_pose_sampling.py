"""Query-side complementary sampling without a provisional world-frame pose.

MoGe depth supplies relative geometry, not map overlap or metric ground truth.
Half of samples remain uniform. The local information proxy is not a posterior.
"""
import cv2
import numpy as np


def query_information(points,valid,K,k1):
    points=np.asarray(points,float);valid=np.asarray(valid,bool)&np.isfinite(points).all(1)&(points[:,2]>0)
    scale=np.median(points[valid,2]) if valid.any() else 1.
    safe=np.where(valid[:,None],points/scale,np.array([0.,0.,1.]))
    _,jac=cv2.projectPoints(safe,np.zeros(3),np.zeros(3),np.asarray(K,float),np.array([k1,0.,0.,0.,0.]))
    J=jac[:,:6].reshape(-1,2,6);J[~valid]=0
    column=np.sqrt(np.maximum(np.sum(J*J,axis=(0,1))/max(int(valid.sum()),1),1e-12))
    J=J/column
    return np.einsum('nij,nik->njk',J,J)


class ComplementarySampler:
    def __init__(self,information=None,pool_size=16):
        self.information=information;self.pool_size=pool_size

    def __call__(self,rng,groups,world,tokens):
        n=len(groups)
        if rng.random()<.5:
            chosen=rng.choice(n,4,replace=False)
        else:
            ids=np.array([tokens[g[0]] for g in groups],int)
            xy=np.c_[ids%64,ids//64].astype(float)
            chosen=[int(rng.integers(n))]
            H=None if self.information is None else np.eye(6)*1e-3+self.information[ids[chosen[0]]]
            for _ in range(3):
                available=np.setdiff1d(np.arange(n),chosen,assume_unique=False)
                pool=rng.choice(available,min(self.pool_size,len(available)),replace=False)
                if H is None:
                    score=np.min(np.sum((xy[pool,None]-xy[chosen][None])**2,axis=-1),axis=1)
                else:score=np.linalg.slogdet(H[None]+self.information[ids[pool]])[1]
                j=int(pool[np.argmax(score)]);chosen.append(j)
                if H is not None:H+=self.information[ids[j]]
        return np.array([groups[g][rng.integers(len(groups[g]))] for g in chosen],int)
