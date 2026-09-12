"""Mapping-supervised conditional fine updates; inference reads no pose labels."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize


def reliability_features(sim,delta,variance):
    sim=np.asarray(sim,float);delta=np.asarray(delta,float);variance=np.asarray(variance,float)
    if sim.ndim!=2 or sim.shape[1]!=9 or delta.shape!=(len(sim),2) or variance.shape!=(len(sim),) or not np.isfinite(sim[:,4]).all() or not np.isfinite(delta).all() or not np.isfinite(variance).all() or np.any(variance<=0):raise ValueError('invalid reliability features')
    valid=np.isfinite(sim);safe=np.where(valid,sim,-2.);ordered=np.sort(safe,axis=1)
    return np.c_[sim[:,4],ordered[:,-1]-sim[:,4],ordered[:,-1]-ordered[:,-2],np.linalg.norm(delta,axis=1),np.log(variance),valid.mean(1)]


def predict(model,features,variance):
    features=np.asarray(features,float);variance=np.asarray(variance,float)
    if features.ndim!=2 or features.shape[1]!=6 or variance.shape!=(len(features),) or not np.isfinite(features).all() or not np.isfinite(variance).all() or np.any(variance<=0):raise ValueError('invalid conditional readout inputs')
    x=np.c_[np.ones(len(features)),(features-np.asarray(model['mean']))/np.asarray(model['scale'])]
    alpha=expit(x@np.asarray(model['alpha_weights']))
    multiplier=np.exp(np.clip(x@np.asarray(model['variance_weights']),-2.,2.))
    return alpha,np.asarray(variance)*multiplier


def fit(features,original,update,target,variance,weights,alpha_prior,variance_prior):
    features=np.asarray(features,float);w=np.asarray(weights,float);w=w/w.sum();delta=update-original
    mean=np.sum(w[:,None]*features,axis=0);scale=np.sqrt(np.sum(w[:,None]*(features-mean)**2,axis=0));scale=np.maximum(scale,1e-5);x=np.c_[np.ones(len(features)),(features-mean)/scale]
    initial=np.zeros(x.shape[1]);initial[0]=np.log(alpha_prior/(1-alpha_prior));ridge=.01
    def loss(beta):
        a=expit(x@beta);r=original+a[:,None]*delta-target
        return np.sum(w*np.sum(r*r,axis=1)) + ridge*np.sum(beta[1:]**2)
    result=minimize(loss,initial,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-12})
    if not result.success:raise ValueError('conditional mean fit failed: '+result.message)
    r=original+expit(x@result.x)[:,None]*delta-target;ratio=np.sum(r*r,axis=1)/(2*variance)
    vi=np.zeros(x.shape[1]);vi[0]=np.log(variance_prior)
    def vloss(beta):
        logscale=np.clip(x@beta,-2.,2.)
        return np.sum(w*(logscale+ratio*np.exp(-logscale)))+ridge*np.sum(beta[1:]**2)
    vr=minimize(vloss,vi,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-12})
    if not vr.success:raise ValueError('conditional variance fit failed: '+vr.message)
    return dict(mean=mean.tolist(),scale=scale.tolist(),alpha_weights=result.x.tolist(),variance_weights=vr.x.tolist(),ridge=ridge,feature_names=['center_cosine','peak_gain','peak_margin','shift_norm','log_original_variance','valid_probe_fraction'],alpha_loss=float(result.fun),variance_loss=float(vr.fun))
