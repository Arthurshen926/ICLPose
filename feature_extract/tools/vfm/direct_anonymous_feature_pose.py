"""Local pose alignment to a frozen anonymous feature field.

Feature residuals are not independent calibrated observations. The image-motion
penalty is a trust-region regularizer, not a second measurement likelihood.
"""
import cv2
import numpy as np
from scipy.optimize import minimize


def sample_with_gradient(grid,pixels):
    grid=np.asarray(grid,float);p=np.asarray(pixels,float);h,w,d=grid.shape
    raw=(p+.5)*np.array([w/256,h/144])-.5
    x=np.clip(raw[:,0],0,w-1);y=np.clip(raw[:,1],0,h-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);x1=np.minimum(x0+1,w-1);y1=np.minimum(y0+1,h-1)
    wx=x-x0;wy=y-y0
    a,b,c,e=grid[y0,x0],grid[y0,x1],grid[y1,x0],grid[y1,x1]
    value=(1-wy[:,None])*((1-wx[:,None])*a+wx[:,None]*b)+wy[:,None]*((1-wx[:,None])*c+wx[:,None]*e)
    dx=((1-wy[:,None])*(b-a)+wy[:,None]*(e-c))*w/256
    dy=((1-wx[:,None])*(c-a)+wx[:,None]*(e-b))*h/144
    dx[(raw[:,0]<=0)|(raw[:,0]>=w-1)]=0;dy[(raw[:,1]<=0)|(raw[:,1]>=h-1)]=0
    norm=np.maximum(np.linalg.norm(value,axis=1),1e-12);f=value/norm[:,None]
    gradient=np.stack([(g-f*np.sum(f*g,axis=1)[:,None])/norm[:,None] for g in [dx,dy]],axis=-1)
    return f,gradient


def contrast_with_gradient(grid,coarse,pixels):
    fine,df=sample_with_gradient(grid,pixels);low,dl=sample_with_gradient(coarse,pixels)
    value=fine-low;norm=np.maximum(np.linalg.norm(value,axis=1),1e-12);feature=value/norm[:,None];d=df-dl
    gradient=(d-feature[:,:,None]*np.einsum('nd,ndk->nk',feature,d)[:,None,:])/norm[:,None,None]
    return feature,gradient


def objective(x,camera0,K,k1,grid,target,initial_pixels,regularization,weights=None,robust=False,bias=None,profile_bias=False,contrast_grid=None):
    pixels,jac=cv2.projectPoints(camera0,x[:3],x[3:],K,np.array([k1,0.,0.,0.,0.]))
    pixels=pixels.reshape(-1,2);jac=jac[:,:6].reshape(-1,2,6)
    feature,gradient=sample_with_gradient(grid,pixels) if contrast_grid is None else contrast_with_gradient(grid,contrast_grid,pixels)
    mass=np.ones(len(feature))/len(feature) if weights is None else np.asarray(weights,float)/np.sum(weights)
    diff=feature-target
    if profile_bias:diff=diff-np.sum(mass[:,None]*diff,axis=0)
    elif bias is not None:diff=diff-np.asarray(bias)
    squared=np.sum(diff*diff,axis=1)
    loss=.5*squared
    scale=np.ones(len(diff))
    if robust:
        scale=1/np.sqrt(1+squared/.25);loss=.25*(np.sqrt(1+squared/.25)-1)
    mass=np.ones(len(diff))/len(diff) if weights is None else np.asarray(weights,float)/np.sum(weights)
    motion=pixels-initial_pixels
    coefficient=diff*scale[:,None]
    if profile_bias:coefficient-=np.sum(mass[:,None]*coefficient,axis=0)
    pixel_grad=np.einsum('nd,ndk->nk',coefficient,gradient)+regularization*motion
    value=float(np.sum(mass*(loss+.5*regularization*np.sum(motion*motion,axis=1))))
    derivative=np.einsum('n,nk,nkj->j',mass,pixel_grad,jac)
    return value,derivative



def trust_constraints(x,camera0,K,k1,initial_pixels):
    pixels,jac=cv2.projectPoints(camera0,x[:3],x[3:],K,np.array([k1,0.,0.,0.,0.]))
    difference=pixels.reshape(-1,2)-initial_pixels;jac=jac[:,:6].reshape(-1,2,6)
    values=np.r_[.5**2-np.dot(x[3:],x[3:]),np.deg2rad(3)**2-np.dot(x[:3],x[:3]),16-np.sum(difference*difference,axis=1)]
    gradient=np.zeros((len(camera0)+2,6));gradient[0,3:]=-2*x[3:];gradient[1,:3]=-2*x[:3];gradient[2:]=-2*np.einsum('nk,nkj->nj',difference,jac)
    return values,gradient


def refine(initial,world,K,k1,grid,target,regularization=.02,weights=None,robust=False,constrained=False,box_slsqp=False,bias_mode="none",additional_grid=None,additional_target=None,contrast_grid=None):
    camera0=world@initial[:3,:3].T+initial[:3,3]
    target=np.asarray(target,float);target=target/np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-12)
    pix=cv2.projectPoints(camera0,np.zeros(3),np.zeros(3),K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2)
    if bias_mode not in ['none','frozen','profiled']:raise ValueError('unknown descriptor bias mode')
    if (additional_grid is None)!=(additional_target is None):raise ValueError('both additional field and target are required')
    bias=None
    if bias_mode=='frozen':
        diff=sample_with_gradient(grid,pix)[0]-target
        bias=np.average(diff,axis=0,weights=weights)
    def fn(x):
        value,gradient=objective(x,camera0,K,k1,grid,target,pix,regularization,weights,robust,bias,bias_mode=='profiled',contrast_grid)
        if additional_grid is not None:
            t=np.asarray(additional_target,float);t=t/np.maximum(np.linalg.norm(t,axis=1,keepdims=True),1e-12)
            other,g=objective(x,camera0,K,k1,additional_grid,t,pix,regularization,weights,robust)
            value=(value+other)/2;gradient=(gradient+g)/2
        return value,gradient
    before=fn(np.zeros(6))[0]
    bounds=[(-np.deg2rad(3),np.deg2rad(3))]*3+[(-.5,.5)]*3
    if constrained or box_slsqp:
        constraint=dict(type='ineq',fun=lambda x:trust_constraints(x,camera0,K,k1,pix)[0],jac=lambda x:trust_constraints(x,camera0,K,k1,pix)[1])
        result=minimize(fn,np.zeros(6),jac=True,method='SLSQP',bounds=bounds,constraints=constraint if constrained else (),options=dict(maxiter=100,ftol=1e-10))
    else:
        result=minimize(fn,np.zeros(6),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(maxiter=80,ftol=1e-11,gtol=1e-7,maxls=25))
    x=result.x;R=cv2.Rodrigues(x[:3])[0];proposal=initial.copy();proposal[:3,:3]=R@initial[:3,:3];proposal[:3,3]=R@initial[:3,3]+x[3:]
    projected=cv2.projectPoints(camera0,x[:3],x[3:],K,np.array([k1,0.,0.,0.,0.]))[0].reshape(-1,2)
    positive=((camera0@R.T+x[3:])[:,2]>0).all();inside=((projected>=0)&(projected<=np.array([255,143]))).all()
    tolerance=1e-6 if constrained or box_slsqp else 0.
    accepted=bool(result.success and np.isfinite(x).all() and positive and inside and np.linalg.norm(x[3:])<=.5+tolerance and np.linalg.norm(x[:3])<=np.deg2rad(3)+tolerance and result.fun<before-1e-10 and np.max(np.linalg.norm(projected-pix,axis=1))<=4+tolerance)
    return proposal if accepted else initial.copy(),dict(accepted=accepted,bias_mode=bias_mode,multiscale=additional_grid is not None,contrast=contrast_grid is not None,constrained_optimizer=constrained,box_slsqp=box_slsqp,optimizer_success=bool(result.success),iterations=int(result.nit),initial_objective=before,final_objective=float(result.fun),translation_step_m=float(np.linalg.norm(x[3:])),rotation_step_deg=float(np.rad2deg(np.linalg.norm(x[:3]))),max_image_motion_px=float(np.max(np.linalg.norm(projected-pix,axis=1))),minimum_support_fixed=True)
