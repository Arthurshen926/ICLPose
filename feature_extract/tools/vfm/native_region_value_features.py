"""Pose-free features for adding one region to a fixed eight-region set."""
import numpy as np


def addition_features(base_tokens,base_scores,tokens,scores,context,base_context,center,base_centers):
 base_tokens=np.unique(base_tokens);tokens=np.unique(tokens);scores=np.asarray(scores);base_scores=np.asarray(base_scores)
 return np.array([len(scores)/1024,len(tokens)/2304,len(np.setdiff1d(tokens,base_tokens))/2304,len(base_tokens)/2304,float(scores.mean()) if len(scores) else 0.,float(scores.max()) if len(scores) else 0.,float(base_scores.mean()) if len(base_scores) else 0.,float(np.max(context)),float(np.mean(context)),float(np.maximum(context-np.max(base_context,axis=0),0).sum()),float(np.min(np.linalg.norm(center-base_centers,axis=1)))/6.,float(np.std(tokens%64))/64 if len(tokens) else 0.,float(np.std(tokens//64))/36 if len(tokens) else 0.])
