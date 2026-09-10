"""Couple coordinate errors without forcing irreducible map-height error to zero."""
import torch


def joint_error(image_mean,image_target,uv_mean,uv_target,anchor_camera,tangent_camera,K,k1,lower,upper):
    uv=torch.minimum(torch.maximum(uv_mean,lower),upper)
    def project(offset):
        camera=anchor_camera+torch.einsum('nij,nj->ni',tangent_camera,offset)
        xy=camera[:,:2]/camera[:,2:].clamp_min(1e-6)
        xy=xy*(1+k1[:,None]*(xy*xy).sum(dim=1,keepdim=True))
        return xy*torch.stack([K[:,0,0],K[:,1,1]],dim=1)+K[:,:2,2]
    # The target projection is on the SAME anchor tangent sheet. Its offset
    # from the mapping query world point is nuisance geometry, not zero noise.
    return image_mean-image_target-(project(uv)-project(uv_target))
