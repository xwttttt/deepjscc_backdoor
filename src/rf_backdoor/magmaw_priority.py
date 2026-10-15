"""Primary-objective gradient protection for complete universal attacks."""
import numpy as np


def project_auxiliary_np(primary, auxiliary, fraction):
    primary=np.asarray(primary,dtype=float)
    auxiliary=np.asarray(auxiliary,dtype=float)
    if primary.shape!=auxiliary.shape or not np.all(np.isfinite(primary)) or not np.all(np.isfinite(auxiliary)):
        raise ValueError('finite gradients of matching shape required')
    if not np.isfinite(fraction) or fraction<0: raise ValueError('nonnegative finite gradient fraction required')
    norm2=float(np.sum(primary**2))
    if norm2<=1e-30: return np.zeros_like(auxiliary)
    corrected=auxiliary-min(float(np.sum(primary*auxiliary)),0.)/norm2*primary
    norm=float(np.linalg.norm(corrected))
    return corrected*min(1.,fraction*np.sqrt(norm2)/max(norm,1e-30))


def protected_gradients(primary, auxiliary, fraction):
    """Remove opposing component, cap auxiliary norm, never amplify it.

    Applied before Adam. Nonnegative raw-gradient dot product is a local
    property, not a claim that nonlinear validation PSNR must improve.
    """
    import torch
    if not np.isfinite(fraction) or fraction<0: raise ValueError('invalid gradient fraction')
    dot=sum((p*a).sum() for p,a in zip(primary,auxiliary))
    norm2=sum(p.square().sum() for p in primary)
    coefficient=dot.clamp(max=0)/norm2.clamp_min(1e-30)
    corrected=[a-coefficient*p for p,a in zip(primary,auxiliary)]
    norm=sum(a.square().sum() for a in corrected).sqrt()
    scale=(fraction*norm2.sqrt()/norm.clamp_min(1e-30)).clamp(max=1)
    scale=torch.where(norm2>1e-30,scale,torch.zeros_like(scale))
    corrected=[a*scale for a in corrected]
    diagnostics={'original_dot':dot.detach(),'projected_dot':sum((p*a).sum() for p,a in zip(primary,corrected)).detach(),
                 'primary_l2':norm2.sqrt().detach(),'applied_aux_l2':(norm*scale).detach()}
    return corrected,diagnostics
