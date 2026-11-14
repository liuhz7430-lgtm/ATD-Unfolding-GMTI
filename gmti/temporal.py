"""Offline sliding-window Gauss–Newton refinement with robust displacement constraints.

Each frame has its own position and velocity. No constant-velocity trajectory or
penalty forcing adjacent velocities to be equal is imposed. All windows start
from the same single-frame outputs; one designated estimate is kept per frame.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def window_layout(frames, window, device):
    indices=torch.arange(frames-window+1,device=device)[:,None]+torch.arange(window,device=device)
    output_window=(torch.arange(frames,device=device)-window//2).clamp(0,frames-window)
    output_offset=torch.arange(frames,device=device)-output_window
    return indices,output_window,output_offset


def motion_system(position,velocity,times,basis,scale):
    """Normalized trapezoidal displacement residual and exact local-state Jacobian."""
    width=position.shape[-2]
    dt=times[...,1:]-times[...,:-1]
    residual=((position[...,1:,:]-position[...,:-1,:])-
              .5*dt[...,None]*(velocity[...,1:,:]+velocity[...,:-1,:]))@basis/scale
    jac=torch.zeros(*times.shape[:-1],width-1,2,width,4,device=position.device,dtype=position.dtype)
    for edge in range(width-1):
        for axis in range(2):
            jac[...,edge,axis,edge,axis]=-1/scale
            jac[...,edge,axis,edge+1,axis]=1/scale
            jac[...,edge,axis,edge,axis+2]=-.5*dt[...,edge]/scale
            jac[...,edge,axis,edge+1,axis+2]=-.5*dt[...,edge]/scale
    return residual,jac.reshape(*times.shape[:-1],2*(width-1),4*width)


class TemporalRefiner(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.settings=cfg['temporal']
        m=self.settings
        self.batch=cfg['wide']['window_batch']
        self.log_damping=nn.Parameter(torch.full((m['stages'],4),math.log(m['initial_damping']),dtype=torch.float64))
        self.step_logit=nn.Parameter(torch.full((m['stages'],4),math.log(m['initial_step']/(1-m['initial_step'])),dtype=torch.float64))
        self.register_buffer('max_update',torch.tensor(m['max_update'],dtype=torch.float64))

    def _step(self,position,velocity,y,radar,indices,stage):
        a,da=radar.atom_jacobian(position,velocity,frame_indices=indices)
        norm=a.abs().square().sum(-1,keepdim=True).clamp_min(1e-12)
        amplitude=(a.conj()*y).sum(-1,keepdim=True)/norm
        residual=y-amplitude*a
        projected=da-a[...,None]*(a.conj()[...,None]*da).sum(-2,keepdim=True)/norm[...,None]
        jac=amplitude[...,None]*projected
        # Per-frame relative echo loss: independent of target amplitude/echo length.
        energy=y.abs().square().sum(-1).clamp_min(1e-10)
        h=(jac.mH@jac).real/energy[...,None,None]
        rhs=(jac.conj()*residual[...,None]).sum(-2).real/energy[...,None]
        width=position.shape[-2]
        eye=torch.eye(width,device=h.device,dtype=h.dtype)
        h=torch.einsum('...wij,wv->...wivj',h,eye).reshape(*h.shape[:-3],4*width,4*width)
        rhs=rhs.flatten(-2)
        weight=self.settings['motion_weight']
        if width>1 and weight>0:
            e,j=motion_system(position,velocity,radar.centers[indices],radar.basis,self.settings['motion_scale_m'])
            # Vector pseudo-Huber IRLS: large motion residuals exert bounded force.
            robust=torch.rsqrt(1+e.square().sum(-1))
            row_weight=(weight*robust)[...,None].expand(*robust.shape,2).flatten(-2)
            h=h+j.transpose(-1,-2)@(row_weight[...,None]*j)
            rhs=rhs-(j.transpose(-1,-2)@(row_weight*e.flatten(-2))[...,None]).squeeze(-1)
        scale=h.diagonal(dim1=-2,dim2=-1).clamp_min(1e-12).sqrt()
        normalized=h/(scale[...,None]*scale[...,None,:])
        damping=self.log_damping[stage].clamp(-16,4).exp().repeat(width)
        solve=torch.linalg.solve(normalized+torch.diag(damping),(rhs/scale)[...,None]).squeeze(-1)/scale
        solve=solve.reshape(*position.shape[:-1],4)
        update=self.max_update*torch.tanh(solve/self.max_update)*self.step_logit[stage].sigmoid()
        return position+update[...,:2]@radar.basis.T,velocity+update[...,2:]@radar.basis.T

    def forward(self,position,velocity,target_echo,radar):
        if not self.settings['enabled'] or self.settings['stages']==0:return position,velocity
        indices,window,offset=window_layout(position.shape[1],self.settings['window'],position.device)
        outputs=[]
        # Each short-window solve is unchanged; only its execution batch is bounded.
        for start in range(0,len(indices),self.batch):
            selected=indices[start:start+self.batch]
            p,v=position[:,selected],velocity[:,selected]
            y=target_echo.flatten(2).to(torch.complex128)[:,selected]
            for stage in range(self.settings['stages']):
                if torch.is_grad_enabled():p,v=checkpoint(self._step,p,v,y,radar,selected,stage,use_reentrant=False)
                else:p,v=self._step(p,v,y,radar,selected,stage)
            outputs.append((p,v))
        p=torch.cat([row[0] for row in outputs],1);v=torch.cat([row[1] for row in outputs],1)
        return p[:,window,offset],v[:,window,offset]
