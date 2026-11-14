"""Joint spatial/temporal unfolding with frame streaming and recomputation."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .physics import RadarPhysics,los
from .frontend import DecompositionFrontEnd
from .temporal import TemporalRefiner

class JointUnfoldNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.radar = RadarPhysics(cfg)
        m = cfg['model']
        self.log_damping = nn.Parameter(torch.full((m['stages'],4), math.log(m['initial_damping']), dtype=torch.float64))
        self.step_logit = nn.Parameter(torch.full((m['stages'],4), math.log(m['initial_step']/(1-m['initial_step'])), dtype=torch.float64))
        self.register_buffer('max_update', torch.tensor(m['max_update'], dtype=torch.float64))
        self.frontend = DecompositionFrontEnd(cfg)
        self.temporal = TemporalRefiner(cfg)

    def _single_step(self,position,velocity,y,k):
        a,da=self.radar.atom_jacobian(position,velocity)
        norm=a.abs().square().sum(-1,keepdim=True)
        amplitude=(a.conj()*y).sum(-1,keepdim=True)/norm
        residual=y-amplitude*a
        projected=da-a[...,None]*(a.conj()[...,None]*da).sum(-2,keepdim=True)/norm[...,None]
        jac=amplitude[...,None]*projected
        h=(jac.mH@jac).real;rhs=(jac.conj()*residual[...,None]).sum(-2).real
        scale=h.diagonal(dim1=-2,dim2=-1).clamp_min(1e-12).sqrt()
        solve=torch.linalg.solve(h/(scale[...,None]*scale[...,None,:])+torch.diag(self.log_damping[k].clamp(-16,4).exp()),(rhs/scale)[...,None]).squeeze(-1)/scale
        update=self.max_update*torch.tanh(solve/self.max_update)*self.step_logit[k].sigmoid()
        return position+update[...,:2]@self.radar.basis.T,velocity+update[...,2:]@self.radar.basis.T

    def forward(self,echo):
        r=self.cfg['radar']
        if tuple(echo.shape[1:])!=(r['frames'],3,r['frequency_samples'],r['pulses']) or not echo.is_complex():raise ValueError('Expected complex [B,T,3,F,N]')
        front=self.frontend(echo,self.radar)
        p,v=front['position'],front['velocity'];y=front['target_echo'].flatten(2).to(torch.complex128)
        for stage in range(self.cfg['model']['stages']):
            if torch.is_grad_enabled():p,v=checkpoint(self._single_step,p,v,y,stage,use_reentrant=False)
            else:p,v=self._single_step(p,v,y,stage)
        single_p,single_v=p,v
        p,v=self.temporal(p,v,front['target_echo'],self.radar)
        a,_=self.radar.atom_jacobian(p,v)
        amplitude=(a.conj()*y).sum(-1,keepdim=True)/a.abs().square().sum(-1,keepdim=True)
        reconstruction=(amplitude*a).reshape_as(echo)
        _,n,_,_=los(p,self.radar.platform,self.radar.vp,self.radar.ap)
        return dict(single_position=single_p,single_velocity=single_v,position=p,velocity=v,
            radial=(n[...,:2]*v).sum(-1),ati_radial=front['radial'],initial_position=front['position'],
            apparent_position=front['apparent_position'],target_echo=front['target_echo'],
            front_reconstruction=front['reconstruction'],reconstruction=reconstruction+front['background_echo'],
            target_reconstruction=reconstruction,amplitude=amplitude.squeeze(-1),
            front_X1=front['X1'],front_dX1=front['dX1'],front_P=front['P'])
