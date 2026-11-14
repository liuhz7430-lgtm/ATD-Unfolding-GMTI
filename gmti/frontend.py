"""Differentiable three-channel decomposition with cached geometry-only scales."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .physics import los
from .config import support
from .operators import DictionaryView,FrameRadar,adjoint_product

def forward_images(x, d, p):
    return torch.stack((x, x+d, x+d*(1+p)), dim=2)

class DecompositionFrontEnd(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        m=cfg['model']; k=m['front_stages']
        self.x_step=nn.Parameter(torch.full((k,),math.log(m['front_step']/(1-m['front_step']))))
        self.d_step=nn.Parameter(self.x_step.detach().clone())
        self.p_step=nn.Parameter(torch.full((k,),math.log(m['phase_step']/(1-m['phase_step']))))
        self.log_threshold=nn.Parameter(torch.full((k,),math.log(m['support_threshold'])))
        self._cache=None
        self._frame_key=None
        self._scales={}
        self.scale_builds=0
        self.matrix_builds=0
        self._support_device=None

    def clear(self):
        self._cache=None
        self._frame_key=None

    def dictionary(self,radar):
        key=(radar.index,radar.tau.device)
        if key!=self._frame_key:
            self.clear()
            with torch.no_grad():
                if self._support_device!=radar.tau.device:
                    uq,mask=support(self.cfg)
                    self._points=torch.as_tensor(uq,device=radar.tau.device)@radar.basis.T
                    self._target_mask=torch.as_tensor(mask,device=radar.tau.device)
                    self._support_device=radar.tau.device
                a=radar.static_matrix(self._points);self.matrix_builds+=1
                if key not in self._scales:
                    frame=a[0];z=torch.ones(len(self._points),dtype=a.dtype,device=a.device)
                    for _ in range(12):
                        z=adjoint_product(frame,frame@z)
                        z=z/torch.linalg.vector_norm(z).clamp_min(1e-12)
                    self._scales[key]=(1.2*(frame@z).abs().square().sum().clamp_min(1e-6)).reshape(1)
                    self.scale_builds+=1
                self._cache=(DictionaryView(a),self._points,self._scales[key])
                self._frame_key=key
        return self._cache

    def _decompose(self,echo,radar):
        return decompose(self,echo,radar)

    def _frame(self,echo,radar,index):
        result=self._decompose(echo,FrameRadar(radar,index))
        return tuple(result[key] for key in self.keys)
    def forward(self,echo,radar):
        self.keys=('X1','dX1','P','position','velocity','radial','apparent_position','target_echo','background_echo','reconstruction')
        frames=[]
        for index in range(radar.frames):
            if torch.is_grad_enabled():
                values=checkpoint(self._frame,echo[:,index:index+1],radar,index,use_reentrant=False)
            else:values=self._frame(echo[:,index:index+1],radar,index)
            frames.append(values)
        result={key:torch.cat([row[j] for row in frames],dim=1) for j,key in enumerate(self.keys)}
        self.clear()
        return result

def decompose(self,echo,radar):
    a,points,L=self.dictionary(radar)
    def A(x):
        return torch.stack([(a[t]@x[:,t].transpose(-1,-2)).transpose(-1,-2) for t in range(a.shape[0])],dim=1)
    def AH(y):
        return torch.stack([(a[t].mH@y[:,t].transpose(-1,-2)).transpose(-1,-2) for t in range(a.shape[0])],dim=1)
    data=echo.flatten(-2).to(a.dtype)
    bp=AH(data)
    x=bp[:,:,0]/L[None,:,None]
    d=(bp[:,:,1]-bp[:,:,0])/L[None,:,None]
    cross=(bp[:,:,2]-bp[:,:,1])*(bp[:,:,1]-bp[:,:,0]).conj()
    # Smooth normalization avoids undefined angle/gradient at zero signal.
    phi=torch.atan2(cross.imag,cross.real+1e-10)
    p=torch.polar(torch.ones_like(phi),phi)
    # Only the initial discrete candidate is fixed. All subsequent estimates retain gradients.
    score=(bp[:,:,1]-bp[:,:,0]).abs().square()
    index=score.masked_fill(~self._target_mask[None,None],-torch.inf).argmax(-1)
    center=points[index]
    local=(points[None,None]-center[:,:,None])@radar.basis
    half=torch.tensor(self.cfg['model']['roi_half_uq_m'],device=echo.device)
    roi=torch.exp(-.5*(local/half).square().sum(-1)).to(data.real.dtype)
    temperature=self.cfg['model']['support_temperature']
    for k in range(len(self.x_step)):
        residual=A(forward_images(x,d,p))-data
        x=x-self.x_step[k].sigmoid()*AH(residual).sum(2)/(3*L[None,:,None])
        residual=A(forward_images(x,d,p))-data
        g=AH(residual)[:,:,2]*d.conj()
        phase_gradient=(p.conj()*g).imag
        denom=L[None,:,None]*d.abs().square()+1e-5
        phi=torch.atan2(p.imag,p.real)-self.p_step[k].sigmoid()*torch.tanh(phase_gradient/denom)
        candidate=torch.polar(torch.ones_like(phi),phi)
        # P=1 outside soft support; enforce unit magnitude exactly.
        gate=torch.sigmoid(((candidate-1).abs()-self.log_threshold[k].exp())/temperature)
        phi=gate*phi
        p=torch.polar(torch.ones_like(phi),phi)
        back=AH(A(forward_images(x,d,p))-data)
        gradient=back[:,:,1]+(1+p).conj()*back[:,:,2]
        d=gate*(d-self.d_step[k].sigmoid()*gradient/(5*L[None,:,None]))
    q=p-1
    dynamic=d*q.conj()/(q.abs().square()+self.cfg['model']['recovery_epsilon'])
    # Soft single-candidate extraction; outside the candidate remains in the static estimate.
    dynamic=dynamic*roi
    static=x-dynamic
    background=A(static[:,:,None]).expand(-1,-1,radar.channels,-1)
    target=(data-background).reshape_as(echo)
    weight=roi*dynamic.abs().square()+1e-12*roi
    weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-12)
    apparent=(weight.double()[...,None]*points).sum(-2)
    mean_phase=(weight*p).sum(-1)
    radial=torch.atan2(mean_phase.imag,mean_phase.real+1e-10).double()/radar.ati_scale
    pp,vp,ap=radar.platform,radar.vp,radar.ap
    rr,n,_,_=los(apparent,pp,vp,ap)
    rate=-(n*vp).sum(-1)
    position=apparent+vp[None,:,:2]/radar.speed*(radial*rr/radar.speed)[...,None]
    for _ in range(6):
        distance,n,proj,_=los(position,pp,vp,ap)
        error=torch.stack((distance-rr,(-(n*vp).sum(-1)+radial-rate)*rr/radar.speed),-1)
        row=-torch.einsum('...i,...ij->...j',vp,proj)[...,:2]/distance[...,None]*rr[...,None]/radar.speed
        jac=torch.stack((n[...,:2],row),-2)
        position=position-torch.linalg.solve(jac,error[...,None]).squeeze(-1).clamp(-50,50)
    _,n,_,_=los(position,pp,vp,ap)
    velocity=radial[...,None]*n[...,:2]/n[...,:2].square().sum(-1,keepdim=True)
    return {'X1':x,'dX1':d,'P':p,'position':position,'velocity':velocity,'radial':radial,
            'apparent_position':apparent,'target_echo':target,'background_echo':background.reshape_as(echo),
            'reconstruction':A(forward_images(x,d,p)).reshape_as(echo)}
