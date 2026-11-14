"""圆轨三通道子孔径匀速物理算子。距离用float64，最终回波用complex64。

B=batch，T=帧，C=通道，F=频率，N=慢时间，P=像素；M=F*N。
归一化每个通道每帧原子能量为1。理想配准后通道延迟作用于目标运动时间。
仅雷达参数进入本模块；目标真值仅由simulate.py生成。
"""
import math
import torch
from torch import nn

LIGHT_SPEED = 299792458.0


def los(position, platform, vp, ap):
    """输入地面位置[...,2]；返回R,n,P及静止点曲率，单位m和m/s²。"""
    q = torch.cat((position.double(), torch.zeros_like(position[..., :1]).double()), -1)
    delta = q-platform
    distance = torch.linalg.vector_norm(delta, dim=-1)
    n = delta/distance[..., None]
    projection = torch.eye(3, device=q.device, dtype=q.dtype)-n[..., :, None]*n[..., None, :]
    static = torch.einsum('...i,...ij,...j->...', vp, projection, vp)/distance-(n*ap).sum(-1)
    return distance, n, projection, static


class RadarPhysics(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        r = config['radar']
        self.channels, self.frames = r['channels'], r['frames']
        self.radius, self.height, self.speed = r['radius_m'], r['height_m'], r['speed_m_s']
        self.norm = math.sqrt(r['frequency_samples']*r['pulses'])
        self.reference = math.hypot(self.radius, self.height)
        self.ati_scale = 4*math.pi*r['carrier_hz']/LIGHT_SPEED*r['baseline_m']/self.speed
        self.chunk = config['model']['operator_chunk']
        theta0 = math.radians(r['initial_angle_deg'])
        centers = (torch.arange(r['frames'],dtype=torch.float64)-(r['frames']-1)/2)*r['frame_interval_s']
        tau = (torch.arange(r['pulses'],dtype=torch.float64)-(r['pulses']-1)/2)*r['aperture_s']/r['pulses']
        self.register_buffer('centers',centers)
        self.register_buffer('tau',tau)
        self.register_buffer('frequency',torch.linspace(r['carrier_hz']-r['bandwidth_hz']/2,r['carrier_hz']+r['bandwidth_hz']/2,r['frequency_samples'],dtype=torch.float64))
        self.register_buffer('delay',torch.arange(r['channels'],dtype=torch.float64)*r['baseline_m']/self.speed)
        # u为轨道切向，q为地面径向，列为局部坐标基。
        self.register_buffer('basis',torch.tensor([[-math.sin(theta0), math.cos(theta0)],[math.cos(theta0),math.sin(theta0)]],dtype=torch.float64))
        def state(times):
            theta = theta0+self.speed/self.radius*times
            er = torch.stack((theta.cos(),theta.sin(),torch.zeros_like(theta)),-1)
            et = torch.stack((-theta.sin(),theta.cos(),torch.zeros_like(theta)),-1)
            p = self.radius*er+torch.tensor([0.,0.,self.height])
            return p,self.speed*et,-self.speed**2/self.radius*er
        p,v,a = state(centers)
        self.register_buffer('platform',p)
        self.register_buffer('vp',v)
        self.register_buffer('ap',a)
        self.register_buffer('pulse_platform',state(centers[:,None]+tau)[0])

    def grid(self, shape, half):
        u = torch.linspace(-half[0],half[0],shape[0],dtype=torch.float64,device=self.tau.device)
        q = torch.linspace(-half[1],half[1],shape[1],dtype=torch.float64,device=self.tau.device)
        qq,uu = torch.meshgrid(q,u,indexing='ij')
        return torch.stack((uu.flatten(),qq.flatten()),-1)@self.basis.T

    def static_matrix(self, points):
        """静止场景字典[T,M,P]；通道理想配准后共用，不假设任意背景低秩。"""
        matrix = torch.empty(self.frames, self.frequency.numel()*self.tau.numel(),
                             len(points), device=points.device, dtype=torch.complex64)
        offset = 0
        for block in points.split(self.chunk):
            delta = self.pulse_platform[:,:,None,:2]-block[None,None,:,:]
            distance = (delta.square().sum(-1)+self.height**2).sqrt()
            phase = -4*math.pi/LIGHT_SPEED*self.frequency[None,:,None,None]*(distance[:,None]-self.reference)
            matrix[:,:,offset:offset+len(block)] = torch.polar(torch.ones_like(phase),phase).to(torch.complex64).flatten(1,2)/self.norm
            offset += len(block)
        return matrix

    def moving_matrix(self, points, velocity):
        """points[B,T,P,2],velocity[B,T,2] -> [B,T,C,M,P]。

        仅使用匀速精确距离，不增加加速度或二阶相位误差参数。
        """
        dt = self.tau[None,:]-self.delay[:,None] # [C,N]
        blocks = []
        center = points.mean(2)
        for block in points.split(self.chunk,dim=2):
            target = block[:,:,None,None,:,:]+dt[None,None,:,:,None,None]*velocity[:,:,None,None,None,:].double()
            delta = target-self.pulse_platform[None,:,None,:,None,:2]
            distance = (delta.square().sum(-1)+self.height**2).sqrt()
            phase = -4*math.pi/LIGHT_SPEED*self.frequency[None,None,None,:,None,None]*(distance[:,:,:,None]-self.reference)
            blocks.append(torch.polar(torch.ones_like(phase),phase).to(torch.complex64).flatten(3,4)/self.norm)
        return torch.cat(blocks,-1)

    def atom_jacobian(self, position, velocity, frame_indices=None):
        """CV atom and local [u,q,vu,vq] derivatives; optional indexed windows.

        Default states: [B,T,2]. Indexed states: [B,...,2], with frame_indices
        matching the state frame dimensions. The final observation axis is C*F*N.
        """
        dt = self.tau[None,:]-self.delay[:,None]
        platform = self.pulse_platform if frame_indices is None else self.pulse_platform[frame_indices]
        target = position[...,None,None,:]+dt[...,None]*velocity[...,None,None,:]
        delta = target-platform[None,...,None,:,:2]
        distance = (delta.square().sum(-1)+self.height**2).sqrt()
        frequency = self.frequency.reshape(*([1]*(position.ndim-1)),1,-1,1)
        phase = -4*math.pi/LIGHT_SPEED*frequency*(distance.unsqueeze(-2)-self.reference)
        atom = torch.polar(torch.ones_like(phase), phase)/self.norm
        line = (delta/distance[...,None])@self.basis
        dr = torch.cat((line,line*dt[...,None]),-1)
        jac = atom[...,None]*(-4j*math.pi/LIGHT_SPEED)*frequency[...,None]*dr.unsqueeze(-3)
        return atom.flatten(position.ndim-1), jac.flatten(position.ndim-1,-2)
