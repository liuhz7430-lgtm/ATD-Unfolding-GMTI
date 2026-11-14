"""Predetermined frame refocusing; truth is used only for plot markers."""
import hashlib
import math
import numpy as np
import torch
from .physics import LIGHT_SPEED
from .artifacts import write_json

def example_ids(entries,count=None):
    """One SCR closest to 0 dB for EACH trajectory, without error ranking."""
    groups={}
    for i,entry in enumerate(entries):
        if entry.get('target_present',True):
            groups.setdefault((entry['mode'],entry['trajectory']),[]).append((i,entry))
    selected=[min(groups[key],key=lambda row:(abs(row[1]['scr_db']),row[0]))[0]
        for key in sorted(groups)]
    return selected if count is None else selected[:count]

@torch.no_grad()
def save_focus(model,sample,pred,index,output,frame_count=3):
    r=model.radar;device=r.tau.device
    folder=output/'data/focus'/f'sample_{index:04d}';folder.mkdir(parents=True,exist_ok=True)
    frames=np.unique(np.linspace(0,r.frames-1,min(frame_count,r.frames),dtype=int))
    az=LIGHT_SPEED/float(r.frequency.mean())*r.reference/(2*r.speed*model.cfg['radar']['aperture_s'])
    rg=LIGHT_SPEED/(2*model.cfg['radar']['bandwidth_hz'])*r.reference/r.radius
    # Equal physical window size and sampling before/after. Each window is
    # centered on its corresponding ESTIMATE; truth is used only for markers.
    uoff=torch.linspace(-2*az,2*az,81,device=device,dtype=torch.float64)
    qoff=torch.linspace(-3*rg,3*rg,65,device=device,dtype=torch.float64)
    qq,uu=torch.meshgrid(qoff,uoff,indexing='ij')
    offsets=torch.stack((uu.flatten(),qq.flatten()),-1)
    record=dict(sample_id=index,info=sample['info'],frames=frames.tolist(),
        selected_without_error_ranking=True,window_half_uq_m=[2*az,3*rg],grid_shape_qu=[65,81],
        observation='Same frontend target residual, receive channel 0, before and after',
        compensation='Before: zero velocity. After: trained estimated velocity. Truth only marks the plots.',
        display='Separate estimate-centered local windows of equal physical size; shared power reference per frame.',
        failures=[])
    for k in frames:
        center=torch.stack((pred['apparent_position'][0,k],pred['position'][0,k]))@r.basis
        velocity=pred['velocity'][0,k]
        path=folder/f'frame_{k:03d}.npz'
        if not torch.isfinite(center).all() or not torch.isfinite(velocity).all():
            record['failures'].append(int(k));continue
        data=pred['target_echo'][0,k,0].flatten().to(torch.complex128)
        results=[]
        for j,v in enumerate((torch.zeros_like(velocity),velocity)):
            points=(offsets+center[j])@r.basis.T
            values=[]
            for block in points.split(64):
                target=block[None]+r.tau[:,None,None]*v
                delta=target-r.pulse_platform[k,:,None,:2]
                distance=(delta.square().sum(-1)+r.height**2).sqrt()
                phase=-4*math.pi/LIGHT_SPEED*r.frequency[:,None,None]*(distance[None]-r.reference)
                a=(torch.polar(torch.ones_like(phase),phase)/r.norm).flatten(0,1)
                values.append((a.T@data.conj()).conj())
            results.append(torch.cat(values).reshape(qq.shape).cpu().numpy())
        np.savez_compressed(path,before=results[0],after=results[1],
            u_before=(uoff+center[0,0]).cpu().numpy(),q_before=(qoff+center[0,1]).cpu().numpy(),
            u_after=(uoff+center[1,0]).cpu().numpy(),q_after=(qoff+center[1,1]).cpu().numpy(),
            centers_uq=center.cpu().numpy(),truth_uq=sample['truth']['position'][k].numpy()@r.basis.cpu().numpy(),
            estimated_velocity_xy=velocity.cpu().numpy(),frame=int(k),time_s=float(r.centers[k]),
            target_residual_sha256=hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest())
        print(f'Focus sample {index}, frame {k+1}/{r.frames}',flush=True)
    write_json(folder/'selection.json',record)
