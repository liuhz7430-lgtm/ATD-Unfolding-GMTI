"""Deterministic synthetic data, bounded trajectories, and numeric echo caches."""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from torch.utils.data import Dataset
from .physics import LIGHT_SPEED
from .config import PROJECT
SPLIT_OFFSET={'train':0,'validation':10000000,'test':20000000}

def sample_trajectory(rng, cfg):
    """Sample physical states; reject whole trajectories before generating any echoes.

    The optional radial-speed floor applies at every frame center, with one
    consistent sign across the window. Zero preserves the original sampling.
    """
    r, s = cfg['radar'], cfg['scene']
    t = (np.arange(r['frames'])-(r['frames']-1)/2)*r['frame_interval_s']
    theta0 = np.deg2rad(r['initial_angle_deg'])
    basis = np.array([[-np.sin(theta0),np.cos(theta0)],[np.cos(theta0),np.sin(theta0)]])
    theta = theta0+r['speed_m_s']/r['radius_m']*t
    platform = r['radius_m']*np.stack((np.cos(theta),np.sin(theta)), -1)
    minimum = s.get('min_abs_radial_velocity_m_s', 0.0)
    for _ in range(10000):
        p0 = (rng.uniform(-1,1,2)*s['position_half_uq_m'])@basis.T
        v0 = rng.uniform(-1,1,2)*s['velocity_half_xy_m_s']
        acceleration = rng.uniform(-1,1,2)*s['acceleration_half_xy_m_s2']
        position = p0+t[:,None]*v0+.5*t[:,None]**2*acceleration
        velocity = v0+t[:,None]*acceleration
        delta = position-platform
        radial = (delta*velocity).sum(-1)/np.sqrt((delta**2).sum(-1)+r['height_m']**2)
        if minimum == 0 or np.all(radial >= minimum) or np.all(radial <= -minimum):
            return position, velocity, radial
    raise ValueError('Cannot sample a trajectory satisfying min_abs_radial_velocity_m_s '
                     'within 10000 attempts; relax the floor or the motion ranges')

class SyntheticEchoDataset(Dataset):
    def __init__(self, cfg, split):
        self.cfg, self.split = cfg, split
        self.size = cfg['training'][split+'_samples']
    def __len__(self): return self.size
    def __getitem__(self, index):
        cfg, r, s = self.cfg, self.cfg['radar'], self.cfg['scene']
        rng = np.random.default_rng(cfg['seed']+SPLIT_OFFSET[self.split]+index)
        t = (np.arange(r['frames'])-(r['frames']-1)/2)*r['frame_interval_s']
        tau = (np.arange(r['pulses'])-(r['pulses']-1)/2)*r['aperture_s']/r['pulses']
        theta0 = np.deg2rad(r['initial_angle_deg'])
        theta = theta0+r['speed_m_s']/r['radius_m']*(t[:,None]+tau)
        platform = r['radius_m']*np.stack((np.cos(theta), np.sin(theta)), -1)
        basis = np.array([[-np.sin(theta0),np.cos(theta0)],[np.cos(theta0),np.sin(theta0)]])
        position, velocity, radial = sample_trajectory(rng, cfg)
        amplitude = rng.uniform(*s['amplitude'])*np.exp(1j*rng.uniform(-np.pi,np.pi))
        freq = np.linspace(r['carrier_hz']-r['bandwidth_hz']/2,r['carrier_hz']+r['bandwidth_hz']/2,r['frequency_samples'])
        norm = np.sqrt(r['frequency_samples']*r['pulses'])
        target = []
        for channel in range(r['channels']):
            dt = tau-channel*r['baseline_m']/r['speed_m_s']
            pos = position[:,None,:]+dt[None,:,None]*velocity[:,None,:]
            distance = np.sqrt(((pos-platform)**2).sum(-1)+r['height_m']**2)
            target.append(amplitude*np.exp(-4j*np.pi/LIGHT_SPEED*freq[None,:,None]*(distance[:,None]-np.hypot(r['radius_m'],r['height_m'])))/norm)
        clean = np.stack(target, axis=1)
        background = np.zeros_like(clean)
        count = s['clutter_points']
        if count:
            level = abs(amplitude)*10**(rng.uniform(*s['clutter_db'])/20)/np.sqrt(count)
            for _ in range(count):
                point = (rng.uniform(-1,1,2)*s['clutter_half_uq_m'])@basis.T
                coefficient = level*np.exp(1j*rng.uniform(-np.pi,np.pi))
                distance = np.sqrt(((point-platform)**2).sum(-1)+r['height_m']**2)
                signal = coefficient*np.exp(-4j*np.pi/LIGHT_SPEED*freq[None,:,None]*(distance[:,None]-np.hypot(r['radius_m'],r['height_m'])))/norm
                background += signal[:,None]
        snr = rng.uniform(*s['snr_db'])
        noise_power = abs(amplitude)**2/(norm**2*10**(snr/10))
        noise = np.sqrt(noise_power/2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))
        return {'echo':torch.from_numpy((clean+background+noise).astype(np.complex64)), 'truth':{
            'position':torch.tensor(position), 'velocity':torch.tensor(velocity),
            'clean':torch.from_numpy(clean.astype(np.complex64)),
            'background':torch.from_numpy(background.astype(np.complex64)),
            'radial':torch.tensor(radial), 'sample_id':torch.tensor(index),
            'snr_db':torch.tensor(snr), 'amplitude':torch.tensor(amplitude)}}

class BoundedEchoDataset:
    def __init__(self,cfg,split):self.cfg=cfg;self.split=split
    def __len__(self):return self.cfg['training'][self.split+'_samples']
    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        c=self.cfg;r=c['radar'];theta=np.deg2rad(r['initial_angle_deg'])
        basis=np.array([[-np.sin(theta),np.cos(theta)],[np.cos(theta),np.sin(theta)]])
        half=np.asarray(c['wide']['background_half_uq_m'])*.98
        for attempt in range(10000):
            candidate=copy.deepcopy(c)
            candidate['seed']=int(np.random.SeedSequence([c['seed'],{'train':1,'validation':2,'test':3}[self.split],index,attempt]).generate_state(1)[0])
            sample=SyntheticEchoDataset(candidate,'train')[0]
            p,v=sample['truth']['position'].numpy(),sample['truth']['velocity'].numpy()
            edges=p[:,None]+np.array([-r['aperture_s']/2-2*r['baseline_m']/r['speed_m_s'],r['aperture_s']/2])[None,:,None]*v[:,None]
            if np.all(np.abs(edges@basis)<=half):
                sample['truth']['sample_id']=torch.tensor(index)
                return sample
        raise ValueError('Unable to generate an in-ROI trajectory')

class EchoDataset(BoundedEchoDataset):
    def __init__(self,cfg,split):
        super().__init__(cfg,split)
        self.hits=0;self.misses=0
        self.enabled=cfg['optimization']['cache_echoes']
        generation=dict(seed=cfg['seed'],radar=cfg['radar'],scene=cfg['scene'],
            background_half_uq_m=cfg['wide']['background_half_uq_m'])
        sources={name:hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()
            for name in ('gmti/data.py',)}
        payload=dict(generation=generation,sources=sources,format_version=1)
        fingerprint=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        directory=Path(cfg['optimization']['cache_dir']).expanduser()
        if not directory.is_absolute():directory=PROJECT/directory
        self.directory=directory/fingerprint/split
        if self.enabled:self.directory.mkdir(parents=True,exist_ok=True)

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        if not self.enabled:return super().__getitem__(index)
        path=self.directory/f'{index:06d}.npz'
        if not path.exists():
            # Shared across ranks and independent experiments; never expose a
            # partially written archive. Lock only this sample, not the dataset.
            with (self.directory/f'{index:06d}.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if not path.exists():
                    sample=super().__getitem__(index)
                    values={'echo':sample['echo'].numpy()}
                    values.update({'truth__'+k:v.numpy() for k,v in sample['truth'].items()})
                    temporary=None
                    try:
                        with tempfile.NamedTemporaryFile(dir=self.directory,suffix='.tmp',delete=False) as stream:
                            temporary=stream.name
                            np.savez(stream,**values)
                        os.replace(temporary,path)
                    finally:
                        if temporary is not None and os.path.exists(temporary):os.unlink(temporary)
                    self.misses+=1
                    return sample
        with np.load(path,allow_pickle=False) as archive:
            sample={'echo':torch.from_numpy(archive['echo']),
                'truth':{k[7:]:torch.from_numpy(archive[k]) for k in archive.files if k.startswith('truth__')}}
        self.hits+=1
        return sample

def trajectory_velocity(times, positions):
    positions = np.asarray(positions, dtype=float)
    if not np.isfinite(positions).all():
        # Preserve failed trajectories in the denominator instead of crashing/omitting them.
        return np.full_like(positions, np.nan)
    dense = np.linspace(times[0],times[-1],(len(times)-1)*20+1)
    points = CubicSpline(times,positions,axis=0)(dense)
    return np.gradient(points,dense,axis=0,edge_order=2)[::20]



class SimulatedTestDataset:
    """Ideal channels; shared trajectories and base noise across SCR settings."""
    def __init__(self,cfg,trajectories,scr_db,snr_db,seed):
        c=copy.deepcopy(cfg);c['seed']=seed;c['training']['test_samples']=trajectories
        self.source=EchoDataset(c,'test');self.seed=seed;self.snr_db=snr_db
        self.entries=[dict(mode='ideal',scr_db=float(scr),trajectory=i,target_present=True)
            for scr in scr_db for i in range(trajectories)]
        self._index=None;self._sample=None
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        entry=self.entries[index];i=entry['trajectory']
        if self._index!=i:self._sample=self.source[i];self._index=i
        truth=self._sample['truth']
        clean=truth['clean'].clone();background=truth['background'].clone()
        power=clean.abs().square().mean().item()
        background*=np.sqrt(power/(10**(entry['scr_db']/10)*background.abs().square().mean().item()))
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,200000,i]))
        noise=torch.from_numpy((np.sqrt(power/10**(self.snr_db/10)/2)*(rng.normal(size=clean.shape)+1j*rng.normal(size=clean.shape))).astype(np.complex64))
        echo=clean+background+noise
        info=dict(entry,sample_id=index,snr_db=self.snr_db,channel_model='ideal',
            echo_sha256=hashlib.sha256(echo.numpy().tobytes()).hexdigest(),
            measured_echo_scr_db=10*np.log10(power/background.abs().square().mean().item()))
        t=dict(truth);t.update(clean=clean,background=background,noise=noise,sample_id=torch.tensor(index),snr_db=torch.tensor(self.snr_db))
        return dict(echo=echo,truth=t,info=info)


def to_device(batch,device):
    if isinstance(batch,dict):return {k:to_device(v,device) for k,v in batch.items()}
    return batch.to(device)
