"""Measured-image-driven ideal static background with injected known motion."""
import copy
import hashlib
from .config import load
import numpy as np
import torch
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.signal.windows import tukey
from .gff import read_gff
from .physics import RadarPhysics
from .operators import FrameRadar
from .data import sample_trajectory

def scatter_field(scene, crop, radar, spacing=(.5, .25), half=(12., 8.)):
    """Approximate image-axis mapping; no recovery of original MiniSAR geometry.

    Image range and azimuth spacings are used as local q/u spacings by convention.
    No claim is made that this is a georeferenced ground projection. The complex
    field is low-pass filtered before sampling, then tapered and energy normalized.
    """
    image, metadata = read_gff(scene)
    r0, r1, a0, a1 = crop
    if not (0 <= r0 < r1 <= image.shape[0] and 0 <= a0 < a1 <= image.shape[1]):
        raise ValueError('Invalid source crop')
    patch = image[r0:r1, a0:a1].copy()
    dr, da = metadata['range_pixel_m'], metadata['azimuth_pixel_m']
    if min(dr, da) <= 0 or half[0] > (patch.shape[1]-1)*da/2 or half[1] > (patch.shape[0]-1)*dr/2:
        raise ValueError('Physical background ROI exceeds source crop')
    # Anti-alias filter for the equivalent scattering lattice, not display pixels.
    sigma = (spacing[1]/dr/2.355, spacing[0]/da/2.355)
    smooth = gaussian_filter(patch.real, sigma) + 1j*gaussian_filter(patch.imag, sigma)
    u = np.linspace(-half[0], half[0], int(round(2*half[0]/spacing[0]))+1)
    q = np.linspace(-half[1], half[1], int(round(2*half[1]/spacing[1]))+1)
    qq, uu = np.meshgrid(q, u, indexing='ij')
    coords = np.stack(((patch.shape[0]-1)/2+qq/dr, (patch.shape[1]-1)/2+uu/da))
    coeff = map_coordinates(smooth.real, coords, order=1) + 1j*map_coordinates(smooth.imag, coords, order=1)
    coeff *= tukey(len(q), .2)[:, None]*tukey(len(u), .2)[None, :]
    energy = np.linalg.norm(coeff)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError('Empty or invalid complex background field')
    coeff = (coeff/energy).astype(np.complex64)
    local = np.stack((uu, qq), -1)
    points = torch.as_tensor(local.reshape(-1, 2), dtype=torch.float64, device=radar.basis.device) @ radar.basis.T
    return points, coeff, dict(metadata=metadata, crop=list(crop),
        coordinate_convention='Image azimuth -> local u, image range -> local ground q, using image sampling numerically; approximate texture transfer, not georeferencing',
        u_m=u.tolist(), q_m=q.tolist(), spacing_uq_m=list(spacing), half_uq_m=list(half),
        gaussian_sigma_pixels=list(sigma), scatterers=int(coeff.size)), patch

def source_pixels(local_uq, field):
    """Return original-file [azimuth, range] pixels using the synthesis mapping."""
    r0,r1,a0,a1=field['crop']
    center=np.array([(a0+a1-1)/2,(r0+r1-1)/2])
    spacing=np.array([field['metadata']['azimuth_pixel_m'],field['metadata']['range_pixel_m']])
    return np.asarray(local_uq)/spacing+center


@torch.no_grad()
def project_background(radar,points,coeff,device):
    """One shared ideal receive-channel field, projected frame by frame."""
    weights=torch.as_tensor(coeff.reshape(-1,1),dtype=torch.complex64,device=device)
    projected=torch.zeros((radar.frames,radar.frequency.numel()*radar.tau.numel(),1),dtype=torch.complex64,device=device)
    for index in range(radar.frames):
        frame=FrameRadar(radar,index)
        for start in range(0,len(points),radar.chunk):
            a=frame.static_matrix(points[start:start+radar.chunk])[0]
            projected[index]+=a@weights[start:start+radar.chunk]
            del a
    f,n=radar.frequency.numel(),radar.tau.numel()
    return projected[...,0].reshape(radar.frames,1,f,n).expand(-1,3,-1,-1).contiguous().cpu()


class SemiRealDataset:
    """Ideal channels only; truth is generated here and never passed to inference."""
    def __init__(self,cfg,background,trajectories,scr_db,snr_db,seed,half_uq):
        self.cfg,self.background=cfg,background
        self.seed,self.snr_db=seed,snr_db;self.half_uq=np.asarray(half_uq,dtype=float)
        self.entries=[dict(mode='ideal',scr_db=float(scr),trajectory=i,target_present=True)
            for scr in scr_db for i in range(trajectories)]
    def __len__(self):return len(self.entries)
    def __getitem__(self,index):
        entry=self.entries[index];cfg,r=self.cfg,self.cfg['radar']
        rng=np.random.default_rng(self.seed+100000+entry['trajectory'])
        radar=RadarPhysics(cfg)
        for attempt in range(10000):
            position,velocity,radial=sample_trajectory(rng,cfg)
            endpoints=position[:,None,:]+np.array([-r['aperture_s']/2-2*r['baseline_m']/r['speed_m_s'],r['aperture_s']/2])[None,:,None]*velocity[:,None,:]
            if np.all(np.abs(endpoints@radar.basis.numpy())<=self.half_uq*.98):break
        else:raise ValueError('Could not sample an in-ROI trajectory')
        with torch.no_grad():atom,_=radar.atom_jacobian(torch.from_numpy(position)[None],torch.from_numpy(velocity)[None])
        clean=atom.reshape(r['frames'],3,r['frequency_samples'],r['pulses']).to(torch.complex64)
        clean*=np.exp(1j*rng.uniform(-np.pi,np.pi))
        power=clean.abs().square().mean().item()
        scale=np.sqrt(power/(10**(entry['scr_db']/10)*self.background.abs().square().mean().item()))
        background=self.background*scale
        noise_rng=np.random.default_rng(self.seed+200000+entry['trajectory'])
        noise=np.sqrt(power/10**(self.snr_db/10)/2)*(noise_rng.normal(size=clean.shape)+1j*noise_rng.normal(size=clean.shape))
        noise=torch.from_numpy(noise.astype(np.complex64));echo=clean+background+noise
        info=dict(entry,sample_id=index,snr_db=self.snr_db,channel_model='ideal',trajectory_roi_rejections=attempt,
            scr_definition='Mean target/background power over all frames, channels, frequencies and pulses',
            measured_echo_scr_db=(10*torch.log10(clean.abs().square().mean()/background.abs().square().mean())).item(),
            background_scale=float(scale),measured_snr_db=(10*torch.log10(torch.tensor(power)/noise.abs().square().mean())).item(),
            echo_sha256=hashlib.sha256(echo.numpy().tobytes()).hexdigest())
        return dict(echo=echo,truth=dict(position=torch.from_numpy(position),velocity=torch.from_numpy(velocity),radial=torch.from_numpy(radial),
            clean=clean,background=background,noise=noise,sample_id=torch.tensor(index),snr_db=torch.tensor(self.snr_db)),info=info)


def configure_scene(cfg,scene_config,crop=None):
    """Select current image texture without changing checkpoint radar/model settings."""
    selection=load(scene_config)['wide']
    updated=copy.deepcopy(cfg)
    chosen=list(selection['crop'] if crop is None else crop)
    if len(chosen)!=4 or any(not isinstance(v,int) or isinstance(v,bool) for v in chosen):
        raise ValueError('Scene crop requires four integer native-pixel bounds')
    r0,r1,a0,a1=chosen
    if not (0<=r0<r1 and 0<=a0<a1):raise ValueError('Invalid source crop')
    updated['wide']['scene_file']=selection['scene_file']
    updated['wide']['crop']=chosen
    return updated
