"""Canonical current configuration, union support, and sampling checks."""
from pathlib import Path
import math
import numpy as np
import yaml
PROJECT=Path(__file__).resolve().parent.parent

def validate(cfg):
    def numeric(x):
        if isinstance(x, dict): return {k:numeric(v) for k,v in x.items()}
        if isinstance(x, list): return [numeric(v) for v in x]
        if isinstance(x, str): return float(x)
        return x
    cfg = numeric(cfg)
    # Frozen pre-constraint configurations retain their original scene distribution.
    cfg['scene'].setdefault('min_abs_radial_velocity_m_s', 0.0)
    schema = yaml.safe_load((PROJECT/'configs/current.yaml').read_text())
    schema.pop('wide');schema.pop('optimization')
    def check(x, ref):
        if isinstance(ref, dict):
            if not isinstance(x, dict) or set(x) != set(ref): raise ValueError('Configuration keys mismatch')
            for k in ref: check(x[k], ref[k])
        elif isinstance(x, list):
            if len(x) != len(ref): raise ValueError('Configuration list length mismatch')
            for v in x:
                if not math.isfinite(v): raise ValueError('Non-finite configuration')
        elif not math.isfinite(x): raise ValueError('Non-finite configuration')
    check(cfg, schema)
    r, m, t = cfg['radar'], cfg['model'], cfg['training']
    for key in ('channels', 'frames', 'frequency_samples', 'pulses'):
        if not isinstance(r[key], int) or r[key] < 3: raise ValueError(key)
    if r['aperture_s'] != r['frame_interval_s']: raise ValueError('This experiment uses contiguous subapertures')
    for key in ('carrier_hz','bandwidth_hz','aperture_s','radius_m','height_m','speed_m_s','baseline_m'):
        if r[key] <= 0: raise ValueError(key)
    if r['bandwidth_hz'] >= 2*r['carrier_hz']: raise ValueError('Invalid bandwidth')
    for key in ('epochs','train_samples','validation_samples','test_samples','batch_size'):
        if not isinstance(t[key], int) or t[key] < 1: raise ValueError(key)
    if m['stages'] < 1 or m['initial_damping'] <= 0 or not 0 < m['initial_step'] < 1: raise ValueError('Invalid unfolding settings')
    if any(n < 3 or n%2 != 1 for n in m['coarse_grid']): raise ValueError('Odd grids required')
    if r['channels'] != 3: raise ValueError('X1/P/dX1 front end requires exactly three channels')
    if m['front_stages'] < 1 or not 0 < m['front_step'] < 1 or not 0 < m['phase_step'] < 1: raise ValueError('Invalid front-end iterations/steps')
    if min(m['support_threshold'],m['support_temperature'],m['recovery_epsilon']) <= 0: raise ValueError('Positive front-end regularization required')
    if cfg['scene']['min_abs_radial_velocity_m_s'] < 0:
        raise ValueError('min_abs_radial_velocity_m_s must be nonnegative')
    temporal=cfg['temporal']
    if not isinstance(temporal['enabled'],bool): raise ValueError('temporal.enabled must be boolean')
    if not isinstance(temporal['window'],int) or temporal['window']<1 or temporal['window']%2!=1 or temporal['window']>r['frames']:
        raise ValueError('temporal.window must be odd and within the frame count')
    if not isinstance(temporal['stages'],int) or temporal['stages']<0: raise ValueError('Invalid temporal stages')
    if temporal['motion_weight']<0 or temporal['motion_scale_m']<=0 or temporal['initial_damping']<=0:
        raise ValueError('Invalid temporal constraint or damping')
    if not 0<temporal['initial_step']<1 or any(x<=0 for x in temporal['max_update']):
        raise ValueError('Invalid temporal step/update')
    return cfg

def background_diagnostics(cfg):
    r=cfg['radar'];hu,hq=cfg['wide']['background_half_uq_m']
    c=299792458.;distance=math.hypot(r['radius_m'],r['height_m'])
    range_period=c*(r['frequency_samples']-1)/(2*r['bandwidth_hz'])
    # Exact corner range span at each pulse; no small-angle FOV assumption.
    import numpy as np
    times=(np.arange(r['frames'])-(r['frames']-1)/2)*r['frame_interval_s']
    tau=(np.arange(r['pulses'])-(r['pulses']-1)/2)*r['aperture_s']/r['pulses']
    angle0=math.radians(r['initial_angle_deg'])
    basis=np.array([[-math.sin(angle0),math.cos(angle0)],[math.cos(angle0),math.sin(angle0)]])
    corners=np.array([[-hu,-hq],[-hu,hq],[hu,-hq],[hu,hq]])@basis.T
    theta=angle0+r['speed_m_s']/r['radius_m']*(times[:,None]+tau)
    platform=r['radius_m']*np.stack((np.cos(theta),np.sin(theta)),-1)
    ranges=np.sqrt(((platform[...,None,:]-corners)**2).sum(-1)+r['height_m']**2)
    span=float(np.ptp(ranges,axis=-1).max())
    az_period=c/r['carrier_hz']*distance*r['pulses']/(2*r['speed_m_s']*r['aperture_s'])
    if span>=.9*range_period:raise ValueError(f'Wide ROI range span {span:.2f} m is too close to sampling period {range_period:.2f} m')
    if 2*hu>=.9*az_period:raise ValueError('Wide ROI exceeds approximate azimuth sampling period')
    return dict(frequency_samples=r['frequency_samples'],pulses=r['pulses'],
        slant_range_resolution_m=c/(2*r['bandwidth_hz']),
        approximate_ground_range_resolution_m=c/(2*r['bandwidth_hz'])*distance/r['radius_m'],
        approximate_azimuth_resolution_m=c/r['carrier_hz']*distance/(2*r['speed_m_s']*r['aperture_s']),
        slant_range_period_m=range_period,exact_roi_range_span_m=span,
        approximate_azimuth_period_m=az_period,background_size_uq_m=[2*hu,2*hq],
        one_frame_dictionary_gib=r['frequency_samples']*r['pulses']*math.prod(cfg['model']['coarse_grid'])*8/2**30)

def support(cfg):
    """Shared lattice: target apparent-image support UNION physical background."""
    shape=cfg['model']['coarse_grid'];half=cfg['model']['coarse_half_uq_m']
    spacing=2*np.asarray(half)/(np.asarray(shape)-1)
    target=(np.asarray(shape)-1)//2
    background=np.asarray(cfg['optimization']['background_grid_half_uq_m'])/spacing
    if not np.allclose(background,np.rint(background),atol=1e-9,rtol=0):
        raise ValueError('Background bounds must align with the target coarse lattice')
    background=np.rint(background).astype(int)
    limit=np.maximum(target,background)
    indices=[];mask=[]
    for q in range(-limit[1],limit[1]+1):
        for u in range(-limit[0],limit[0]+1):
            in_target=abs(u)<=target[0] and abs(q)<=target[1]
            in_background=abs(u)<=background[0] and abs(q)<=background[1]
            if in_target or in_background:
                indices.append((u,q));mask.append(in_target)
    return np.asarray(indices,dtype=np.float64)*spacing,np.asarray(mask,dtype=bool)

def diagnostics(cfg):
    info=background_diagnostics(cfg)
    points,mask=support(cfg);r=cfg['radar']
    # Check ALL union points at all frame/pulse centers, not a rectangular
    # bounding box whose unused corners overestimate the range span.
    theta0=np.deg2rad(r['initial_angle_deg'])
    basis=np.array([[-np.sin(theta0),np.cos(theta0)],[np.cos(theta0),np.sin(theta0)]])
    t=(np.arange(r['frames'])-(r['frames']-1)/2)*r['frame_interval_s']
    tau=(np.arange(r['pulses'])-(r['pulses']-1)/2)*r['aperture_s']/r['pulses']
    theta=theta0+r['speed_m_s']/r['radius_m']*(t[:,None]+tau)
    platform=r['radius_m']*np.stack((np.cos(theta),np.sin(theta)),-1)
    lo=np.full(theta.shape,np.inf);hi=np.full(theta.shape,-np.inf)
    for block in np.array_split(points@basis.T,max(1,math.ceil(len(points)/128))):
        distances=np.sqrt(((platform[...,None,:]-block)**2).sum(-1)+r['height_m']**2)
        lo=np.minimum(lo,distances.min(-1));hi=np.maximum(hi,distances.max(-1))
    span=float((hi-lo).max())
    az_period=info['approximate_azimuth_period_m']*r['carrier_hz']/(r['carrier_hz']+r['bandwidth_hz']/2)
    if span>=.9*info['slant_range_period_m']:raise ValueError('Union support exceeds range sampling margin')
    if np.ptp(points[:,0])>=.9*az_period:raise ValueError('Union support exceeds azimuth sampling margin')
    info.update(dictionary_points=len(points),target_search_points=int(mask.sum()),
        exact_union_range_span_m=span,highest_frequency_azimuth_period_m=az_period,
        coarse_spacing_uq_m=(2*np.asarray(cfg['model']['coarse_half_uq_m'])/(np.asarray(cfg['model']['coarse_grid'])-1)).tolist(),
        one_frame_dictionary_gib=r['frequency_samples']*r['pulses']*len(points)*8/2**30)
    return info

def load(path):
    raw=yaml.safe_load(Path(path).read_text())
    wide=raw.pop('wide');optimization=raw.pop('optimization')
    cfg=validate(raw)
    if set(wide)!={'background_half_uq_m','scatter_spacing_uq_m','crop','scene_file','window_batch'}:
        raise ValueError('Unexpected wide-scene settings')
    if set(optimization)!={'background_grid_half_uq_m','cache_echoes','cache_dir'}:
        raise ValueError('Unexpected optimization settings')
    for values in (wide['background_half_uq_m'],wide['scatter_spacing_uq_m'],optimization['background_grid_half_uq_m']):
        if len(values)!=2 or not all(math.isfinite(x) and x>0 for x in values):raise ValueError('Invalid spatial bounds')
    if not isinstance(wide['window_batch'],int) or wide['window_batch']<1:raise ValueError('window_batch')
    if not isinstance(optimization['cache_echoes'],bool) or not isinstance(optimization['cache_dir'],str):raise ValueError('Invalid cache settings')
    if any(a<b for a,b in zip(optimization['background_grid_half_uq_m'],wide['background_half_uq_m'])):
        raise ValueError('Reconstruction grid must cover the physical background')
    if cfg['model']['operator_chunk']<1:raise ValueError('operator_chunk')
    cfg.update(wide=wide,optimization=optimization)
    diagnostics(cfg)
    return cfg
