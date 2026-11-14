"""Current implementation: geometry, gradients, paired ideal data, and reports."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import torch
from gmti.config import PROJECT,load,diagnostics,support
from gmti.data import EchoDataset,BoundedEchoDataset,SimulatedTestDataset
from gmti.model import JointUnfoldNet
from gmti.objective import loss_function
from gmti.physics import RadarPhysics
from gmti.operators import FrameRadar
from gmti.temporal import motion_system
from gmti.report import make_report,render_focus
from gmti.focus import save_focus


def small_config(cache):
    cfg=load(PROJECT/'configs/current.yaml')
    cfg['radar'].update(frames=5,frequency_samples=12,pulses=24,bandwidth_hz=60e6)
    cfg['model'].update(coarse_grid=[9,7],coarse_half_uq_m=[20.,3.],operator_chunk=4,stages=2,front_stages=2)
    cfg['temporal'].update(window=3,stages=2)
    cfg['scene'].update(position_half_uq_m=[1.,.3],velocity_half_xy_m_s=[.8,.5],acceleration_half_xy_m_s2=[.02,.02],min_abs_radial_velocity_m_s=.1,clutter_points=3,clutter_half_uq_m=[5.,2.])
    cfg['wide'].update(background_half_uq_m=[5.,2.],window_batch=3)
    cfg['optimization'].update(background_grid_half_uq_m=[5.,2.],cache_dir=str(cache))
    cfg['training'].update(epochs=2,train_samples=3,validation_samples=3,test_samples=3)
    return cfg


def test_support_sampling_and_frame_operator():
    cfg=load(PROJECT/'configs/current.yaml');info=diagnostics(cfg);points,mask=support(cfg)
    assert len(points)==8911 and mask.sum()==7399
    assert info['background_size_uq_m']==[60.,60.]
    assert info['exact_union_range_span_m']<.9*info['slant_range_period_m']
    assert info['one_frame_dictionary_gib']<3.3
    assert np.any((points[:,1]==30)&(~mask))


def test_operator_and_cached_scales_preserve_differentiation(tmp_path):
    torch.set_num_threads(1);cfg=small_config(tmp_path)
    model=JointUnfoldNet(cfg);r=model.radar
    points=r.grid([5,3],[3.,1.]);full=r.static_matrix(points)
    for k in range(r.frames):torch.testing.assert_close(FrameRadar(r,k).static_matrix(points)[0],full[k],atol=0,rtol=0)
    sample=EchoDataset(cfg,'train')[0];echo=sample['echo'][None]
    truth={k:v[None] for k,v in sample['truth'].items()}
    first=model(echo);loss,_=loss_function(first,truth,echo,cfg);loss.backward()
    grads={n:p.grad.clone() for n,p in model.named_parameters()}
    assert all(torch.isfinite(g).all() for g in grads.values())
    assert model.frontend.scale_builds==r.frames
    model.zero_grad(set_to_none=True);second=model(echo)
    for key in first:torch.testing.assert_close(first[key],second[key],atol=0,rtol=0)
    loss,_=loss_function(second,truth,echo,cfg);loss.backward()
    for n,p in model.named_parameters():torch.testing.assert_close(p.grad,grads[n],atol=0,rtol=0)
    assert model.frontend.scale_builds==r.frames


def test_echo_cache_is_exact_and_invalidates_generation_changes(tmp_path):
    cfg=small_config(tmp_path)
    expected=BoundedEchoDataset(cfg,'train')[0]
    first=EchoDataset(cfg,'train')[0]
    with patch.object(BoundedEchoDataset,'__getitem__',side_effect=AssertionError('regenerated')):
        cached=EchoDataset(cfg,'train')[0]
    for row in (first,cached):
        torch.testing.assert_close(row['echo'],expected['echo'],atol=0,rtol=0)
        for key in row['truth']:torch.testing.assert_close(row['truth'][key],expected['truth'][key],atol=0,rtol=0)
    changed=copy.deepcopy(cfg);changed['radar']['bandwidth_hz']*=1.1
    assert EchoDataset(cfg,'train').directory!=EchoDataset(changed,'train').directory
    assert EchoDataset(cfg,'train').directory!=EchoDataset(cfg,'validation').directory


def test_ideal_pairs_share_motion_and_noise_across_scr(tmp_path):
    cfg=small_config(tmp_path);data=SimulatedTestDataset(cfg,2,[-5.,5.],15.,88)
    assert len(data)==4 and all(e['mode']=='ideal' for e in data.entries)
    a,b=data[0],data[2]
    for key in ('position','velocity','noise'):torch.testing.assert_close(a['truth'][key],b['truth'][key],atol=0,rtol=0)
    for i in range(len(data)):
        s=data[i]
        assert abs(s['info']['measured_echo_scr_db']-s['info']['scr_db'])<2e-5
        torch.testing.assert_close(s['echo'],s['truth']['clean']+s['truth']['background']+s['truth']['noise'],atol=0,rtol=0)


def test_motion_constraint_does_not_penalize_shared_translation():
    times=torch.arange(5,dtype=torch.float64)*.45
    velocity=torch.tensor([.4,-.2],dtype=torch.float64).expand(5,-1)
    position=times[:,None]*velocity
    a,j=motion_system(position,velocity,times,torch.eye(2,dtype=torch.float64),.5)
    b,_=motion_system(position+torch.tensor([10.,-3.]),velocity,times,torch.eye(2,dtype=torch.float64),.5)
    torch.testing.assert_close(a,b,atol=1e-13,rtol=0)


def test_reports_use_mean_vector_error_and_ideal_only_figures(tmp_path):
    cfg=small_config(tmp_path/'cache')
    row=dict(id=0,info=dict(mode='ideal',scr_db=0.,trajectory=0),position=[[3.,4.],[0.,0.]],velocity=[[0.,2.],[0.,0.]],true_position=[[0.,0.],[0.,0.]],true_velocity=[[0.,0.],[0.,0.]])
    make_report(tmp_path,{'trained':[copy.deepcopy(row)],'traditional':[copy.deepcopy(row)]},dict(dataset='simulated',samples=1),cfg)
    table=json.loads((tmp_path/'mean_errors.json').read_text())
    assert all(r['mean_error']==(2.5 if r['quantity']=='position' else 1.) for r in table)
    assert {p.name for p in (tmp_path/'figures').glob('*.png')}=={'mean_position_error.png','mean_velocity_error.png','position_estimates.png','velocity_estimates.png'}


def test_focus_uses_truth_only_for_markers(tmp_path):
    cfg=small_config(tmp_path/'cache');model=JointUnfoldNet(cfg).eval();sample=EchoDataset(cfg,'test')[0]
    sample['info']=dict(mode='ideal',scr_db=0.,trajectory=0)
    with torch.no_grad():pred=model(sample['echo'][None])
    save_focus(model,sample,pred,0,tmp_path/'a',frame_count=1)
    changed=copy.deepcopy(sample);changed['truth']['position']+=1000
    save_focus(model,changed,pred,0,tmp_path/'b',frame_count=1)
    with np.load(tmp_path/'a/data/focus/sample_0000/frame_000.npz') as a,np.load(tmp_path/'b/data/focus/sample_0000/frame_000.npz') as b:
        for key in ('before','after','u_before','q_before','u_after','q_after'):np.testing.assert_array_equal(a[key],b[key])
        assert not np.array_equal(a['truth_uq'],b['truth_uq'])
    (tmp_path/'a/figures').mkdir();render_focus(tmp_path/'a')
    assert (tmp_path/'a/figures/refocus_selected_frames.png').is_file()


def test_missing_checkpoint_fails_before_gpu_launch(tmp_path):
    result=subprocess.run([sys.executable,str(PROJECT/'test.py'),'--gpus','0,1,2','--checkpoint',str(tmp_path/'missing.pt')],capture_output=True,text=True)
    assert result.returncode==2 and 'Checkpoint not found' in result.stderr
    assert 'torch.distributed' not in result.stderr


def test_example_selection_includes_every_trajectory_once():
    from gmti.focus import example_ids
    entries=[dict(mode='ideal',scr_db=scr,trajectory=i,target_present=True) for scr in (-5.,0.,10.) for i in range(4)]
    assert example_ids(entries)==[4,5,6,7]
    assert example_ids(entries,2)==[4,5]
    assert {entries[i]['trajectory'] for i in example_ids(entries)}=={0,1,2,3}


def test_state_figures_include_all_four_trajectories(tmp_path):
    from gmti.report import render_states
    cfg=small_config(tmp_path/'cache');rows=[]
    for scr in (-5.,0.,10.):
        for trajectory in range(4):
            rows.append(dict(id=len(rows),info=dict(mode='ideal',scr_db=scr,trajectory=trajectory,sample_id=len(rows)),
                position=[[.1,.1],[.2,.2]],velocity=[[.2,.1],[.3,.2]],true_position=[[0.,0.],[.1,.1]],true_velocity=[[.1,0.],[.2,.1]]))
    (tmp_path/'figures').mkdir()
    render_states(tmp_path,{'trained':rows},cfg)
    selected=json.loads((tmp_path/'state_plot_selection.json').read_text())
    assert [r['trajectory'] for r in selected]==[0,1,2,3]
    assert [r['sample_id'] for r in selected]==[4,5,6,7]
    for i in range(4):
        folder=tmp_path/'figures'/f'trajectory_{i:03d}'
        assert (folder/'position_estimates.png').is_file()
        assert (folder/'velocity_estimates.png').is_file()
