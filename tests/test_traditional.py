"""Independent invariants for ATI initialization and safeguarded physical fitting."""
import torch
from gmti.config import PROJECT, load
from gmti.operators import FrameDictionary
from gmti.physics import los
from gmti.traditional import TraditionalEstimator, clutter_free, local_velocity


def config():
    cfg = load(PROJECT/'configs/current.yaml')
    cfg['radar'].update(frames=5, frequency_samples=16, pulses=32)
    cfg['model'].update(coarse_grid=[17, 13], coarse_half_uq_m=[40., 3.], operator_chunk=32)
    return cfg


def test_local_derivative_recovers_acceleration_and_handles_short_tracks():
    times = torch.arange(7, dtype=torch.float64)*.45
    p0 = torch.tensor([[2., -3.]], dtype=torch.float64)
    v0 = torch.tensor([[.7, -.4]], dtype=torch.float64)
    accel = torch.tensor([[.1, .2]], dtype=torch.float64)
    position = p0[:, None]+times[None, :, None]*v0[:, None]+.5*times[None, :, None]**2*accel[:, None]
    torch.testing.assert_close(local_velocity(times, position), v0[:, None]+times[None, :, None]*accel[:, None], atol=1e-13, rtol=0)
    torch.testing.assert_close(local_velocity(times[:1], position[:, :1]), torch.zeros_like(position[:, :1]))
    linear = p0[:, None]+times[None, :2, None]*v0[:, None]
    torch.testing.assert_close(local_velocity(times[:2], linear), v0[:, None].expand(-1, 2, -1))


def test_contrasts_cancel_static_clutter_and_preserve_white_noise():
    h = clutter_free(torch.eye(3, dtype=torch.float64)[None, None])[0, 0]
    torch.testing.assert_close(h@torch.ones(3, dtype=torch.float64), torch.zeros(2, dtype=torch.float64), atol=1e-15, rtol=0)
    torch.testing.assert_close(h@h.T, torch.eye(2, dtype=torch.float64), atol=1e-15, rtol=0)
    generator = torch.Generator().manual_seed(102)
    target = torch.randn(2, 5, 3, 20, dtype=torch.complex128, generator=generator)
    static = torch.randn(2, 5, 1, 20, dtype=torch.complex128, generator=generator)
    torch.testing.assert_close(clutter_free(target+static), clutter_free(target), atol=1e-15, rtol=0)


def test_initial_velocity_retains_measured_ati_radial_component():
    torch.set_num_threads(1)
    model = TraditionalEstimator(config(), 'cpu', iterations=0)
    radar = model.radar
    velocity = torch.tensor([.4, .7], dtype=torch.float64).expand(1, 5, 2)
    position = radar.centers[None, :, None]*velocity
    atom, _ = radar.atom_jacobian(position, velocity)
    echo = atom.reshape(1, 5, 3, 16, 32).to(torch.complex64)
    out = model(echo)
    _, n, _, _ = los(out['position'], radar.platform, radar.vp, radar.ap)
    torch.testing.assert_close((n[..., :2]*out['single_velocity']).sum(-1), out['ati_radial'], atol=1e-14, rtol=0)
    assert isinstance(model._dictionary[0], FrameDictionary)


def test_noiseless_refinement_decreases_observed_objective_every_iteration():
    torch.set_num_threads(1)
    model = TraditionalEstimator(config(), 'cpu', iterations=15)
    radar = model.radar
    velocity = torch.tensor([.4, .7], dtype=torch.float64).expand(1, 5, 2)
    position = radar.centers[None, :, None]*velocity
    atom, _ = radar.atom_jacobian(position, velocity)
    y = clutter_free(atom.reshape(1, 5, 3, -1)).flatten(2)
    initial_p = position+torch.tensor([.4, .04], dtype=torch.float64)@radar.basis.T
    initial_v = velocity+torch.tensor([.15, .005], dtype=torch.float64)@radar.basis.T
    p, v, costs, _ = model.refine(initial_p, initial_v, y)
    assert torch.isfinite(p).all() and torch.isfinite(v).all()
    assert torch.all(costs[..., 1:] <= costs[..., :-1])
    assert torch.all(costs[..., -1] < .01*costs[..., 0])
