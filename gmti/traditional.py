"""Untrained DPCA/ATI relocation and variable-projection Gauss–Newton.

Only radar configuration and observed echoes enter the estimator. Adjacent
channel differences cancel the registered static clutter. No network is used.
"""
import math
import torch
from .physics import RadarPhysics, los
from .operators import FrameDictionary


ALGORITHM_VERSION = 'dpca-ati-gn-v2'


def local_velocity(times, positions, window=5):
    """Local quadratic derivative, without interpolating coarse-grid jitter."""
    width = min(window, len(times))
    if width == 1:
        return torch.zeros_like(positions)
    values = []
    for i in range(len(times)):
        start = max(0, min(i-width//2, len(times)-width))
        dt = times[start:start+width]-times[i]
        design = torch.stack([dt**k for k in range(min(2, width-1)+1)], -1)
        values.append((torch.linalg.pinv(design) @ positions[:, start:start+width])[:, 1])
    return torch.stack(values, 1)


def clutter_free(values):
    """Orthonormal contrasts along channel axis 2.

    Adjacent DPCA differences have covariance sigma² [[2,-1],[-1,2]].
    These contrasts span the same clutter-free subspace with white noise.
    """
    c0, c1, c2 = values.unbind(2)
    return torch.stack(((c1-c0)/math.sqrt(2), (2*c2-c0-c1)/math.sqrt(6)), 2)


class TraditionalEstimator:
    def __init__(self, cfg, device, iterations=20):
        self.cfg = cfg
        self.radar = RadarPhysics(cfg).to(device)
        self.iterations = iterations
        self._dictionary = None

    def eval(self):
        return self

    @torch.no_grad()
    def __call__(self, echo):
        radar = self.radar
        if self._dictionary is None:
            points = radar.grid(self.cfg['model']['coarse_grid'], self.cfg['model']['coarse_half_uq_m'])
            self._dictionary = FrameDictionary(radar, points), points
        dictionary, points = self._dictionary
        differences = echo[:, :, 1:] - echo[:, :, :-1]
        images = torch.stack([
            (dictionary[t].mH @ differences[:, t].flatten(-2).transpose(-1, -2)).transpose(-1, -2)
            for t in range(radar.frames)], dim=1)
        index = images.abs().square().sum(2).argmax(-1)
        peaks = images.gather(-1, index[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
        radial = torch.angle(peaks[:, :, 1] * peaks[:, :, 0].conj()).double() / radar.ati_scale
        apparent = points[index]
        rr, n, _, _ = los(apparent, radar.platform, radar.vp, radar.ap)
        rate = -(n * radar.vp).sum(-1)
        position = apparent + radar.vp[None, :, :2] / radar.speed * (radial * rr / radar.speed)[..., None]
        for _ in range(6):
            distance, n, projection, _ = los(position, radar.platform, radar.vp, radar.ap)
            error = torch.stack((distance - rr, (-(n * radar.vp).sum(-1) + radial - rate) * rr / radar.speed), -1)
            row = -torch.einsum('...i,...ij->...j', radar.vp, projection)[..., :2] / distance[..., None] * rr[..., None] / radar.speed
            jac = torch.stack((n[..., :2], row), -2)
            position -= torch.linalg.solve(jac, error[..., None]).squeeze(-1).clamp(-50, 50)
        initial = position.clone()
        velocity = local_velocity(radar.centers, position)
        # ATI directly observes n_xy dot v. Differentiation supplies only the
        # unobserved tangential component; do not overwrite the radial estimate.
        _, n, _, _ = los(position, radar.platform, radar.vp, radar.ap)
        n = n[..., :2]
        velocity += n*((radial-(n*velocity).sum(-1))/n.square().sum(-1).clamp_min(1e-12))[..., None]
        single_velocity = velocity.clone()
        y = clutter_free(echo.flatten(3).to(torch.complex128)).flatten(2)
        position, velocity, costs, rejected = self.refine(position, velocity, y)
        _, n, _, _ = los(position, radar.platform, radar.vp, radar.ap)
        return dict(position=position, velocity=velocity, radial=(n[..., :2]*velocity).sum(-1),
                    ati_radial=radial, initial_position=initial,
                    single_position=initial, single_velocity=single_velocity,
                    objective_history=costs, rejected_updates=rejected)

    def _atom(self, position, velocity):
        atom, jac = self.radar.atom_jacobian(position, velocity)
        atom = clutter_free(atom.reshape(*position.shape[:2], 3, -1)).flatten(2)
        jac = clutter_free(jac.reshape(*position.shape[:2], 3, -1, 4)).flatten(2, 3)
        return atom, jac

    @staticmethod
    def _objective(atom, y):
        norm = atom.abs().square().sum(-1, keepdim=True).clamp_min(1e-12)
        amplitude = (atom.conj()*y).sum(-1, keepdim=True)/norm
        residual = y-amplitude*atom
        return residual.abs().square().sum(-1), amplitude, residual, norm

    @torch.no_grad()
    def refine(self, position, velocity, y):
        radar = self.radar
        limit = torch.tensor(self.cfg['model']['max_update'], device=y.device, dtype=torch.float64)
        costs = []
        rejected = torch.zeros(position.shape[:2], device=y.device, dtype=torch.int64)
        for _ in range(self.iterations):
            atom, jac = self._atom(position, velocity)
            cost, amplitude, residual, norm = self._objective(atom, y)
            costs.append(cost)
            jac = amplitude[..., None] * (jac - atom[..., None] * (atom.conj()[..., None] * jac).sum(-2, keepdim=True) / norm[..., None])
            h = (jac.mH @ jac).real
            rhs = (jac.conj() * residual[..., None]).sum(-2).real
            scale = h.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12).sqrt()
            h = h / (scale[..., None] * scale[..., None, :])
            delta = torch.linalg.solve(h + self.cfg['model']['initial_damping'] * torch.eye(4, device=y.device), (rhs / scale)[..., None]).squeeze(-1) / scale
            # A shared scale preserves the descent direction in this strongly
            # coupled system. Componentwise tanh can turn it into an ascent step.
            factor = (delta.abs()/limit).amax(-1, keepdim=True).clamp_min(1)
            update = delta/factor*self.cfg['model']['initial_step']
            for _ in range(10):
                candidate_p = position+update[..., :2]@radar.basis.T
                candidate_v = velocity+update[..., 2:]@radar.basis.T
                candidate_atom, _ = self._atom(candidate_p, candidate_v)
                candidate_cost = self._objective(candidate_atom, y)[0]
                accept = torch.isfinite(candidate_cost) & (candidate_cost <= cost)
                if accept.all():
                    break
                update = torch.where(accept[..., None], update, update*.5)
            position = torch.where(accept[..., None], candidate_p, position)
            velocity = torch.where(accept[..., None], candidate_v, velocity)
            rejected += ~accept
        final_atom, _ = self._atom(position, velocity)
        costs.append(self._objective(final_atom, y)[0])
        return position, velocity, torch.stack(costs, -1), rejected
