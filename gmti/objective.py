"""联合前后端监督：各损失保留到X1/P/dX1前端的梯度。"""
import torch.nn.functional as F


def loss_function(output,truth,echo,cfg):
    s=cfg['loss']
    def relative(a,b):
        return ((a-b).abs().square().flatten(2).sum(-1)/b.abs().square().flatten(2).sum(-1).clamp_min(1e-10)).mean()
    values={
        'position':F.smooth_l1_loss(output['position']/s['position_scale_m'],truth['position']/s['position_scale_m']),
        'velocity':F.smooth_l1_loss(output['velocity']/s['velocity_scale_m_s'],truth['velocity']/s['velocity_scale_m_s']),
        'consistency':relative(output['reconstruction'],echo),
        'front_consistency':relative(output['front_reconstruction'],echo),
        'target_echo':relative(output['target_echo'],truth['clean']),
        'radial':F.smooth_l1_loss(output['ati_radial'],truth['radial'])}
    return sum(s[k]*v for k,v in values.items()),values
