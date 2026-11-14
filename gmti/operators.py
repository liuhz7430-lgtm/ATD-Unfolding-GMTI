"""Frame-streamed fixed-geometry operators with exact value gradients."""
import math
import torch
from .physics import LIGHT_SPEED

def adjoint_product(matrix,value):
    # A^H x = conj(A^T conj(x)). Avoid resolving a lazy conjugate of huge A.
    return (matrix.transpose(-1,-2)@value.conj()).conj()

class FixedMatrixProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx,matrix,value,adjoint):
        ctx.save_for_backward(matrix);ctx.adjoint=adjoint
        return adjoint_product(matrix,value) if adjoint else matrix@value
    @staticmethod
    def backward(ctx,gradient):
        matrix,=ctx.saved_tensors
        value_gradient=matrix@gradient if ctx.adjoint else adjoint_product(matrix,gradient)
        return None,value_gradient,None

class AdjointView:
    def __init__(self,matrix):self.matrix=matrix
    def __matmul__(self,value):return FixedMatrixProduct.apply(self.matrix,value,True)

class FrameMatrix:
    def __init__(self,matrix):self.matrix=matrix
    def __matmul__(self,value):return FixedMatrixProduct.apply(self.matrix,value,False)
    @property
    def mH(self):return AdjointView(self.matrix)

class DictionaryView:
    def __init__(self,matrix):self.matrix=matrix
    def __getattr__(self,key):return getattr(self.matrix,key)
    def __getitem__(self,index):return FrameMatrix(self.matrix[index])

class FrameRadar:
    def __init__(self,radar,index):
        self.radar,self.index=radar,index
        self.frames=1
        for key in ('platform','vp','ap','pulse_platform'):
            setattr(self,key,getattr(radar,key)[index:index+1])
    def __getattr__(self,key):return getattr(self.radar,key)
    def static_matrix(self,points):
        r=self.radar
        matrix=torch.empty(1,r.frequency.numel()*r.tau.numel(),len(points),device=points.device,dtype=torch.complex64)
        for start in range(0,len(points),r.chunk):
            block=points[start:start+r.chunk]
            delta=self.pulse_platform[:,:,None,:2]-block[None,None]
            distance=(delta.square().sum(-1)+r.height**2).sqrt()
            phase=-4*math.pi/LIGHT_SPEED*r.frequency[None,:,None,None]*(distance[:,None]-r.reference)
            matrix[:,:,start:start+len(block)]=torch.polar(torch.ones_like(phase),phase).to(torch.complex64).flatten(1,2)/r.norm
        return matrix

class FrameDictionary:
    def __init__(self,radar,points):self.radar=radar;self.points=points
    def __getitem__(self,index):return FrameMatrix(FrameRadar(self.radar,index).static_matrix(self.points)[0])
