import torch
from spenpy.spen import spen

class physical_model:
    def __init__(self, img_size=(96, 96), device='cuda'):
        self.InvA, self.AFinal = spen(acq_point=img_size).get_InvA()
        self.InvA = torch.as_tensor(self.InvA).detach().to(device=device, dtype=torch.complex64)
        self.AFinal = torch.as_tensor(self.AFinal).detach().to(device=device, dtype=torch.complex64)
    
    def __call__(self, x, phase_map=None):
        x = x.detach().to(torch.complex64)
        x = torch.matmul(self.AFinal * 1j, x)
        if phase_map is not None:
            x[:, 1::2, :] *= torch.exp(1j * phase_map)
        return x
    
    def recons(self, x, phase_map=None):
        if phase_map is not None:
            x[:, 1::2, :] *= torch.exp(-1j * phase_map)
        return torch.matmul(self.InvA, x)    