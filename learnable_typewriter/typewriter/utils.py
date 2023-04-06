from collections import OrderedDict

import torch
from torch import nn

from learnable_typewriter.typewriter.typewriter.utils import Identity

# model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

# Clamps
class Clamp(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return torch.clamp(x, 0, 1)

class SoftClamp(nn.Module):
    def __init__(self, alpha=0.01, inplace=False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, x):
        x0 = torch.min(x, torch.zeros(x.shape, device=x.device))
        x1 = torch.max(x - 1, torch.zeros(x.shape, device=x.device))
        if self.inplace:
            return x.clamp_(0, 1).add_(x0, alpha=self.alpha).add_(x1, alpha=self.alpha)
        else:
            return torch.clamp(x, 0, 1) + self.alpha * x0 + self.alpha * x1

class DifferentiableClampFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        return inp.clamp(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()

class DiffClamp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return DifferentiableClampFunc.apply(x)


def get_clamp_func(name):
    if name in [True, 'clamp', 'normal']:
        func = Clamp()
    elif not name:
        func = Identity()
    elif name.startswith('soft') or name.startswith('leaky'):
        alpha = name.replace('soft', '').replace('leaky_relu', '')
        kwargs = {'alpha': float(alpha)} if len(alpha) > 0 else {}
        func = SoftClamp(**kwargs)
    elif name.startswith('diff'):
        func = DiffClamp()
    else:
        raise NotImplementedError(f'{name} is not a valid clamp function')
    return func
