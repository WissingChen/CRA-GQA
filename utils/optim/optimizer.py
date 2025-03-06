import torch
import math


def build_optimizer(cfgs, model):
    optim = cfgs["optim"]["optimizer"]
    weight_decay = cfgs["optim"]["weight_decay"]
    lr = cfgs["optim"]["lr"]
    amsgrad = cfgs["optim"]["amsgrad"]
    params_for_optimization = list(p for p in model.parameters() if p.requires_grad)
    if optim == 'SGD':
        optimizer = getattr(torch.optim, optim)(
            [{'params': params_for_optimization, 'lr': lr}],
            weight_decay=weight_decay,
        )
    else:
        optimizer = getattr(torch.optim, optim)(
            [{'params': params_for_optimization, 'lr': lr}],
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    return optimizer

