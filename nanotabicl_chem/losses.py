import torch

def pinball_loss(pred_quantiles,target,levels):
    error=target[...,None]-pred_quantiles
    return torch.maximum(levels*error,(levels-1.0)*error).mean()
