import torch

def clip_gradient(model, clip_norm):
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.norm()
            totalnorm += modulenorm ** 2

    totalnorm = torch.sqrt(totalnorm).item()
    norm = (clip_norm / max(totalnorm, clip_norm))

    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)