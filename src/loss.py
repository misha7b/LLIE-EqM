# src/loss.py

import torch
import torch.nn as nn
import torch.fft


def c_gamma_linear(gamma, lambda_mult=1.0, **kwargs):
    return lambda_mult * (1.0 - gamma)


def c_gamma_quadratic(gamma, lambda_mult=1.0, **kwargs):
    return lambda_mult * (1.0 - gamma) ** 2


def c_gamma_truncated(gamma, a=0.8, lambda_mult=4.0, **kwargs):
    a_t = torch.tensor(a, device=gamma.device)
    c = torch.where(gamma <= a_t, torch.ones_like(gamma), (1.0 - gamma) / (1.0 - a))
    return lambda_mult * c


def c_gamma_piecewise(gamma, a=0.5, b=2.0, lambda_mult=1.0, **kwargs):
    a_t = torch.tensor(a, device=gamma.device)
    b_t = torch.tensor(b, device=gamma.device)
    c = torch.where(gamma <= a_t, b_t - (b_t - 1.0) / a * gamma, (1.0 - gamma) / (1.0 - a))
    return lambda_mult * c


DECAY_FUNCTIONS = {
    "linear": c_gamma_linear,
    "quadratic": c_gamma_quadratic,
    "truncated": c_gamma_truncated,
    "piecewise": c_gamma_piecewise,
}

def get_c_gamma(gamma, variant="linear", **kwargs):
    if variant not in DECAY_FUNCTIONS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(DECAY_FUNCTIONS)}")
    return DECAY_FUNCTIONS[variant](gamma, **kwargs)


# ── Loss functions ────────────────────────────────────────────────────

class CharbonnierLoss(nn.Module):
    """Charbonnier loss: sqrt((pred - target)^2 + eps^2)"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps_sq))


class FFTLoss(nn.Module):
    """L1 loss on 2D FFT amplitude spectrum."""

    def forward(self, pred, target):
        pred_amp = torch.fft.rfft2(pred, norm="ortho").abs()
        target_amp = torch.fft.rfft2(target, norm="ortho").abs()
        return torch.mean(torch.abs(pred_amp - target_amp))


class CosineLoss(nn.Module):
    """1 - cosine similarity across the channel dim, averaged per pixel."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        dot = (pred * target).sum(dim=1)                # (B, H, W)
        norm_p = pred.norm(dim=1).clamp(min=self.eps)   # (B, H, W)
        norm_t = target.norm(dim=1).clamp(min=self.eps)
        cos_sim = dot / (norm_p * norm_t)
        return torch.mean(1.0 - cos_sim)


# ── EqM Loss ──────────────────────────────────────────────────────────

class EqMLoss(nn.Module):
    """Equilibrium Matching training loss.

    Args:
        lambda_fft:  Weight for FFT amplitude loss (0 to disable).
        lambda_cos:  Weight for cosine direction loss (0 to disable).
    """

    def __init__(self, variant="truncated", lambda_mult=4.0, a=0.8, b=2.0,
                 loss_type="charbonnier", gamma_sampling="uniform",
                 lambda_fft=0.0, lambda_cos=0.0):
        super().__init__()
        self.variant = variant
        self.lambda_mult = lambda_mult
        self.a = a
        self.b = b
        self.gamma_sampling = gamma_sampling
        self.lambda_fft = lambda_fft
        self.lambda_cos = lambda_cos

        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "charbonnier":
            self.criterion = CharbonnierLoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        if lambda_fft > 0:
            self.fft_loss = FFTLoss()
        if lambda_cos > 0:
            self.cos_loss = CosineLoss()

    def _sample_gamma(self, B, device):
        if self.gamma_sampling == "uniform":
            return torch.rand(B, 1, 1, 1, device=device)
        elif self.gamma_sampling == "beta":
            # More samples near 0 (dark) and 1 (light)
            dist = torch.distributions.Beta(0.5, 0.5)
            return dist.sample((B, 1, 1, 1)).to(device)
        else:
            raise ValueError(f"Unknown gamma_sampling: {self.gamma_sampling}")

    def forward(self, model, x_light, x_dark):
        B = x_light.shape[0]
        device = x_light.device

        # Sample interpolation point gamme in [0, 1]
        gamma = self._sample_gamma(B, device)
        x_gamma = gamma * x_light + (1.0 - gamma) * x_dark

        c = get_c_gamma(gamma, variant=self.variant, a=self.a, b=self.b, lambda_mult=self.lambda_mult)
        target = c * (x_dark - x_light)

        pred = model(torch.cat([x_gamma, x_dark], dim=1))
        
        #pred = model(torch.cat([x_gamma, x_dark], dim=1), gamma.view(B))

        loss = self.criterion(pred, target)

        if self.lambda_fft > 0:
            loss = loss + self.lambda_fft * self.fft_loss(pred, target)
        if self.lambda_cos > 0:
            loss = loss + self.lambda_cos * self.cos_loss(pred, target)

        return loss
    

        # DDPM
        #pred = model(torch.cat([x_gamma, x_dark], dim=1), gamma.view(B))
        

