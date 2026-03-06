# src/loss.py

import torch
import torch.nn as nn


def c_gamma_linear(gamma, lambda_mult=1.0):
    return lambda_mult * (1.0 - gamma)


def c_gamma_quadratic(gamma, lambda_mult=1.0):
    return lambda_mult * (1.0 - gamma) ** 2


def c_gamma_truncated(gamma, a=0.8, lambda_mult=4.0):
    a_t = torch.tensor(a, device=gamma.device)
    c = torch.where(gamma <= a_t, torch.ones_like(gamma), (1.0 - gamma) / (1.0 - a))
    return lambda_mult * c


def c_gamma_piecewise(gamma, a=0.5, b=2.0, lambda_mult=1.0):
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


class EqMLoss(nn.Module):

    def __init__(self, variant="truncated", lambda_mult=4.0, a=0.8, b=2.0):
        super().__init__()
        self.variant = variant
        self.lambda_mult = lambda_mult
        self.a = a
        self.b = b
        self.mse = nn.MSELoss()

    def forward(self, model, x_light, x_dark):
        B = x_light.shape[0]
        device = x_light.device

        # Sample interpolation point gamma in [0, 1]
        gamma = torch.rand(B, 1, 1, 1, device=device)
        x_gamma = gamma * x_light + (1.0 - gamma) * x_dark

        # Target: c(gamma) (x_dark - x_light)
        c = get_c_gamma(gamma, variant=self.variant, a=self.a, b=self.b, lambda_mult=self.lambda_mult)
        target = c * (x_dark - x_light)

        pred = model(torch.cat([x_gamma, x_dark], dim=1))

        return self.mse(pred, target)
