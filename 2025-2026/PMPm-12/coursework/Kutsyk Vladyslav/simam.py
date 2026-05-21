"""
simam.py
SimAM — Simple, Parameter-Free Attention Module.
Yang & Zheng, ICML 2021.

Реєструється у просторі імен Ultralytics перед побудовою моделі
(викликається з model_builder.py).
"""

import torch
import torch.nn as nn
from config import SIMAM_LAMBDA


class SimAM(nn.Module):
    """
    Параметрично-вільний 3D-модуль уваги.
    Не додає жодних навчуваних ваг і практично не впливає на FPS.

    Математика (формула з розд. 1):
        e*(t) = 4*(sigma^2 + lambda) / ((t - mu)^2 + 2*sigma^2 + 2*lambda)
        out   = x * sigmoid(1 / e*(x))
    """

    def __init__(self, e_lambda: float = SIMAM_LAMBDA):
        super().__init__()
        self.e_lambda = e_lambda
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        n = w * h - 1  # кількість просторових позицій мінус 1

        # Відхилення від просторового середнього
        mu     = x.mean(dim=[2, 3], keepdim=True)
        x_diff = (x - mu).pow(2)

        # Оцінка дисперсії по просторових осях
        sigma2 = x_diff.sum(dim=[2, 3], keepdim=True) / n

        # Енергетична функція (менше = більш інформативний нейрон)
        e_star = x_diff / (4.0 * (sigma2 + self.e_lambda)) + 0.5

        return x * self.sigmoid(e_star)

    def extra_repr(self) -> str:
        return f"e_lambda={self.e_lambda}"
