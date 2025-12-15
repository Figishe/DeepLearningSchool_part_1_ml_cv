import torch
import torch.nn as nn
import math

class HSVSimpleLoss(nn.Module):
    def __init__(self, hue_weight=1.0, s_weight=1.0, v_weight=1.0):
        super().__init__()
        self.hue_weight = hue_weight
        self.s_weight = s_weight
        self.v_weight = v_weight
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        
        # pred, target: [B, 3, H, W], H/S/V нормализованы
        # Hue: циклическая разница
        dH = torch.abs(pred[:,0] - target[:,0])
        dH = torch.min(dH, 1 - dH)  # учитываем кольцо

        loss_H = self.mse(dH, torch.zeros_like(dH))
        loss_S = self.mse(pred[:,1], target[:,1])
        loss_V = self.mse(pred[:,2], target[:,2])

        return self.hue_weight*loss_H + self.s_weight*loss_S + self.v_weight*loss_V


class HSVPerceptualLoss(nn.Module):
    def __init__(self, hue_weight=1.0, s_weight=1.0, v_weight=1.0,
                 hue_green_factor=0.5, s_slope=0.7, v_exponent=0.5, eps=1e-6):
        super().__init__()
        self.hue_weight = hue_weight
        self.s_weight = s_weight
        self.v_weight = v_weight
        self.hue_green_factor = hue_green_factor
        self.s_slope = s_slope
        self.v_exponent = v_exponent
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # --- Hue
        dH = torch.remainder(pred[:,0] - target[:,0] + 0.5, 1.0) - 0.5
        dH = dH.abs()

        h = target[:,0]
        green_weight = 1.0 + self.hue_green_factor * torch.cos((h - 1/3) * 2 * math.pi)
        s_weight_hue = torch.clamp(target[:,1], min=0.05) 
        loss_H = ((green_weight * s_weight_hue * dH)**2).mean()

        # --- Saturation
        s_weight = (1 - self.s_slope) + self.s_slope * target[:,1] 
        s_weight = torch.clamp(s_weight, min=0.05)
        loss_S = ((s_weight * (pred[:,1] - target[:,1]))**2).mean()

        # --- Value
        v = torch.clamp(pred[:,2], min=self.eps, max=1.0 - self.eps)
        v_weight = ((target[:,2]**self.v_exponent) * (1 - target[:,2]**self.v_exponent)) * 4
        v_weight = torch.clamp(v_weight, min=self.eps)
        loss_V = (v_weight * (pred[:,2] - target[:,2])**2).mean()

        return self.hue_weight*loss_H + self.s_weight*loss_S + self.v_weight*loss_V
