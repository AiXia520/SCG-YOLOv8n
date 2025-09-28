import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility blocks from Ultralytics style
# -----------------------------
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, (k // 2 if p is None else p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class DWConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c1, c1, k, s, k//2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(),
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU() if act else nn.Identity(),
        )
    def forward(self, x):
        return self.conv(x)

# -----------------------------
# (1) C‑SPD: Contextual Space‑to‑Depth
# -----------------------------
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(c // r, c, 1, bias=True),
            nn.Sigmoid(),
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class CSPD(nn.Module):
    """Space-to-Depth (block=2) + SE + depthwise separable conv, without stride.
    Keeps more spatial detail than strided conv/pool.
    """
    def __init__(self, c_out):
        super().__init__()
        self.c_out = c_out
        self.se = SE(c_out * 4)
        self.dw = DWConv(c_out * 4, c_out)

    def forward(self, x):
        # Space-to-Depth (block=2)
        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, "H,W must be even for CSPD"
        x = x.view(B, C, H//2, 2, W//2, 2).permute(0,1,3,5,2,4).contiguous()
        x = x.view(B, C*4, H//2, W//2)
        x = self.se(x)
        return self.dw(x)

# -----------------------------
# (2) S‑CARAFE: lightweight content‑aware upsampling (2x)
# -----------------------------
class SCARAFE(nn.Module):
    def __init__(self, c_mid, k_enc=3, k_up=5):
        super().__init__()
        self.reduce = Conv(c_mid, c_mid//2, 1, 1)
        self.kernel_enc = nn.Sequential(
            Conv(c_mid//2, c_mid//2, k_enc, 1),
            nn.Conv2d(c_mid//2, (k_up*k_up), 1)  # per‑pixel kernel logits
        )
        self.k_up = k_up

    def forward(self, x):
        B, C, H, W = x.shape
        r = 2
        # predict kernels on low‑dim features
        f = self.reduce(x)
        k_logits = self.kernel_enc(f)  # [B, k^2, H, W]
        k = F.softmax(k_logits, dim=1)
        # unfold input patches
        pad = self.k_up // 2
        unfold = F.unfold(x, kernel_size=self.k_up, padding=pad)  # [B, C*k^2, HW]
        unfold = unfold.view(B, C, self.k_up*self.k_up, H*W)
        # weighted sum
        k = k.view(B, 1, self.k_up*self.k_up, H*W)  # broadcast over C
        out = (unfold * k).sum(dim=2)  # [B, C, HW]
        out = out.view(B, C, H, W)
        # 2x upscale by nearest + refinement (fast path)
        out = F.interpolate(out, scale_factor=2, mode="nearest")
        return out

# -----------------------------
# (3) GhostShuffleConv for detection head
# -----------------------------
class GhostModule(nn.Module):
    def __init__(self, c1, c2, ratio=2, k=1, dw_k=3):
        super().__init__()
        c_prim = int(c2 / ratio)
        c_ghost = c2 - c_prim
        self.primary = Conv(c1, c_prim, k, 1)
        self.cheap = nn.Sequential(
            nn.Conv2d(c_prim, c_ghost, dw_k, 1, dw_k//2, groups=c_prim, bias=False),
            nn.BatchNorm2d(c_ghost),
            nn.SiLU(),
        )
    def forward(self, x):
        p = self.primary(x)
        g = self.cheap(p)
        return torch.cat([p, g], dim=1)

class ChannelShuffle(nn.Module):
    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups
    def forward(self, x):
        B, C, H, W = x.shape
        g = self.groups
        assert C % g == 0
        x = x.view(B, g, C//g, H, W).permute(0,2,1,3,4).contiguous()
        return x.view(B, C, H, W)

class GhostShuffle(nn.Module):
    """Dual-branch Ghost + GhostDW + channel shuffle, then simple prediction heads per scale.
    Expects list of pyramid features; returns list of [B, anchors, (cx,cy,w,h,conf,classes...)].
    For simplicity we wrap a Conv predictor per feature map.
    """
    def __init__(self, nc: int, c_in=(256, 512, 1024)):
        super().__init__()
        self.nc = nc
        self.branches = nn.ModuleList()
        self.pred = nn.ModuleList()
        for c in c_in:
            branch = nn.Sequential(
                GhostModule(c, c),
                ChannelShuffle(2),
                GhostModule(c, c),
                ChannelShuffle(2),
            )
            self.branches.append(branch)
            self.pred.append(nn.Conv2d(c, nc + 4 + 1, 1))  # [bbox(4) + obj + nc]

    def forward(self, feats):  # feats: [P3, P4, P5]
        out = []
        for f, br, pr in zip(feats, self.branches, self.pred):
            y = br(f)
            out.append(pr(y))
        return out