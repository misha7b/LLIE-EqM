"""
EqMNet v2 — Improvements over v1:
  1. Post-PixelShuffle depthwise smoothing to suppress checkerboard artifacts.
  2. No gamma conditioning (removed entirely).
  3. 3x3 skip fusion conv (was 1x1) for spatial context when blending scales.
  4. 3x3 output head (was 1x1) for spatially coherent gradient field.

Checkpoint-compatible with v1 via strict=False:
  - Gamma-related keys in the old checkpoint are simply ignored.
  - New smoother layers are identity-initialized (no effect before fine-tuning).
  - Changed conv shapes (fuse, head) will use fresh random init.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ──────────────────────────────────────────────────────────


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


# ── Channel Attention (SE-style) ─────────────────────────────────────


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)[:, :, None, None]


# ── Core building block: EqMBlock ────────────────────────────────────


class SimpleGate(nn.Module):
    """Split channels in half, multiply: x_a * x_b.  No learned params."""

    def forward(self, x):
        x_a, x_b = x.chunk(2, dim=1)
        return x_a * x_b


class EqMBlock(nn.Module):
    """
    NAFNet-inspired block.

    DWConv -> SimpleGate -> ChannelAttention -> residual
    then LayerNorm -> FFN -> residual.
    """

    def __init__(self, channels, ffn_expansion=2):
        super().__init__()

        # -- spatial mixing --
        self.norm1 = LayerNorm2d(channels)
        self.dwconv = nn.Conv2d(channels, channels * 2, kernel_size=3,
                                padding=1, groups=channels, bias=True)
        self.gate = SimpleGate()
        self.ca = ChannelAttention(channels)

        # -- feed-forward --
        self.norm2 = LayerNorm2d(channels)
        hidden = channels * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden * 2, 1),
            SimpleGate(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.dwconv(x)
        x = self.gate(x)
        x = self.ca(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual

        return x


# ── Transposed Multi-Head Attention (Restormer MDTA) ─────────────────


class TransposedAttention(nn.Module):
    """
    Restormer-style multi-DConv head transposed attention.
    Computes attention over the *channel* dimension (C x C map).
    Cost: O(C^2 * HW) -- linear in spatial size.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm2d(channels)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(channels * 3, channels * 3, 3,
                                padding=1, groups=channels * 3, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        heads = self.num_heads
        q = q.reshape(B, heads, C // heads, H * W)
        k = k.reshape(B, heads, C // heads, H * W)
        v = v.reshape(B, heads, C // heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)

        return self.proj(out) + residual


# ── Cross-Attention (state queries dark features) ────────────────────


class CrossAttention(nn.Module):
    """
    Transposed cross-attention: state features attend to dark features
    over the *channel* dimension (C x C map), not spatial (HW x HW).
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm_q = LayerNorm2d(channels)
        self.norm_kv = LayerNorm2d(channels)

        self.to_q = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        )
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, state, dark):
        B, C, H, W = state.shape
        residual = state

        q = self.to_q(self.norm_q(state))
        k = self.to_k(self.norm_kv(dark))
        v = self.to_v(dark)

        heads = self.num_heads
        q = q.reshape(B, heads, C // heads, H * W)
        k = k.reshape(B, heads, C // heads, H * W)
        v = v.reshape(B, heads, C // heads, H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v).reshape(B, C, H, W)

        return self.proj(out) + residual


# ── Anti-checkerboard smoothing layer ────────────────────────────────


class PostShuffleSmoother(nn.Module):
    """
    Depthwise 3x3 conv applied after PixelShuffle to smooth out
    checkerboard artifacts.  Initialized to identity (center=1, rest=0)
    so the model produces identical output to v1 before fine-tuning.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1,
                              groups=channels, bias=False)
        nn.init.zeros_(self.conv.weight)
        with torch.no_grad():
            self.conv.weight[:, :, 1, 1] = 1.0

    def forward(self, x):
        return self.conv(x)


# ── Encoder / Decoder stages ─────────────────────────────────────────


class DownBlock(nn.Module):
    """N EqMBlocks then pixel-unshuffle to halve spatial dims."""

    def __init__(self, in_ch, out_ch, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            EqMBlock(in_ch) for _ in range(num_blocks)
        ])
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_ch * 4, out_ch, 1),
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        x = self.downsample(x)
        return x, skip


class AdditiveFusion(nn.Module):
    """Cheap dark-feature injection: project + add.  O(C*HW), no attention."""

    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, state, dark):
        if dark.shape[2:] != state.shape[2:]:
            dark = F.interpolate(dark, size=state.shape[2:],
                                 mode='bilinear', align_corners=False)
        return state + self.proj(self.norm(dark))


class UpBlock(nn.Module):
    """Pixel-shuffle to double spatial dims, smooth, concat skip, N EqMBlocks."""

    def __init__(self, in_ch, skip_ch, out_ch, num_blocks=2,
                 dark_fuse="none", cross_heads=4):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 1),
            nn.PixelShuffle(2),
        )
        # depthwise 3x3 smoother after pixel-shuffle to kill grid artifacts
        self.smooth = PostShuffleSmoother(out_ch)

        # 3x3 skip fusion for spatial context when blending scales
        self.fuse = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)

        if dark_fuse == "cross_attn":
            self.dark_fuse = CrossAttention(out_ch, num_heads=cross_heads)
        elif dark_fuse == "additive":
            self.dark_fuse = AdditiveFusion(out_ch)
        else:
            self.dark_fuse = None

        self.blocks = nn.ModuleList([
            EqMBlock(out_ch) for _ in range(num_blocks)
        ])

    def forward(self, x, skip, dark_feat=None):
        x = self.upsample(x)
        x = self.smooth(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        x = self.fuse(torch.cat([x, skip], dim=1))

        if self.dark_fuse is not None and dark_feat is not None:
            if dark_feat.shape[2:] != x.shape[2:]:
                dark_feat = F.interpolate(dark_feat, size=x.shape[2:],
                                          mode='bilinear', align_corners=False)
            x = self.dark_fuse(x, dark_feat)

        for blk in self.blocks:
            x = blk(x)
        return x


# ── Dark Encoder (lightweight) ───────────────────────────────────────


class DarkEncoder(nn.Module):
    """
    Extracts multi-scale features from x_dark.
    At inference this runs ONCE; features are cached and reused for
    every descent step.
    """

    def __init__(self, in_ch=3, base=32):
        super().__init__()
        C, C2, C4 = base, base * 2, base * 4

        self.stem = nn.Conv2d(in_ch, C, 3, padding=1)
        self.block0 = EqMBlock(C)

        self.down1 = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(C * 4, C2, 1))
        self.block1 = EqMBlock(C2)

        self.down2 = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(C2 * 4, C4, 1))
        self.block2 = EqMBlock(C4)

        self.out_channels = [C, C2, C4]

    def forward(self, x_dark):
        x = self.stem(x_dark)
        f0 = self.block0(x)
        x = self.down1(f0)
        f1 = self.block1(x)
        x = self.down2(f1)
        f2 = self.block2(x)
        return [f0, f1, f2]


# ── Full EqMNet ──────────────────────────────────────────────────────


class EqMNet(nn.Module):
    """
    EqMNet v2: dual-encoder architecture for Equilibrium Matching LLIE.

    Changes from v1:
      - No gamma conditioning (simpler, fewer params)
      - Post-PixelShuffle smoothing (anti-checkerboard)
      - 3x3 skip fusion and output head (better spatial coherence)

    Forward signature:
        model(x)           -- x is (B, 6, H, W), split internally.
        model.encode_dark(x_dark)                -- pre-compute dark features.
        model.forward_with_cache(x_gamma, feats) -- use cached dark features.
    """

    def __init__(self, in_ch=3, out_ch=3, base=32,
                 enc_blocks=2, dec_blocks=2,
                 bottleneck_attn=4, bottleneck_heads=4):
        super().__init__()
        C = base
        C2 = C * 2
        C4 = C * 4

        self.dark_encoder = DarkEncoder(in_ch=in_ch, base=C)

        self.state_stem = nn.Conv2d(in_ch, C, 3, padding=1)
        self.enc1 = DownBlock(C, C2, num_blocks=enc_blocks)
        self.enc2 = DownBlock(C2, C4, num_blocks=enc_blocks)

        self.bottleneck_blocks = nn.ModuleList([
            TransposedAttention(C4, num_heads=bottleneck_heads)
            for _ in range(bottleneck_attn)
        ])
        self.bottleneck_cross = CrossAttention(C4, num_heads=bottleneck_heads)
        self.bottleneck_conv = nn.ModuleList([
            EqMBlock(C4) for _ in range(2)
        ])

        self.dark_proj_bottleneck = nn.Conv2d(C4, C4, 1)
        self.dark_proj_scale1 = nn.Conv2d(C2, C2, 1)
        self.dark_proj_scale0 = nn.Conv2d(C, C, 1)

        self.dec2 = UpBlock(C4, C2, C2, num_blocks=dec_blocks,
                            dark_fuse="cross_attn", cross_heads=bottleneck_heads)
        self.dec1 = UpBlock(C2, C, C, num_blocks=dec_blocks,
                            dark_fuse="additive")

        # 3x3 output head for spatially coherent gradient field
        self.head = nn.Conv2d(C, out_ch, 3, padding=1)

    def encode_dark(self, x_dark):
        """Pre-compute dark features.  Call once per image at inference."""
        return self.dark_encoder(x_dark)

    def forward_with_cache(self, x_gamma, dark_feats):
        d0 = self.dark_proj_scale0(dark_feats[0])
        d1 = self.dark_proj_scale1(dark_feats[1])
        d2 = self.dark_proj_bottleneck(dark_feats[2])

        x = self.state_stem(x_gamma)
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)

        for sa in self.bottleneck_blocks:
            x = sa(x)
        x = self.bottleneck_cross(x, d2)
        for blk in self.bottleneck_conv:
            x = blk(x)

        x = self.dec2(x, skip2, dark_feat=d1)
        x = self.dec1(x, skip1, dark_feat=d0)

        return self.head(x)

    def forward(self, x):
        x_gamma = x[:, :3]
        x_dark = x[:, 3:]
        dark_feats = self.encode_dark(x_dark)
        return self.forward_with_cache(x_gamma, dark_feats)


# ── Convenience constructors ─────────────────────────────────────────


def eqmnet2_small(**kwargs):
    """~1.3M params."""
    return EqMNet(base=32, enc_blocks=2, dec_blocks=2,
                  bottleneck_attn=4, bottleneck_heads=4, **kwargs)


def eqmnet2_base(**kwargs):
    """~2.6M params."""
    return EqMNet(base=48, enc_blocks=2, dec_blocks=2,
                  bottleneck_attn=4, bottleneck_heads=8, **kwargs)


def eqmnet2_medium(**kwargs):
    """~3.2M params."""
    return EqMNet(base=48, enc_blocks=3, dec_blocks=3,
                  bottleneck_attn=6, bottleneck_heads=8, **kwargs)


def eqmnet2_large(**kwargs):
    """~5.4M params."""
    return EqMNet(base=64, enc_blocks=3, dec_blocks=3,
                  bottleneck_attn=6, bottleneck_heads=8, **kwargs)


def eqmnet2_xl(**kwargs):
    """~6.3M params."""
    return EqMNet(base=64, enc_blocks=4, dec_blocks=4,
                  bottleneck_attn=8, bottleneck_heads=8, **kwargs)


# ── Quick sanity check ───────────────────────────────────────────────

if __name__ == "__main__":
    device = "cpu"

    for name, fn in [("Small", eqmnet2_small), ("Base", eqmnet2_base),
                     ("Medium", eqmnet2_medium), ("Large", eqmnet2_large),
                     ("XL", eqmnet2_xl)]:
        model = fn().to(device)
        n = sum(p.numel() for p in model.parameters()) / 1e6
        x = torch.randn(1, 6, 128, 128, device=device)
        out = model(x)
        print(f"EqMNet2-{name:6s}  {n:6.2f}M  ->  {out.shape}")
