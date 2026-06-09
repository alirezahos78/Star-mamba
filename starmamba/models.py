import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as exc:
    raise ImportError("Please install mamba-ssm: pip install mamba-ssm") from exc


def _directional_scans(module_n, module_s, module_w, module_e, x, h, w):
    batch, length, dim = x.shape
    x_grid = x.view(batch, h, w, dim)

    inp_n = x_grid.permute(0, 2, 1, 3).reshape(batch * w, h, dim)
    out_n = module_n(inp_n).view(batch, w, h, dim).permute(0, 2, 1, 3).reshape(batch, length, dim)

    inp_s = torch.flip(x_grid.permute(0, 2, 1, 3), dims=[2]).reshape(batch * w, h, dim)
    out_s = module_s(inp_s).view(batch, w, h, dim)
    out_s = torch.flip(out_s, dims=[2]).permute(0, 2, 1, 3).reshape(batch, length, dim)

    inp_w = x_grid.reshape(batch * h, w, dim)
    out_w = module_w(inp_w).view(batch, h, w, dim).reshape(batch, length, dim)

    inp_e = torch.flip(x_grid, dims=[2]).reshape(batch * h, w, dim)
    out_e = module_e(inp_e).view(batch, h, w, dim)
    out_e = torch.flip(out_e, dims=[2]).reshape(batch, length, dim)

    return out_n, out_s, out_w, out_e


class StarMambaBlock(nn.Module):
    """Full Star-Mamba block with global, north/south, and west/east paths."""

    def __init__(self, dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.global_mamba = Mamba(d_model=dim)
        self.m_n = Mamba(d_model=dim)
        self.m_s = Mamba(d_model=dim)
        self.m_w = Mamba(d_model=dim)
        self.m_e = Mamba(d_model=dim)
        self.lam_n = nn.Parameter(torch.tensor([-2.0] * dim))
        self.lam_s = nn.Parameter(torch.tensor([-2.0] * dim))
        self.lam_w = nn.Parameter(torch.tensor([-2.0] * dim))
        self.lam_e = nn.Parameter(torch.tensor([-2.0] * dim))
        self.reg = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, h, w):
        shortcut = x
        x = self.norm(x)
        global_out = self.global_mamba(x)
        out_n, out_s, out_w, out_e = _directional_scans(self.m_n, self.m_s, self.m_w, self.m_e, x, h, w)

        local_sum = (
            out_n * F.softplus(self.lam_n)
            + out_s * F.softplus(self.lam_s)
            + out_w * F.softplus(self.lam_w)
            + out_e * F.softplus(self.lam_e)
        )

        x = shortcut + global_out + self.reg(local_sum)
        return x + self.mlp(self.norm2(x))


class NoGlobalStarMambaBlock(nn.Module):
    """Ablation block without the global Mamba path."""

    def __init__(self, dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.m_n = Mamba(d_model=dim)
        self.m_s = Mamba(d_model=dim)
        self.m_w = Mamba(d_model=dim)
        self.m_e = Mamba(d_model=dim)
        self.lam_n = nn.Parameter(torch.tensor([-2.0] * dim))
        self.lam_s = nn.Parameter(torch.tensor([-2.0] * dim))
        self.lam_w = nn.Parameter(torch.tensor([-2.0] * dim))
        self.lam_e = nn.Parameter(torch.tensor([-2.0] * dim))
        self.reg = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, h, w):
        shortcut = x
        x = self.norm(x)
        out_n, out_s, out_w, out_e = _directional_scans(self.m_n, self.m_s, self.m_w, self.m_e, x, h, w)
        local_sum = (
            out_n * F.softplus(self.lam_n)
            + out_s * F.softplus(self.lam_s)
            + out_w * F.softplus(self.lam_w)
            + out_e * F.softplus(self.lam_e)
        )
        x = shortcut + self.reg(local_sum)
        return x + self.mlp(self.norm2(x))


class NoEWStarMambaBlock(nn.Module):
    """Ablation block with global and north/south paths only."""

    def __init__(self, dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.global_mamba = Mamba(d_model=dim)
        self.m_n = Mamba(d_model=dim)
        self.m_s = Mamba(d_model=dim)
        self.lam_n = nn.Parameter(torch.tensor([-2.0] * dim))
        self.lam_s = nn.Parameter(torch.tensor([-2.0] * dim))
        self.reg = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, h, w):
        batch, length, dim = x.shape
        shortcut = x
        x = self.norm(x)
        global_out = self.global_mamba(x)
        x_grid = x.view(batch, h, w, dim)

        inp_n = x_grid.permute(0, 2, 1, 3).reshape(batch * w, h, dim)
        out_n = self.m_n(inp_n).view(batch, w, h, dim).permute(0, 2, 1, 3).reshape(batch, length, dim)

        inp_s = torch.flip(x_grid.permute(0, 2, 1, 3), dims=[2]).reshape(batch * w, h, dim)
        out_s = self.m_s(inp_s).view(batch, w, h, dim)
        out_s = torch.flip(out_s, dims=[2]).permute(0, 2, 1, 3).reshape(batch, length, dim)

        local_sum = out_n * F.softplus(self.lam_n) + out_s * F.softplus(self.lam_s)
        x = shortcut + global_out + self.reg(local_sum)
        return x + self.mlp(self.norm2(x))


class NoNSStarMambaBlock(nn.Module):
    """Ablation block with global and west/east paths only."""

    def __init__(self, dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.global_mamba = Mamba(d_model=dim)
        self.m_w = Mamba(d_model=dim)
        self.m_e = Mamba(d_model=dim)
        self.lam_w = nn.Parameter(torch.tensor([-2.0] * dim))
        self.lam_e = nn.Parameter(torch.tensor([-2.0] * dim))
        self.reg = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, h, w):
        batch, length, dim = x.shape
        shortcut = x
        x = self.norm(x)
        global_out = self.global_mamba(x)
        x_grid = x.view(batch, h, w, dim)

        inp_w = x_grid.reshape(batch * h, w, dim)
        out_w = self.m_w(inp_w).view(batch, h, w, dim).reshape(batch, length, dim)

        inp_e = torch.flip(x_grid, dims=[2]).reshape(batch * h, w, dim)
        out_e = self.m_e(inp_e).view(batch, h, w, dim)
        out_e = torch.flip(out_e, dims=[2]).reshape(batch, length, dim)

        local_sum = out_w * F.softplus(self.lam_w) + out_e * F.softplus(self.lam_e)
        x = shortcut + global_out + self.reg(local_sum)
        return x + self.mlp(self.norm2(x))


class OnlyGlobalMambaBlock(nn.Module):
    """Ablation block with only the global Mamba path."""

    def __init__(self, dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.global_mamba = Mamba(d_model=dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, h, w):
        del h, w
        x = x + self.global_mamba(self.norm(x))
        return x + self.mlp(self.norm2(x))


class VisionModel(nn.Module):
    """Patch-based vision classifier that stacks Star-Mamba style blocks."""

    def __init__(
        self,
        block_type=StarMambaBlock,
        img_size=32,
        patch_size=2,
        in_channels=3,
        dim=112,
        depth=6,
        num_classes=10,
    ):
        super().__init__()
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.h * self.w, dim) * 0.02)
        self.layers = nn.ModuleList([block_type(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x, self.h, self.w)
        x = self.norm(x).mean(dim=1)
        return self.head(x)
