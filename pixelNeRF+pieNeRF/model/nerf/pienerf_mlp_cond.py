"""Conditional pieNeRF MLP with latent concatenation."""

from functools import partial

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class PieNeRFConditionalMLP(nn.Module):
    """
    pieNeRF MLP extended with latent conditioning via concatenation.

    Inputs:
        xyz_pe: positional-encoded 3D coordinates, shape (N, C_xyz_pe).
        z_feat: global latent vector from encoder, shape (B, C_latent); in this step we assume B=1.

    Forward:
        - Take z_feat[0], expand to (N, C_latent).
        - Concatenate with xyz_pe → (N, C_xyz_pe + C_latent).
        - Pass through NeRF-style MLP (no viewdir branch) with skips.

    Output:
        sigma: density values sigma(x), shape (N, 1). Attenuation (mu) is derived from CT elsewhere.
    """

    def __init__(
        self,
        input_ch_xyz: int,
        latent_dim: int = 512,
        D: int = 8,
        W: int = 256,
        skips: list[int] | None = None,
    ):
        """
        Args:
            input_ch_xyz: dimensionality of positional-encoded xyz.
            latent_dim: dimensionality of global latent z_feat.
            D: depth (number of linear layers).
            W: width (hidden units).
            skips: skip connection indices (defaults to [4] like NeRF).
        """
        super().__init__()
        if skips is None:
            skips = [4]

        self.D = D
        self.W = W
        self.input_ch = input_ch_xyz + latent_dim
        self.skips = skips

        # Point MLP layers with skip concatenation similar to NeRF (no viewdirs).
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)]
            + [
                nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W)
                for i in range(D - 1)
            ]
        )

        self.output_linear = nn.Linear(W, 1)
        with torch.no_grad():
            self.output_linear.bias.fill_(1e-3)
            self.output_linear.weight.mul_(0.01)

    def forward(self, xyz_pe: Tensor, z_feat: Tensor) -> Tensor:
        """
        Compute sigma given positional-encoded xyz and latent code.

        Args:
            xyz_pe: Tensor (N, C_xyz_pe) positional-encoded coordinates.
            z_feat: Tensor (B, C_latent) global latent; currently only z_feat[0] is used (assumes B=1).

        Returns:
            sigma: Tensor (N, 1) predicted densities sigma(x).
        """
        if z_feat.ndim != 2:
            raise ValueError(f"z_feat must be (B, C_latent); got shape {tuple(z_feat.shape)}")
        if z_feat.shape[0] < 1:
            raise ValueError("z_feat batch dimension is empty.")

        # For now, use the first batch element; future work can map xyz_pe to matching batch indices.
        latent = z_feat[0]  # (C_latent,)
        latent_expanded = latent.expand(xyz_pe.shape[0], -1)  # (N, C_latent)

        h0 = torch.cat([xyz_pe, latent_expanded], dim=-1)  # (N, input_ch)
        h = h0
        relu = partial(F.relu, inplace=True)
        for i, _ in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = relu(h)
            if i in self.skips:
                h = torch.cat([h0, h], dim=-1)

        sigma = self.output_linear(h)  # (N, 1)
        return sigma
