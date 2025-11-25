"""Top-level SpectPixelPieNeRF model."""

import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from model.encoders.ct_pixelnerf_encoder import CTPixelNeRFEncoder
from model.nerf.pienerf_mlp_cond import PieNeRFConditionalMLP
from utils.geometry.ray_sampler_spect import sample_spect_rays
from forward.spect_operator_wrapper import forward_spect

# Import positional encoder from pieNeRF helpers.
_PIENERF_ROOT = Path(__file__).resolve().parents[1] / "pieNeRF"
if _PIENERF_ROOT.exists():
    sys.path.append(str(_PIENERF_ROOT))
try:
    from nerf.run_nerf_helpers_mod import get_embedder  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "Could not import get_embedder from pieNeRF. Ensure pieNeRF is on PYTHONPATH."
    ) from exc


class SpectPixelPieNeRF(nn.Module):
    """
    Pipeline:
        ct, ap, pa
          → CTPixelNeRFEncoder → z_feat (B, latent_dim)
          → sample_spect_rays (AP/PA) → xyz
          → positional encoding xyz
          → PieNeRFConditionalMLP → sigma
          → sigma_volume (D, H, W)
          → forward_spect → AP/PA images
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        voxel_size: float | Tuple[float, float, float],
        num_samples: int,
        latent_dim: int = 512,
        mlp_width: int = 256,
        mlp_depth: int = 8,
        skips: Optional[list[int]] = None,
        multires_xyz: int = 6,
    ):
        super().__init__()
        self.volume_shape = volume_shape
        self.voxel_size = voxel_size
        self.num_samples = num_samples
        self.latent_dim = latent_dim

        self.encoder = CTPixelNeRFEncoder(backbone="resnet18", num_layers=4, latent_channels=latent_dim)
        self.embed_xyz, embed_xyz_out_dim = get_embedder(multires_xyz, i=0)
        self.mlp = PieNeRFConditionalMLP(
            input_ch_xyz=embed_xyz_out_dim,
            latent_dim=latent_dim,
            D=mlp_depth,
            W=mlp_width,
            skips=skips or [4],
        )

    def forward(
        self,
        ct: torch.Tensor,   # (B,1,D,H,W)
        ap: torch.Tensor,   # (B,1,H,W)
        pa: torch.Tensor,   # (B,1,H,W)
        mu_volume: Optional[torch.Tensor] = None,  # (B,D,H,W) or None
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            {
              "ap_pred": (B,1,H,W),
              "pa_pred": (B,1,H,W),
              "sigma_volume": (B,1,D,H,W)
            }
        """
        B, _, D, H, W = ct.shape
        if B != 1:
            raise NotImplementedError("Current forward assumes B=1; extend batching later.")
        if self.num_samples != H:
            raise AssertionError("Assumption num_samples == H (AP dimension) not satisfied.")

        device = ct.device
        dtype = ct.dtype

        # Encoder: z_feat (B, latent_dim)
        z_feat = self.encoder(ct, ap, pa)

        # Ray sampling AP/PA
        xyz_ap, _ = sample_spect_rays(self.volume_shape, self.voxel_size, "AP", self.num_samples, device=device, dtype=dtype)
        xyz_pa, _ = sample_spect_rays(self.volume_shape, self.voxel_size, "PA", self.num_samples, device=device, dtype=dtype)

        # Positional encoding
        xyz_ap_pe = self.embed_xyz(xyz_ap)
        xyz_pa_pe = self.embed_xyz(xyz_pa)

        # MLP to sigma
        sigma_ap = self.mlp(xyz_ap_pe, z_feat)  # (N,1)
        sigma_pa = self.mlp(xyz_pa_pe, z_feat)  # (N,1)

        # Reshape to volumes (D, H, W)
        sigma_ap_grid = sigma_ap.view(D, W, H).permute(0, 2, 1)
        sigma_pa_grid = sigma_pa.view(D, W, H).permute(0, 2, 1)
        sigma_volume = 0.5 * (sigma_ap_grid + sigma_pa_grid)
        sigma_volume = sigma_volume.unsqueeze(0).unsqueeze(1)  # (B,1,D,H,W)

        if mu_volume is None:
            mu_volume = ct[:, 0]  # (B,D,H,W)

        # Forward operator (placeholder): expects per-batch (D,H,W)
        ap_pred, pa_pred = forward_spect(sigma_volume[0, 0], mu_volume[0])
        ap_pred = ap_pred.unsqueeze(0).unsqueeze(0)  # (B,1,H,W)
        pa_pred = pa_pred.unsqueeze(0).unsqueeze(0)  # (B,1,H,W)

        return {
            "ap_pred": ap_pred,
            "pa_pred": pa_pred,
            "sigma_volume": sigma_volume,
        }
