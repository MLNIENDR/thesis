"""Wrapper for SPECT forward projection (placeholder)."""

import torch


def forward_spect(sigma_volume: torch.Tensor, mu_volume: torch.Tensor, *args, **kwargs):
    """
    Minimal placeholder for SPECT forward projection.

    Intended to:
        - Call pieNeRF SPECT routines (attenuation, PSF, scatter) — NOT implemented here.
        - Consume sigma (NeRF-predicted density) (D, H, W) and mu (attenuation from CT) (D, H, W).
        - Produce AP/PA projections ap_pred, pa_pred each shaped (H, W).

    Current behavior (placeholder):
        - Ignores mu_volume (no attenuation yet).
        - Projects by summing along the SI axis:
            ap_pred = sigma_volume.sum(dim=0)  # (H, W)
            pa_pred = sigma_volume.sum(dim=0)  # (H, W)
    """
    if sigma_volume.ndim != 3:
        raise ValueError(f"sigma_volume must be (D, H, W), got {tuple(sigma_volume.shape)}")
    # Placeholder projection: simple sum over SI dimension.
    ap_pred = sigma_volume.sum(dim=0)
    pa_pred = sigma_volume.sum(dim=0)
    return ap_pred, pa_pred
