"""Wrapper that aligns hybrid forward projection with RayNet orientation."""

import torch


def forward_spect(sigma_volume: torch.Tensor, *args, **kwargs):
    """
    Differentiable forward aligned with RayNet gamma_camera_core orientation chain.

    Expected input: sigma_volume (B, 1, SI, AP, LR). Currently assumes B=1.

    RayNet orientation chain (from preprocessing.py/gamma_camera_core):
        act_ap = transpose(0,2,1) -> rot90 -> transpose(1,0,2)
        act_pa = flip(act_ap, axis=2)
        proj = sum over AP axis
        orient_patch: rot90 -> flipud -> transpose

    This implementation mirrors that logic with torch ops so gradients flow.
    """
    if sigma_volume.ndim != 5:
        raise ValueError(f"sigma_volume must be (B,1,SI,AP,LR), got {tuple(sigma_volume.shape)}")
    if sigma_volume.shape[0] != 1:
        raise NotImplementedError("forward_spect currently expects B=1.")

    act_dhw = sigma_volume[0, 0]  # (SI, AP, LR)

    # RayNet expects (LR, AP, SI) before its internal rotations.
    act_xyz = act_dhw.permute(2, 1, 0)  # (LR, AP, SI)

    # MATLAB pre-rotations mirrored in torch.
    act_ap = act_xyz.permute(0, 2, 1)          # (LR, SI, AP)
    act_ap = torch.rot90(act_ap, k=1, dims=(0, 1))  # rot90 over LR/SI
    act_ap = act_ap.permute(1, 0, 2)           # (SI, LR, AP)
    act_pa = torch.flip(act_ap, dims=[2])      # flip along AP

    proj_ap = act_ap.sum(dim=2)  # (SI, LR)
    proj_pa = act_pa.sum(dim=2)  # (SI, LR)

    def orient_patch(P: torch.Tensor) -> torch.Tensor:
        P1 = torch.rot90(P, k=1, dims=(0, 1))
        P2 = torch.flip(P1, dims=[0])
        return P2.t()

    ap_pred = orient_patch(proj_ap)
    pa_pred = orient_patch(proj_pa)

    # Final alignment to dataset convention (SI, LR) = (act_dhw.shape[0], act_dhw.shape[2]).
    # Empirically, flipping both axes aligns with RayNet-generated GT.
    ap_pred = torch.flip(ap_pred, dims=[0, 1])
    pa_pred = torch.flip(pa_pred, dims=[0, 1])

    si_target, _, lr_target = act_dhw.shape
    if ap_pred.shape == (lr_target, si_target):
        ap_pred = ap_pred.t()
    if pa_pred.shape == (lr_target, si_target):
        pa_pred = pa_pred.t()
    return ap_pred, pa_pred
