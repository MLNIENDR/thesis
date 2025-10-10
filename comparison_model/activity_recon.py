"""
AP/PA Activity Reconstruction with CT-based Attenuation and Organ-Aware Regularization
=====================================================================================

This script provides a **reference PyTorch implementation** of the optimization:
- Data terms for AP & PA projections including attenuation from CT (Beer–Lambert)
- Regularization that (1) preserves organ boundaries using CT edges (edge-aware TV)
  and (2) encourages *uniform filling within each organ* (intra-organ variance penalty)
- Non-negativity constraint on the activity x

The code is structured to be a drop-in starting point (by default, the script ships with a DummyProjector
that implements H(x) = A @ x (matrix multiply) so the code is runnable with any precomputed
system matrix `A`). In the pipeline, `DummyProjector` has to be replaced with bindings to the
actual AP/PA forward projector that already models attenuation along rays.


How to use
----------
1) Prepare inputs (NumPy or PyTorch tensors):
   - y_AP, y_PA: measured projections as 1D or 2D arrays (will be flattened internally)
   - CT_mu: 3D volume of linear attenuation coefficients [1/mm] or [1/cm]
   - masks: list of boolean 3D arrays (same shape as CT_mu), one per organ
   - spacing: voxel spacing (dz, dy, dx) in mm (affects TV weighting)

2) Provide forward projectors for AP and PA views. Two options:
   A) Use `DummyProjector(A)` with a precomputed system matrix A (shape [M, N]) where
      N = num_voxels and M = num_detector_pixels. (A should already include attenuation.)
   B) Implement your own subclass of `BaseProjector` that computes H(CT_mu, x).

3) Choose the optimization mode:
   - "organ_scalars_bg": estimate one scalar per organ + a smooth background field
   - "voxelwise": estimate a full voxel-wise activity with organ-variance + edge-aware TV

4) Run `main()`

Notes
-----
- Attenuation: In many pipelines you already fold exp(-∫μ dl) into the projector.
  If not, expose a `ray_integrate_mu()` in your projector and multiply path-sensitivities
  accordingly. This template keeps attenuation inside the projector abstraction.
- Statistics: Data term uses Poisson-NLL by default; WLS also provided.
- Scatter: If needed, add simple per-view scatter offsets as learnable scalars.

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

# ------------------------------------------------------------
# Projector abstraction
# ------------------------------------------------------------

class BaseProjector:
    """Abstract projector interface: y_hat = H(CT_mu, x).
    Replace DummyProjector with your AP/PA forward model.
    """
    def __init__(self, name: str = "H"):
        self.name = name

    def __call__(self, CT_mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def M(self) -> int:
        raise NotImplementedError

    @property
    def N(self) -> int:
        raise NotImplementedError


class DummyProjector(BaseProjector):
    """Matrix-based projector: y = A @ x
    - A is a dense or sparse torch tensor of shape [M, N]
    - Assumes attenuation is already included in A (or negligible)
    - For prototyping and unit tests
    """
    def __init__(self, A: torch.Tensor, name: str = "H"):
        super().__init__(name)
        assert A.ndim == 2, "A must be [M, N]"
        self.A = A

    def __call__(self, CT_mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # CT_mu is unused in this dummy (attenuation already baked into A)
        return self.A @ x

    @property
    def M(self) -> int:
        return self.A.shape[0]

    @property
    def N(self) -> int:
        return self.A.shape[1]


# ------------------------------------------------------------
# Loss terms: Data fidelity (Poisson NLL and WLS)
# ------------------------------------------------------------

def poisson_nll(y: torch.Tensor, yhat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Poisson negative log-likelihood up to const(y!)
    L = sum( yhat - y * log(yhat+eps) )
    """
    return (yhat - y * torch.log(yhat + eps)).sum()


def wls_loss(y: torch.Tensor, yhat: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted least squares: sum( w * (yhat - y)^2 )
    If weight is None, uses 1/max(y,1) as variance proxy.
    """
    r = (yhat - y)
    if weight is None:
        weight = 1.0 / torch.clamp(y, min=1.0)
    return (weight * r * r).sum()


# ------------------------------------------------------------
# Regularizers: edge-aware TV and intra-organ variance
# ------------------------------------------------------------

def _gradient_3d(x: torch.Tensor, spacing: Tuple[float, float, float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward differences with physical scaling.
    x: [D,H,W]
    spacing: (dz, dy, dx) in mm. Returns (gx, gy, gz) with units per mm.
    """
    dz, dy, dx = spacing
    # pad last voxel with zeros (Neumann BC); gradients set to 0 on the boundary
    gx = (F.pad(x[1:, :, :], (0,0,0,0,0,1)) - x) / dz
    gy = (F.pad(x[:, 1:, :], (0,0,0,0,0,0,0,1)) - x) / dy
    gz = (F.pad(x[:, :, 1:], (0,1,0,0,0,0)) - x) / dx
    return gx, gy, gz


def edge_weights_from_ct(ct: torch.Tensor, spacing: Tuple[float, float, float], alpha: float = 10.0) -> torch.Tensor:
    """Compute edge-aware weights w = exp(-alpha * |∇CT|)
    - Larger CT gradients -> smaller weights -> less smoothing across boundaries
    """
    gx, gy, gz = _gradient_3d(ct, spacing)
    grad_mag = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    w = torch.exp(-alpha * grad_mag)
    return w


def edge_aware_tv(x: torch.Tensor, w: torch.Tensor, spacing: Tuple[float, float, float]) -> torch.Tensor:
    """Isotropic, edge-aware total variation: sum( w * ||∇x||_2 )
    x: [D,H,W], w: [D,H,W]
    """
    gx, gy, gz = _gradient_3d(x, spacing)
    tv = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    return (w * tv).sum()


def intra_organ_variance(x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    """Sum of L2 deviations from each organ's mean (encourages uniform filling).
    x: [D,H,W], masks: list of [D,H,W] boolean tensors
    """
    loss = 0.0
    for mk in masks:
        mkf = mk.float()
        nk = mkf.sum().clamp(min=1.0)
        mean_k = (x * mkf).sum() / nk
        loss += ((x - mean_k) ** 2 * mkf).sum()
    return loss


def tv_on_mask(x: torch.Tensor, mask: torch.Tensor, spacing: Tuple[float, float, float]) -> torch.Tensor:
    """Plain TV applied only on a given mask (e.g., background)."""
    gx, gy, gz = _gradient_3d(x, spacing)
    tv = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    return (tv * mask.float()).sum()


# ------------------------------------------------------------
# Parameterizations
# ------------------------------------------------------------

def init_from_organs(masks: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Simple initialization: set each organ to a small positive constant, bg=0
    Returns x0 of shape [D,H,W].
    """
    assert len(masks) > 0
    shape = masks[0].shape
    x0 = torch.zeros(shape, dtype=torch.float32, device=device)
    for mk in masks:
        x0 = x0 + 0.1 * mk.float()  # 0.1 is arbitrary; adjust later
    return x0


def flatten_volume(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def unflatten_volume(x_flat: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    return x_flat.reshape(shape)


# ------------------------------------------------------------
# Config and main optimizer routine
# ------------------------------------------------------------

@dataclass
class ReconConfig:
    mode: str = "voxelwise"  # "voxelwise" or "organ_scalars_bg"
    lambda_tv: float = 1e-3
    lambda_org: float = 1e-2
    lambda_bg: float = 1e-3
    alpha_edges: float = 10.0
    lr: float = 1e-2
    iters: int = 500
    use_poisson: bool = True  # if False, use WLS
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_optimization(
    y_AP: torch.Tensor,
    y_PA: torch.Tensor,
    H_AP: BaseProjector,
    H_PA: BaseProjector,
    CT_mu: torch.Tensor,
    organ_masks: List[torch.Tensor],
    bg_mask: Optional[torch.Tensor],
    spacing: Tuple[float, float, float],
    cfg: ReconConfig,
) -> Tuple[torch.Tensor, dict]:
    """Run the reconstruction/estimation.

    Inputs
    ------
    y_AP, y_PA : tensors of shape [M] or [Hdet, Wdet] -> will be flattened
    H_AP, H_PA : projector instances
    CT_mu      : [D,H,W]
    organ_masks: list of [D,H,W] boolean tensors
    bg_mask    : [D,H,W] boolean (outside all organs), can be None
    spacing    : (dz, dy, dx) in mm
    cfg        : hyper-params

    Returns
    -------
    x_est      : [D,H,W] estimated activity
    logs       : dict with simple traces (loss values)
    """
    device = torch.device(cfg.device)

    # --- Move to device & flatten projections
    y_AP = y_AP.flatten().to(device, dtype=torch.float32)
    y_PA = y_PA.flatten().to(device, dtype=torch.float32)
    CT_mu = CT_mu.to(device, dtype=torch.float32)
    organ_masks = [mk.to(device) for mk in organ_masks]
    if bg_mask is not None:
        bg_mask = bg_mask.to(device)

    D, H, W = CT_mu.shape
    N = D * H * W

    # --- Edge weights from CT (for edge-aware TV)
    w_edges = edge_weights_from_ct(CT_mu, spacing, alpha=cfg.alpha_edges)

    # --- Parameterization
    if cfg.mode == "organ_scalars_bg":
        # Variables: scalars a_k (>=0) for each organ + background field b (>=0)
        K = len(organ_masks)
        a = torch.full((K,), 0.1, device=device, requires_grad=True)
        b = torch.zeros((D, H, W), device=device, requires_grad=True)

        def build_x(a_vec: torch.Tensor, b_vol: torch.Tensor) -> torch.Tensor:
            x = torch.zeros((D, H, W), device=device)
            for i, mk in enumerate(organ_masks):
                x = x + a_vec[i].clamp_min(0.0) * mk.float()
            x = x + b_vol.clamp_min(0.0)  # enforce nonnegativity softly
            return x

        params = [a, b]

    elif cfg.mode == "voxelwise":
        # Variable: full voxel-wise x (>=0)
        x = init_from_organs(organ_masks, device)
        x.requires_grad_(True)
        params = [x]

        def build_x(x_vol: torch.Tensor) -> torch.Tensor:
            return x_vol.clamp_min(0.0)

    else:
        raise ValueError("cfg.mode must be 'organ_scalars_bg' or 'voxelwise'")

    # --- Optimizer
    opt = torch.optim.Adam(params, lr=cfg.lr)

    # --- Logging
    logs = {"loss": [], "data": [], "tv": [], "org": [], "bg": []}

    # --- Main loop
    for it in range(cfg.iters):
        # Compose current x
        if cfg.mode == "organ_scalars_bg":
            x_vol = build_x(a, b)
        else:
            x_vol = build_x(x)

        x_flat = flatten_volume(x_vol)

        # Forward projections
        yhat_AP = H_AP(CT_mu, x_flat)
        yhat_PA = H_PA(CT_mu, x_flat)

        # Data loss
        if cfg.use_poisson:
            L_data = poisson_nll(y_AP, yhat_AP) + poisson_nll(y_PA, yhat_PA)
        else:
            L_data = wls_loss(y_AP, yhat_AP) + wls_loss(y_PA, yhat_PA)

        # Regularizers
        R_tv = edge_aware_tv(x_vol, w_edges, spacing)
        R_org = intra_organ_variance(x_vol, organ_masks) if cfg.lambda_org > 0 else torch.tensor(0.0, device=device)
        R_bg = tv_on_mask(x_vol, bg_mask, spacing) if (bg_mask is not None and cfg.lambda_bg > 0) else torch.tensor(0.0, device=device)

        L = L_data + cfg.lambda_tv * R_tv + cfg.lambda_org * R_org + cfg.lambda_bg * R_bg

        # Backprop & step
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()

        # Optional hard clamp for stability (keeps x >= 0 strictly)
        if cfg.mode == "organ_scalars_bg":
            with torch.no_grad():
                a.clamp_(min=0.0)
                b.clamp_(min=0.0)
        else:
            with torch.no_grad():
                x.clamp_(min=0.0)

        # Logs
        logs["loss"].append(L.item())
        logs["data"].append(L_data.item())
        logs["tv"].append(R_tv.item())
        logs["org"].append(R_org.item())
        logs["bg"].append(R_bg.item())

        if (it + 1) % max(1, cfg.iters // 10) == 0:
            print(f"[Iter {it+1:4d}/{cfg.iters}] L={L.item():.3e} | data={L_data.item():.3e} | tv={R_tv.item():.3e} | org={R_org.item():.3e} | bg={R_bg.item():.3e}")

    # Return last estimate
    if cfg.mode == "organ_scalars_bg":
        x_est = build_x(a, b).detach()
    else:
        x_est = build_x(x).detach()

    return x_est, logs


# ------------------------------------------------------------
# Example wiring (replace with your pipeline)
# ------------------------------------------------------------

def main():
    # ==========================
    # TODO: Load your data here
    # ==========================
    # Shapes below are just for demonstration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example volume shape and spacing (mm)
    D, H, W = 128, 128, 128
    spacing = (2.0, 2.0, 2.0)  # (dz, dy, dx) in mm

    # Dummy CT_mu and masks (replace with real data)
    CT_mu = torch.zeros((D, H, W), dtype=torch.float32)

    # Create K=3 dummy organ masks (mutually exclusive, for demonstration)
    organ_masks = []
    mk1 = torch.zeros_like(CT_mu, dtype=torch.bool)
    mk1[20:60, 40:90, 30:80] = True
    mk2 = torch.zeros_like(CT_mu, dtype=torch.bool)
    mk2[70:100, 20:60, 20:60] = True
    mk3 = torch.zeros_like(CT_mu, dtype=torch.bool)
    mk3[50:90, 70:110, 70:110] = True
    organ_masks = [mk1, mk2, mk3]

    # Background is everything not in organs (optional for bg-TV)
    all_org = torch.zeros_like(CT_mu, dtype=torch.bool)
    for mk in organ_masks:
        all_org |= mk
    bg_mask = ~all_org

    # System matrices for AP & PA (M x N); here random for demo
    N = D * H * W
    M = 8192  # e.g., detector pixels (flattened)
    torch.manual_seed(0)
    A_AP = torch.rand((M, N), dtype=torch.float32) * 1e-4  # VERY sparse/weak in real life!
    A_PA = torch.rand((M, N), dtype=torch.float32) * 1e-4

    H_AP = DummyProjector(A_AP.to(device), name="H_AP")
    H_PA = DummyProjector(A_PA.to(device), name="H_PA")

    # Simulate a ground-truth x and generate synthetic data (demo only)
    x_gt = torch.zeros((D, H, W), dtype=torch.float32)
    x_gt[mk1] = 1.5
    x_gt[mk2] = 0.8
    x_gt[mk3] = 0.3
    x_gt = x_gt.to(device)

    y_AP = (H_AP(CT_mu.to(device), x_gt.flatten()) + 0.1).poisson().float()  # add Poisson noise
    y_PA = (H_PA(CT_mu.to(device), x_gt.flatten()) + 0.1).poisson().float()

    # Config
    cfg = ReconConfig(
        mode="voxelwise",       # or "organ_scalars_bg"
        lambda_tv=1e-3,
        lambda_org=1e-2,
        lambda_bg=1e-3,
        alpha_edges=10.0,
        lr=5e-3,
        iters=50,               # increase for real runs (e.g., 500-2000)
        use_poisson=True,
        device=str(device),
    )

    # Run optimization
    x_est, logs = run_optimization(
        y_AP=y_AP,
        y_PA=y_PA,
        H_AP=H_AP,
        H_PA=H_PA,
        CT_mu=CT_mu.to(device),
        organ_masks=[mk.to(device) for mk in organ_masks],
        bg_mask=bg_mask.to(device),
        spacing=spacing,
        cfg=cfg,
    )

    print("Done. x_est shape:", tuple(x_est.shape))

    # Example: compute simple RMSE against x_gt (demo only)
    rmse = torch.sqrt(((x_est - x_gt) ** 2).mean()).item()
    print(f"RMSE (demo): {rmse:.4f}")


if __name__ == "__main__":
    main()
