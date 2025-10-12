"""
AP/PA Activity Reconstruction with CT-based Attenuation and Organ-Aware Regularization
=====================================================================================


Überblick
-------------
- Daten-Terme: Poisson-NLL (Standard) oder WLS, je View (AP & PA).  
- Regularisierung: (1) CT-kantenbewusste TV, (2) intraorganale Varianz, (3) optionale TV im Hintergrund.  
- Nebenbedingung: Nichtnegativität von x (soft via clamp bzw. Hard-Clamp nach jedem Schritt).  
- Projektor: Abstraktionsklasse `BaseProjector`, Demo-Implementierung `DummyProjector(A)` (Matrix-Forwardmodell).  
- Modi: `voxelwise` (vollvoxelige Aktivität) oder `organ_scalars_bg` (ein Skalar pro Organ + glattes Hintergrundfeld).  

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
    """Abstrakte Projektor-Schnittstelle: y_hat = H(CT_mu, x)
    ----------------------------------------------------------------
    - Diese Klasse definiert die API, die ein Forward-Projektor erfüllen muss.
    - `__call__` nimmt CT-Mu (zur Modellierung der Dämpfung) und den Aktivitätsvektor x.
    - `M`/`N` geben Mess- bzw. Voxelanzahl an und ermöglichen einfache Checks/Wiring.

    `DummyProjector` muss durch reales AP/PA-Forwardmodell, das die
    Strahlwege, Dämpfung (Beer–Lambert) etc. abbildet entspr. ersetzt werden.
    """
    def __init__(self, name: str = "H"):
        self.name = name

    def __call__(self, CT_mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Führt die Vorwärtsprojektion aus.
        Muss in Subklassen überschrieben werden.
        - CT_mu: 3D-Volumen der linearen Schwächungskoeffizienten
        - x:    Aktivität als flacher Vektor der Länge N
        - return: projizierte Daten y_hat (flach, Länge M)
        """
        raise NotImplementedError

    @property
    def M(self) -> int:
        """Anzahl der Messwerte (Detektorpixel * ggf. Views)."""
        raise NotImplementedError

    @property
    def N(self) -> int:
        """Anzahl der Voxel im Rekonstruktionsvolumen."""
        raise NotImplementedError


class DummyProjector(BaseProjector):                                                    # definiert Unterklasse von BaseProjector (erbt die implementierten Schnittstellen)
    """Einfache matrixbasierte Projektor-Demo: y = A @ x
    ----------------------------------------------------------------
    - `A` ist eine (dichte oder sparse) Torch-Matrix der Form [M, N].
    - Dämpfung ist hier bereits in A enthalten (oder wird vernachlässigt).
    - Praktisch für Prototyping, Unit-Tests oder als Platzhalter.
    """
    def __init__(self, A: torch.Tensor, name: str = "H"):
        super().__init__(name)                                                          # ruft Konstruktor der Basisklasse BaseProjector auf, um Namensattribut zu setzen                
        assert A.ndim == 2, "A must be [M, N]"                                          # prüft, ob A zweidimensional ist        
        self.A = A                                                                      # speichert die Matrix im Objekt        

    def __call__(self, CT_mu: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # In dieser Dummy-Variante wird CT_mu nicht verwendet, da die Dämpfung
        # bereits in A eingebunden ist.
        return self.A @ x                                                               # Matrix-Vektor-Multiplikation --> ergibt Tensor der Länge M (also simulierte Projektionen)            

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
    """Poisson-Negativloglikelihood (bis auf konstante Terme log(y!))
    L = sum( yhat - y * log(yhat + eps) )

    - Stabilisiert mit kleinem `eps`, um log(0) zu vermeiden.
    - Passend für Zählstatistik (SPECT/PET-ähnliche Projekte).
    """
    return (yhat - y * torch.log(yhat + eps)).sum()                                                                 # y:        gemessene Counts
                                                                                                                    # yhat:     modellierte / erwartete Counts vom Forwardmodell                                    
                                                                                                                    # eps:      kleiner Stabilisierungsterm, damit nie log(0)            

def wls_loss(y: torch.Tensor, yhat: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Gewichtete Least-Squares-Kosten: sum( w * (yhat - y)^2 )
    - Falls `weight` None ist, wird 1/max(y,1) als Varianz-Proxy genutzt (heteroskedastisch).
    - Praktisch, wenn man lieber einen quadratischen Term möchte.
    """
    r = (yhat - y)
    if weight is None:
        weight = 1.0 / torch.clamp(y, min=1.0)
    return (weight * r * r).sum()


# ------------------------------------------------------------
# Regularizers: edge-aware TV and intra-organ variance
# ------------------------------------------------------------

def _gradient_3d(x: torch.Tensor, spacing: Tuple[float, float, float]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vorwärtsdifferenzen (3D) mit physikalischer Skalierung.
    x: [D,H,W]
    spacing: (dz, dy, dx) in mm -> Rückgabe (gx, gy, gz) in 1/mm

    Randbehandlung: Neumann-ähnlich (letzte Zeile/Spalte/Ebene wird mit 0-Gradient behandelt).
    """
    dz, dy, dx = spacing
    # Pad jeweils am Ende, sodass die Differenz am Rand 0 wird (Neumann Randbedingungen)
    gx = (F.pad(x[1:, :, :], (0,0,0,0,0,1)) - x) / dz
    gy = (F.pad(x[:, 1:, :], (0,0,0,0,0,0,0,1)) - x) / dy
    gz = (F.pad(x[:, :, 1:], (0,1,0,0,0,0)) - x) / dx
    return gx, gy, gz


def edge_weights_from_ct(ct: torch.Tensor, spacing: Tuple[float, float, float], alpha: float = 10.0) -> torch.Tensor:
    """Erzeuge kantenbewusste Gewichte aus CT-Kanten: w = exp(-alpha * |∇CT|)
    - Große CT-Gradienten -> kleine Gewichte -> weniger Glättung über Organränder hinweg.
    - `alpha` steuert, wie stark Kanten geschützt werden.
    """
    gx, gy, gz = _gradient_3d(ct, spacing)
    grad_mag = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    w = torch.exp(-alpha * grad_mag)
    return w


def edge_aware_tv(x: torch.Tensor, w: torch.Tensor, spacing: Tuple[float, float, float]) -> torch.Tensor:
    """Isotrope, kantenbewusste Total Variation: sum( w * ||∇x||_2 )
    - `w` stammt typischerweise aus CT-Kanten (siehe oben).
    - Senkt die Glättung entlang starker CT-Kanten, erhält Grenzen.
    """
    gx, gy, gz = _gradient_3d(x, spacing)
    tv = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    return (w * tv).sum()


def intra_organ_variance(x: torch.Tensor, masks: List[torch.Tensor]) -> torch.Tensor:
    """Summe der quadratischen Abweichungen von der Organmittelung (Uniformitäts-Prior)
    - Für jedes Organ k: sum_{i in Organ k} (x_i - mean_k)^2
    - Fördert gleichmäßige Aktivität innerhalb eines Organs, ohne Werte festzuklemmen.
    """
    loss = 0.0
    for mk in masks:
        mkf = mk.float()
        nk = mkf.sum().clamp(min=1.0)  # Schutz vor Division durch 0
        mean_k = (x * mkf).sum() / nk
        loss += ((x - mean_k) ** 2 * mkf).sum()
    return loss


def tv_on_mask(x: torch.Tensor, mask: torch.Tensor, spacing: Tuple[float, float, float]) -> torch.Tensor:
    """Einfache isotrope TV, nur auf Voxeln eines gegebenen Masks (z.B. Hintergrund)."""
    gx, gy, gz = _gradient_3d(x, spacing)
    tv = torch.sqrt(gx * gx + gy * gy + gz * gz + 1e-12)
    return (tv * mask.float()).sum()


# ------------------------------------------------------------
# Parameterizations
# ------------------------------------------------------------

def init_from_organs(masks: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    """Einfache Initialisierung: jedes Organ auf kleine positive Konstante, Hintergrund 0.
    - Liefert x0 in Form [D,H,W]. Dient als Startwert für den voxelweisen Modus.
    """
    assert len(masks) > 0
    shape = masks[0].shape
    x0 = torch.zeros(shape, dtype=torch.float32, device=device)
    for mk in masks:
        x0 = x0 + 0.1 * mk.float()  # 0.1 ist willkürlich; ggf. projektspezifisch anpassen
    return x0


def flatten_volume(x: torch.Tensor) -> torch.Tensor:
    """Forme [D,H,W] nach [N]"""
    return x.reshape(-1)


def unflatten_volume(x_flat: torch.Tensor, shape: Tuple[int, int, int]) -> torch.Tensor:
    """Forme [N] zurück nach [D,H,W]"""
    return x_flat.reshape(shape)


# ------------------------------------------------------------
# Config and main optimizer routine
# ------------------------------------------------------------

@dataclass
class ReconConfig:
    """Konfigurationscontainer für die Rekonstruktion/Optimierung."""
    mode: str = "voxelwise"  # "voxelwise" oder "organ_scalars_bg"
    lambda_tv: float = 1e-3   # Gewicht der kantenbewussten TV
    lambda_org: float = 1e-2  # Gewicht der Intra-Organ-Varianz
    lambda_bg: float = 1e-3   # Gewicht der TV im Hintergrund (optional)
    alpha_edges: float = 10.0 # Schärfeparameter für w = exp(-alpha*|∇CT|)
    lr: float = 1e-2          # Lernrate für Adam
    iters: int = 500          # Anzahl der Iterationen
    use_poisson: bool = True  # True: Poisson-NLL, False: WLS
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
    """Führt die Rekonstruktion/Schätzung aus.

    Eingaben
    --------
    y_AP, y_PA : gemessene Projektionen, beliebige Form (werden geflattet)
    H_AP, H_PA : Projektor-Instanzen (kompatible M,N)
    CT_mu      : 3D-Volumen [D,H,W] der Schwächungskoeffizienten
    organ_masks: Liste boolscher Masken [D,H,W], eine pro Organ
    bg_mask    : boolsche Maske [D,H,W] für Hintergrund (optional)
    spacing    : (dz, dy, dx) in mm
    cfg        : Hyperparameter/Moduswahl

    Rückgabe
    --------
    x_est      : geschätztes Aktivitätsvolumen [D,H,W]
    logs       : einfache Verlaufs-Logs (Skalare je Iteration)
    """
    device = torch.device(cfg.device)

    # --- Move to device & flach machen
    y_AP = y_AP.flatten().to(device, dtype=torch.float32)
    y_PA = y_PA.flatten().to(device, dtype=torch.float32)
    CT_mu = CT_mu.to(device, dtype=torch.float32)
    organ_masks = [mk.to(device) for mk in organ_masks]
    if bg_mask is not None:
        bg_mask = bg_mask.to(device)

    D, H, W = CT_mu.shape
    N = D * H * W  # nur informativ; Shape-Checks können helfen

    # --- Kantengewichte aus CT (für edge-aware TV)
    w_edges = edge_weights_from_ct(CT_mu, spacing, alpha=cfg.alpha_edges)

    # --- Parameterisierung je nach Modus
    if cfg.mode == "organ_scalars_bg":
        # Variablen: a_k (>=0) je Organ + Hintergrundfeld b (>=0)
        K = len(organ_masks)
        a = torch.full((K,), 0.1, device=device, requires_grad=True)  # Startwerte klein positiv
        b = torch.zeros((D, H, W), device=device, requires_grad=True) # Hintergrundfeld

        def build_x(a_vec: torch.Tensor, b_vol: torch.Tensor) -> torch.Tensor:
            # Baut das Volumen x aus Organ-Skalaren und Hintergrund zusammen (soft clamp >= 0)
            x = torch.zeros((D, H, W), device=device)
            for i, mk in enumerate(organ_masks):
                x = x + a_vec[i].clamp_min(0.0) * mk.float()
            x = x + b_vol.clamp_min(0.0)
            return x

        params = [a, b]

    elif cfg.mode == "voxelwise":
        # Variable: vollständiges Voxel-Volumen x (>=0)
        x = init_from_organs(organ_masks, device)
        x.requires_grad_(True)
        params = [x]

        def build_x(x_vol: torch.Tensor) -> torch.Tensor:
            # Soft-Enforcement der Nichtnegativität
            return x_vol.clamp_min(0.0)

    else:
        raise ValueError("cfg.mode must be 'organ_scalars_bg' or 'voxelwise'")

    # --- Optimierer (Adam ist robust gegen Skalierungsunterschiede)
    opt = torch.optim.Adam(params, lr=cfg.lr)

    # --- Logging-Container
    logs = {"loss": [], "data": [], "tv": [], "org": [], "bg": []}

    # --- Hauptschleife
    for it in range(cfg.iters):
        # Aktuelles x zusammenbauen (abhängig von der Parametrisierung)
        if cfg.mode == "organ_scalars_bg":
            x_vol = build_x(a, b)
        else:
            x_vol = build_x(x)

        x_flat = flatten_volume(x_vol)

        # Vorwärtsprojektionen für AP & PA
        yhat_AP = H_AP(CT_mu, x_flat)
        yhat_PA = H_PA(CT_mu, x_flat)

        # Daten-Fidelität
        if cfg.use_poisson:
            L_data = poisson_nll(y_AP, yhat_AP) + poisson_nll(y_PA, yhat_PA)
        else:
            L_data = wls_loss(y_AP, yhat_AP) + wls_loss(y_PA, yhat_PA)

        # Regularisierungsterme
        R_tv = edge_aware_tv(x_vol, w_edges, spacing)
        R_org = intra_organ_variance(x_vol, organ_masks) if cfg.lambda_org > 0 else torch.tensor(0.0, device=device)
        R_bg = tv_on_mask(x_vol, bg_mask, spacing) if (bg_mask is not None and cfg.lambda_bg > 0) else torch.tensor(0.0, device=device)

        # Gesamtverlust
        L = L_data + cfg.lambda_tv * R_tv + cfg.lambda_org * R_org + cfg.lambda_bg * R_bg

        # Backpropagation & Optimierungsschritt
        opt.zero_grad(set_to_none=True)
        L.backward()
        opt.step()

        # Optionale harte Nichtnegativitätsklemme (Numerik-Stabilität)
        if cfg.mode == "organ_scalars_bg":
            with torch.no_grad():
                a.clamp_(min=0.0)
                b.clamp_(min=0.0)
        else:
            with torch.no_grad():
                x.clamp_(min=0.0)

        # Logs fortschreiben
        logs["loss"].append(L.item())
        logs["data"].append(L_data.item())
        logs["tv"].append(R_tv.item())
        logs["org"].append(R_org.item())
        logs["bg"].append(R_bg.item())

        # Gelegentliche Konsolen-Ausgabe (10x über den Lauf verteilt)
        if (it + 1) % max(1, cfg.iters // 10) == 0:
            print(f"[Iter {it+1:4d}/{cfg.iters}] L={L.item():.3e} | data={L_data.item():.3e} | tv={R_tv.item():.3e} | org={R_org.item():.3e} | bg={R_bg.item():.3e}")

    # Letzte Schätzung zurückgeben (detach, um vom Gradienten-Graphen zu lösen)
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
    # TODO: Daten hier laden
    # ==========================
    # Die folgenden Shapes/Werte sind rein demonstrativ.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Beispiel-Volumengröße und Voxelabstände (mm)
    D, H, W = 128, 128, 128
    spacing = (2.0, 2.0, 2.0)  # (dz, dy, dx) in mm

    # Dummy-CT_mu und Organmasken (ersetzen durch reale Daten)
    CT_mu = torch.zeros((D, H, W), dtype=torch.float32)

    # K=3 Beispiel-Organe (disjunkt, rein demonstrativ)
    organ_masks = []
    mk1 = torch.zeros_like(CT_mu, dtype=torch.bool)
    mk1[20:60, 40:90, 30:80] = True
    mk2 = torch.zeros_like(CT_mu, dtype=torch.bool)
    mk2[70:100, 20:60, 20:60] = True
    mk3 = torch.zeros_like(CT_mu, dtype=torch.bool)
    mk3[50:90, 70:110, 70:110] = True
    organ_masks = [mk1, mk2, mk3]

    # Hintergrundmaske: alles, was nicht Organ ist (optional für TV im Hintergrund)
    all_org = torch.zeros_like(CT_mu, dtype=torch.bool)
    for mk in organ_masks:
        all_org |= mk
    bg_mask = ~all_org

    # Systemmatrizen für AP & PA (M x N); hier rein zufällig für Demo
    N = D * H * W
    M = 8192  # z.B. Anzahl Detektorpixel (geflattet)
    torch.manual_seed(0)
    A_AP = torch.rand((M, N), dtype=torch.float32) * 1e-4  # In Realität stark dünn besetzt/gewichtet
    A_PA = torch.rand((M, N), dtype=torch.float32) * 1e-4

    H_AP = DummyProjector(A_AP.to(device), name="H_AP")
    H_PA = DummyProjector(A_PA.to(device), name="H_PA")

    # Simulierter Ground Truth & synthetische Daten (nur Demo!)
    x_gt = torch.zeros((D, H, W), dtype=torch.float32)
    x_gt[mk1] = 1.5
    x_gt[mk2] = 0.8
    x_gt[mk3] = 0.3
    x_gt = x_gt.to(device)

    # Poisson-Rauschen hinzufügen (kleiner Offset, um 0-Counts zu vermeiden)
    y_AP = (H_AP(CT_mu.to(device), x_gt.flatten()) + 0.1).poisson().float()
    y_PA = (H_PA(CT_mu.to(device), x_gt.flatten()) + 0.1).poisson().float()

    # Konfiguration
    cfg = ReconConfig(
        mode="voxelwise",       # oder "organ_scalars_bg"
        lambda_tv=1e-3,
        lambda_org=1e-2,
        lambda_bg=1e-3,
        alpha_edges=10.0,
        lr=5e-3,
        iters=50,               # Für reale Läufe erhöhen (z.B. 500-2000)
        use_poisson=True,
        device=str(device),
    )

    # Optimierung starten
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

    # Beispielhafte Auswertung: RMSE ggü. x_gt (nur Demo)
    rmse = torch.sqrt(((x_est - x_gt) ** 2).mean()).item()
    print(f"RMSE (demo): {rmse:.4f}")


if __name__ == "__main__":
    main()
