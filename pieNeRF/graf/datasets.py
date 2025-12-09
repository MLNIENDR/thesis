"""Dataset utilities for SPECT data: load AP/PA projections, CT/act volumes from manifest CSV."""

import glob
import numpy as np
from PIL import Image
from pathlib import Path
import csv
import torch

from torchvision.datasets.vision import VisionDataset



class SpectDataset(torch.utils.data.Dataset):
    """
    Dataset für SPECT-ähnliche Rekonstruktion:
    Lädt pro Fall AP, PA und CT, opt. ACT (alles als .npy), basierend auf einem Manifest (CSV).
    """
    def __init__(
        self,
        manifest_path,
        imsize=None,
        transform_img=None,
        transform_ct=None,
        act_scale: float = 1.0,
        use_organ_mask: bool = False,
    ):  
        super().__init__()
        self.manifest_path = Path(manifest_path)                                # manifest_path = Pfad zur csv mit Spalten phantom_id, ap_path, pa_path, ct_path
        self.imsize = imsize                                                    # momentan nicht genutzt
        self.transform_img = transform_img                                      # optionale Transformationsfunktionen für AP/PA und CT
        self.transform_ct = transform_ct                                        # "
        self.act_scale = float(act_scale)                                       # globaler Faktor für ACT/λ (keine Normierung)
        self.use_organ_mask = bool(use_organ_mask)

        self.entries = []                                                       # Liste in der für jeden Fall ein kleines Dict mit Pfaden & ID steht
        with open(self.manifest_path, newline="") as f:                         # CSV öffnen
            reader = csv.DictReader(f)                                          # liest jede Zeile als dict
            for row in reader:
                act_path = row.get("act_path")
                if act_path is None:
                    # optional automatisch aus dem Ordner ableiten
                    candidate = Path(row["ap_path"]).with_name("act.npy")
                    act_path = candidate if candidate.exists() else None
                else:
                    act_path = Path(act_path)

                self.entries.append({                                           # jeweils als Path-Objekte speichern: phantom_id, ap_path, pa_path, ct_path
                    "patient_id": row["patient_id"],
                    "ap_path": Path(row["ap_path"]),
                    "pa_path": Path(row["pa_path"]),
                    "ct_path": Path(row["ct_path"]),
                    "act_path": act_path,
                    "mask_path": Path(row["ct_path"]).with_name("mask.npy"),
                    "ct_att_path": Path(row["ct_path"]).with_name("ct_att.npy"),
                    "spect_att_path": Path(row["ct_path"]).with_name("spect_att.npy"),
                })

    def __len__(self):
        return len(self.entries)                                                # Anzahl der Einträge = Anzahl Zeilen im Manifest = Anzahl Phantome/Patienten


    def _load_npy_image(self, path):                                            # lädt .npy Array, z.B. [H,W] mit Counts
        arr = np.load(path).astype(np.float32)                                  # stellt sicher, dass float32
        
        tensor = torch.from_numpy(arr).unsqueeze(0)                             # macht einen Tensor draus [H,W] -> [1,H,W]

        return tensor


    def _load_npy_ct(self, path):
        if path is None:
            return torch.empty(0)
        path = Path(path)
        if not path.exists():
            return torch.empty(0)

        vol = np.load(path).astype(np.float32)                                  # Original gespeichert als (LR, AP, SI)
        vol = np.transpose(vol, (1, 0, 2))                                      # neu: (AP, LR, SI) -> depth=AP, H=LR, W=SI

        vol *= 10.0                                                             # optionaler scale-factor

        return torch.from_numpy(vol)


    def _load_npy_act(self, path):
        if path is None:
            return torch.empty(0)

        vol = np.load(path).astype(np.float32)
        vol = np.transpose(vol, (1, 0, 2))                                      # (AP, LR, SI)

        # Kein Min/Max-Scaling mehr – optional nur globaler Faktor, damit λ im festen Maßstab bleibt.
        vol = vol * self.act_scale

        return torch.from_numpy(vol)

    def _load_npy_att(self, path):
        """Lädt zusätzliche Attenuations-Volumes (ct_att, spect_att) mit identischer Permutation/Scale wie CT."""
        if path is None:
            return torch.empty(0)
        path = Path(path)
        if not path.exists():
            return torch.empty(0)
        vol = np.load(path).astype(np.float32)
        vol = np.transpose(vol, (1, 0, 2))                                      # (AP, LR, SI)
        vol *= 10.0
        return torch.from_numpy(vol)

    def _load_npy_mask(self, path):
        if path is None or not self.use_organ_mask:
            return torch.empty(0)
        path = Path(path)
        if not path.exists():
            return torch.empty(0)
        vol = np.load(path).astype(np.float32)
        vol = np.transpose(vol, (1, 0, 2))                                      # (AP, LR, SI)
        vol = np.clip(vol, 0.0, 1.0)
        return torch.from_numpy(vol)


    def __getitem__(self, idx):
        e = self.entries[idx]                                                   # holt das idx-te Manifest-Dict
        ap = self._load_npy_image(e["ap_path"])                                 # lädt AP als [1,H,W]-Tensor (roh, keine Min/Max-Norm)
        pa = self._load_npy_image(e["pa_path"])                                 # lädt PA als [1,H,W]-Tensor (roh, keine Min/Max-Norm)
        ct = self._load_npy_ct(e["ct_path"])                                    # lädt CT Volumen (gescaled)  
        act = self._load_npy_act(e["act_path"])                                 # lädt ACT Volumen (ohne Normierung, nur globaler Faktor)
        mask = self._load_npy_mask(e.get("mask_path"))
        ct_att = self._load_npy_att(e.get("ct_att_path"))
        spect_att = self._load_npy_att(e.get("spect_att_path"))
        if ct.numel() == 0 and ct_att.numel() > 0:
            ct = ct_att                                                          # nutze ct_att als CT-Basis, falls ct.npy fehlt

        ap = self._normalize_projection(ap)
        pa = self._normalize_projection(pa)

        if self.transform_img is not None:                                      # optionale zusätzliche Schritte (z.B. Resize, Cropping, ...)
            ap = self.transform_img(ap)
            pa = self.transform_img(pa)
        if self.transform_ct is not None:
            ct = self.transform_ct(ct)

        # Debug-Ausgabe nur beim ersten Item, um Skalen zu prüfen (kein Einfluss auf Verhalten).
        if idx == 0:
            ct_min = ct.min().item() if ct.numel() > 0 else float("nan")
            ct_max = ct.max().item() if ct.numel() > 0 else float("nan")
            print(
                f"[DEBUG][datasets] AP min/max: {ap.min().item():.3e}/{ap.max().item():.3e} | "
                f"PA min/max: {pa.min().item():.3e}/{pa.max().item():.3e} | "
                f"CT min/max: {ct_min:.3e}/{ct_max:.3e} | "
                f"ACT min/max: {(act.min().item() if act.numel()>0 else float('nan')):.3e}/"
                f"{(act.max().item() if act.numel()>0 else float('nan')):.3e} | "
                f"MASK min/max: {(mask.min().item() if mask.numel()>0 else float('nan')):.3e}/"
                f"{(mask.max().item() if mask.numel()>0 else float('nan')):.3e} | "
                f"CT_att min/max: {(ct_att.min().item() if ct_att.numel()>0 else float('nan')):.3e}/"
                f"{(ct_att.max().item() if ct_att.numel()>0 else float('nan')):.3e} | "
                f"SPECT_att min/max: {(spect_att.min().item() if spect_att.numel()>0 else float('nan')):.3e}/"
                f"{(spect_att.max().item() if spect_att.numel()>0 else float('nan')):.3e}",
                flush=True,
            )

        return {                                                                # Rückgabe ist ein Dictionary   
            "ap": ap,
            "pa": pa,
            "ct": ct,
            "act": act,
            "ct_att": ct_att,
            "spect_att": spect_att,
            "mask": mask,
            "meta": {
                "patient_id": e["patient_id"],
                "act_scale": self.act_scale,
            },
        }

    def _normalize_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Einfache per-Projektion-Normierung auf [0,1], wie im ursprünglichen Code."""
        maxv = tensor.max()
        if maxv > 0:
            tensor = tensor / maxv
        return tensor
