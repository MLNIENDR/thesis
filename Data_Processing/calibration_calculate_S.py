#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibration_calculate_S.py

Kurzueberblick:
Erzeugt ein einfaches Aktivitaets-Phantom mit bekannter Gesamtaktivitaet,
projiziert es mit dem Forward-Projektor und berechnet daraus S_eff (cps/MBq).
S_eff ist die Umrechnung von MBq zu erwarteten Counts pro Sekunde fuer das
aktuelle Projektor-Setup (Kernel/Scatter/Attenuation/Geometrie).


python3 calibration_calculate_S.py \
  --kernel_mat LEAP_Kernel.mat \
  --kernel_var kernel_mat \
  --shape 256,256,651 \
  --sd_mm 1.5 \
  --A_total_MBq 100 \
  --spect_att_npy /home/mnguest12/projects/thesis/Data_Processing/phantom_01/out/spect_att.npy


  
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import scipy.io as sio
except Exception:
    sio = None

from preprocessing import load_bin_xyz, convert_mu_units, gamma_camera_core

Z0_SLICES_DEFAULT = 29


def parse_args():
    p = argparse.ArgumentParser(
        description="Berechne S_eff (cps/MBq) fuer den aktuellen Forward-Projektor."
    )
    p.add_argument("--shape", type=str, default="256,256,651",
                   help="Volumen-Shape als 'x,y,z' (default: 256,256,651)")
    p.add_argument("--sd_mm", type=float, default=1.5,
                   help="Voxelspacing in mm (isotrop angenommen)")
    p.add_argument("--mu_bin", type=Path, default=None,
                   help="Optionales µ-Volumen (BIN), gleiche Form wie shape")
    p.add_argument("--mu_unit", type=str, choices=["per_mm", "per_cm"], default="per_mm",
                   help="Einheit des µ-Volumens (1/mm oder 1/cm)")
    p.add_argument("--mu_target_unit", type=str, choices=["per_mm", "per_cm"], default="per_cm",
                   help="Ziel-Einheit fuer µ vor der Projektion")
    p.add_argument("--spect_att_npy", type=Path, default=None,
                   help="Optional: Pfad zu spect_att.npy (mu in 1/cm). "
                        "Wenn gesetzt, wird die Phantom-Maske aus mu>0 erstellt.")
    p.add_argument("--kernel_mat", type=Path, required=True,
                   help="MATLAB-Datei mit LEAP-Kernel (enthaelt 3D-Array)")
    p.add_argument("--kernel_var", type=str, default="kernel_mat",
                   help="Variablenname in der .mat-Datei (z.B. 'kernel_mat')")
    p.add_argument("--psf_sigma", type=float, default=2.0,
                   help="Sigma fuer den Scatter-Gauss (Pixel)")
    p.add_argument("--A_total_MBq", type=float, default=100.0,
                   help="Gesamtaktivitaet im Phantom (MBq)")
    p.add_argument("--phantom_type", type=str, choices=["uniform_cylinder", "uniform_box"],
                   default="uniform_cylinder",
                   help="Phantom-Typ")
    p.add_argument("--radius_mm", type=float, default=100.0,
                   help="Zylinder-Radius in mm (nur uniform_cylinder)")
    p.add_argument("--height_mm", type=float, default=200.0,
                   help="Zylinder-Hoehe in mm (nur uniform_cylinder)")
    p.add_argument("--box_x_mm", type=float, default=200.0,
                   help="Box-Groesse in mm (x) fuer uniform_box")
    p.add_argument("--box_y_mm", type=float, default=200.0,
                   help="Box-Groesse in mm (y) fuer uniform_box")
    p.add_argument("--box_z_mm", type=float, default=200.0,
                   help="Box-Groesse in mm (z) fuer uniform_box")
    p.add_argument("--view", type=str, choices=["AP", "PA", "both"], default="both",
                   help="Welche Projektion ausgewertet wird")
    p.add_argument("--out_dir", type=Path, default=None,
                   help="Optionaler Output-Ordner fuer ap_raw/pa_raw")
    return p.parse_args()


def _parse_shape(shape_str: str) -> Tuple[int, int, int]:
    """Parst 'x,y,z' in ein Shape-Tuple (nx, ny, nz)."""
    x, y, z = [int(s) for s in shape_str.split(",")]
    return x, y, z


def _centered_coords(n: int) -> np.ndarray:
    """Koordinatenachse zentriert um 0 (Voxelindex -> physikalische Lage in Voxeln)."""
    return np.arange(n, dtype=np.float32) - (n - 1) / 2.0


def build_phantom_A_xyz_Bq(shape: Tuple[int, int, int],
                           sd_mm: float,
                           phantom_type: str,
                           radius_mm: float,
                           height_mm: float,
                           box_x_mm: float,
                           box_y_mm: float,
                           box_z_mm: float,
                           A_total_MBq: float) -> np.ndarray:
    """Baut ein einfaches, homogenes Aktivitaetsphantom in Bq/voxel.

    Inputs:
      shape: (nx, ny, nz) in Voxeln.
      sd_mm: Voxelgroesse in mm (isotrop angenommen).
      phantom_type: 'uniform_cylinder' oder 'uniform_box'.
      A_total_MBq: Gesamtaktivitaet im Phantom (MBq).
    Output:
      act: 3D-Volumen in Bq/voxel mit homogener Aktivitaet im Phantom.
    """
    nx, ny, nz = shape
    act = np.zeros(shape, dtype=np.float32)

    # Maske fuer ein homogenes Kalibrierphantom
    if phantom_type == "uniform_cylinder":
        x = _centered_coords(nx)
        y = _centered_coords(ny)
        z = _centered_coords(nz)
        r_vox = radius_mm / sd_mm                   # Radius und Höhe des Zylinders in Voxeln
        h_vox = height_mm / sd_mm
        r2 = x[:, None] ** 2 + y[None, :] ** 2      # erzeugt 2D-Raster der radialen Abstände in der xy-Ebene
        mask_xy = r2 <= (r_vox ** 2)                # markiert alle Punkte innerhalb des Zylinderradius in xy-Ebene
        mask_z = np.abs(z) <= (h_vox / 2.0)         # markiert alle z-Slices innerhalb der halben Höhe
        mask = mask_xy[:, :, None] & mask_z[None, None, :]  # Kombination zu 3D-Maske --> Zylinder im Volumen
    else:
        x = _centered_coords(nx)
        y = _centered_coords(ny)
        z = _centered_coords(nz)
        dx = box_x_mm / sd_mm
        dy = box_y_mm / sd_mm
        dz = box_z_mm / sd_mm
        mask_x = np.abs(x) <= (dx / 2.0)
        mask_y = np.abs(y) <= (dy / 2.0)
        mask_z = np.abs(z) <= (dz / 2.0)
        mask = mask_x[:, None, None] & mask_y[None, :, None] & mask_z[None, None, :]

    nvox = int(mask.sum())
    if nvox <= 0:
        raise ValueError("Phantom-Maske hat 0 Voxel. Bitte Parameter pruefen.")

    # Gleichmaessige Verteilung der Gesamtaktivitaet auf alle Phantom-Voxel
    total_Bq = float(A_total_MBq) * 1e6             # Gesamtaktivität hier 1 GBq
    bq_per_vox = total_Bq / float(nvox)             # Aktivität pro Voxel
    act[mask] = bq_per_vox                          # setzt Wert für jeden Voxel, außerhalb = 0
    return act                                      # gibt Aktivitäts-3D-Arrays zurück


def main():
    # CLI / Parameter
    args = parse_args()
    shape = _parse_shape(args.shape)

    # Schrittweite entlang z passend zur µ-Einheit
    if args.mu_target_unit == "per_cm":
        step_len = args.sd_mm / 10.0
    else:
        step_len = args.sd_mm

    # LEAP-Kernel laden
    if sio is None:
        raise RuntimeError("scipy.io (sio) wird fuer das Laden des LEAP-Kernels benoetigt.")
    kernel_md = sio.loadmat(args.kernel_mat)
    if args.kernel_var not in kernel_md:
        raise KeyError(f"Variable '{args.kernel_var}' nicht in {args.kernel_mat} gefunden.")
    kernel_mat = kernel_md[args.kernel_var].astype(np.float32)

    # Optionales µ-Volumen fuer Attenuation (sonst 0 -> keine Dämpfung)
    if args.spect_att_npy is not None:
        mu_xyz = np.load(args.spect_att_npy).astype(np.float32)
        if mu_xyz.shape != shape:
            raise ValueError(f"spect_att_npy shape {mu_xyz.shape} passt nicht zu shape {shape}.")
    elif args.mu_bin is not None:
        mu_xyz = load_bin_xyz(args.mu_bin, args.shape, dtype="float32", order="F")
        mu_xyz = convert_mu_units(mu_xyz, args.mu_unit, args.mu_target_unit)
    else:
        mu_xyz = np.zeros(shape, dtype=np.float32)

    # Aktivitaetsphantom in Bq/voxel
    if args.spect_att_npy is not None:
        # Maske direkt aus mu>0 (entspricht Phantom-Koerper)
        mask = mu_xyz > 0.0
        nvox = int(mask.sum())
        if nvox <= 0:
            raise ValueError("spect_att_npy Maske hat 0 Voxel (mu>0).")
        total_Bq = float(args.A_total_MBq) * 1e6
        bq_per_vox = total_Bq / float(nvox)
        A_xyz_Bq = np.zeros(shape, dtype=np.float32)
        A_xyz_Bq[mask] = bq_per_vox
    else:
        A_xyz_Bq = build_phantom_A_xyz_Bq(
            shape=shape,
            sd_mm=args.sd_mm,
            phantom_type=args.phantom_type,
            radius_mm=args.radius_mm,
            height_mm=args.height_mm,
            box_x_mm=args.box_x_mm,
            box_y_mm=args.box_y_mm,
            box_z_mm=args.box_z_mm,
            A_total_MBq=args.A_total_MBq,
        )

    # Forward-Projektion (AP/PA) im gleichen Modell wie preprocessing.py
    ap_raw, pa_raw = gamma_camera_core(
        act_data=A_xyz_Bq.astype(np.float32),
        # Achtung: Hier wird entschieden, ob die Kalibrierung Attenuation enthaelt.
        # Wenn mu_bin gesetzt ist, entspricht der Skalierungsfaktor dem *modellspezifischen*
        # Verhalten inkl. Attenuation des Kalibrier-Phantoms.
        atn_data=mu_xyz.astype(np.float32),
        kernel_mat=kernel_mat,
        sigma=args.psf_sigma,
        z0_slices=Z0_SLICES_DEFAULT,
        step_len=step_len,
        comp_scatter=True,
        atn_on=(args.mu_bin is not None or args.spect_att_npy is not None),
        coll_on=True,
    )

    # S_eff = cps/MBq: Summe der Rohprojektion / Gesamtaktivitaet
    total_MBq_check = float(A_xyz_Bq.sum()) / 1e6
    print(f"A_total_MBq (target): {float(args.A_total_MBq):.6f}")
    print(f"A_total_MBq (sum check): {total_MBq_check:.6f}")
    print(f"sum(ap_raw) = {float(ap_raw.sum()):.6e}")
    print(f"sum(pa_raw) = {float(pa_raw.sum()):.6e}")

    if args.view in ("AP", "both"):
        cps_raw_ap = float(ap_raw.sum())
        s_eff_ap = cps_raw_ap / float(args.A_total_MBq)
        print(f"S_eff_AP (cps/MBq) = {s_eff_ap:.6e}")
    if args.view in ("PA", "both"):
        cps_raw_pa = float(pa_raw.sum())
        s_eff_pa = cps_raw_pa / float(args.A_total_MBq)
        print(f"S_eff_PA (cps/MBq) = {s_eff_pa:.6e}")
    if args.view == "both":
        s_eff_mean = 0.5 * (float(ap_raw.sum()) + float(pa_raw.sum())) / float(args.A_total_MBq)
        print(f"S_eff_mean (cps/MBq) = {s_eff_mean:.6e}")

    # Optional: Rohprojektionen speichern (Debug/QA)
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "ap_raw.npy", ap_raw.astype(np.float32))
        np.save(out_dir / "pa_raw.npy", pa_raw.astype(np.float32))
        print(f"[OUT] ap_raw.npy, pa_raw.npy in {out_dir}")


if __name__ == "__main__":
    main()
