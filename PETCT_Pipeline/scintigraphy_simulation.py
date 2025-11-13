#!/usr/bin/env python

"""
scintigraphy_simulation.py

Simuliert planare Szintigrafie-Projektionen (AP/PA) aus einem PET-Volumen,
unter Verwendung einer CT-Serie als Abschwächungskarte.

- PET-DICOM -> Aktivitätsvolumen (counts o.ä.)
- CT-DICOM  -> HU-Volumen -> µ-Map (140 keV, stark vereinfacht)
- Gamma-Kamera-Vorwärtsmodell:
    (1) xy-Gauß-Streuung pro Schicht (optional)
    (2) exponentielle Abschwächung entlang y (optional)
    (3) xy-Gauß-Kollimatorunschärfe pro Schicht (optional)
    (4) Projektion = Summe über z

AP/PA werden separat berechnet (einmal „von vorne“, einmal „von hinten“).

Das ist bewusst ein vereinfachtes physikalisches Modell, reicht aber gut
für synthetische Szintigrafien aus klinischen PET/CT-Daten.

Aufruf:

python scintigraphy_simulation.py \
  --pet_dicom_dir /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/PET_WB_TRUEX2I14S0MM_AC_0102 \
  --ct_dicom_dir  /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/CT_WB_5_0_B30F_0005 \
  --output_dir    /home/mnguest12/projects/thesis/PETCT_Pipeline/data/example_01/results/scintigraphy_sim \
  --scatter_sigma_xy 2.0 \
  --coll_sigma_xy 2.0

"""

import os
import argparse
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# DICOM I/O
# -------------------------------------------------------------
def read_dicom_series(dicom_dir: str) -> sitk.Image:
    """
    Liest eine DICOM-Serie (z.B. PET oder CT) mit SimpleITK ein.
    """
    if not os.path.isdir(dicom_dir):
        raise RuntimeError(f"Verzeichnis existiert nicht (oder ist kein Ordner): {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"Keine DICOM-Serien gefunden in: {dicom_dir}")

    # Wenn mehrere Serien im Ordner sind, nimm die erste
    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
    reader.SetFileNames(file_names)

    image = reader.Execute()
    return image


def resample_to_reference(
    img: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """
    Resample 'img' in den Raum von 'reference' (gleiche Größe, Spacing, Origin, Direction).
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    return resampler.Execute(img)


# -------------------------------------------------------------
# CT -> µ-Map (stark vereinfacht)
# -------------------------------------------------------------
def ct_hu_to_mu(ct_hu: np.ndarray, energy_keV: float = 140.0) -> np.ndarray:
    """
    Sehr einfache HU -> µ-Abbildung.

    Annahme:
      - Wasser bei 140 keV: µ_water ≈ 0.15 1/cm (Pi-mal-Daumen)
      - relative Dichte ~ (HU / 1000 + 1)
      - daraus: µ = µ_water * (HU/1000 + 1)

    """
    mu_water = 0.15  # 1/cm, nur als Maßstab
    rho_rel = (ct_hu / 1000.0) + 1.0
    mu = mu_water * rho_rel
    mu = np.clip(mu, 0.0, None)  # keine negativen µ
    return mu.astype(np.float32)


# -------------------------------------------------------------
# Gamma-Kamera-Vorwärtsmodell (zyx, AP/PA = koronal)
# -------------------------------------------------------------

def _forward_single_view_zyx(
    act_zyx: np.ndarray,
    mu_zyx: np.ndarray,
    scatter_sigma_xy: float,
    coll_sigma_xy: float,
    use_scatter: bool,
    use_attenuation: bool,
    use_collimator: bool,
    spacing_y_mm: float,
) -> np.ndarray:
    """
    Vorwärtsmodell für EINE Blickrichtung entlang der y-Achse (AP ODER PA).

    Input-Shape: (nz, ny, nx) = (z, y, x)
    - Scatter/Kollimator: Gauß in der Bildebene (z,x) für jede y-Schicht
    - Abschwächung: exp(-∫ µ dy) entlang y
    - Projektion: Summe über y -> Bild (nz, nx) = koronal
    """
    assert act_zyx.shape == mu_zyx.shape
    nz, ny, nx = act_zyx.shape

    # (1) Streuung in der Detektorebene (z,x) pro y-Schicht
    if use_scatter and scatter_sigma_xy > 0:
        act_sc = np.empty_like(act_zyx, dtype=np.float32)
        for j in range(ny):
            act_sc[:, j, :] = gaussian_filter(
                act_zyx[:, j, :],
                sigma=scatter_sigma_xy,
                mode="nearest",
            )
    else:
        act_sc = act_zyx.astype(np.float32, copy=True)

    # (2) Abschwächung entlang y (µ in 1/cm, dy in cm)
    if use_attenuation:
        spacing_y_cm = spacing_y_mm / 10.0
        mu_cum = np.cumsum(mu_zyx * spacing_y_cm, axis=1)
        vol_atn = act_sc * np.exp(-mu_cum)
    else:
        vol_atn = act_sc

    # (3) Kollimatorunschärfe (wieder in (z,x) pro y-Schicht)
    if use_collimator and coll_sigma_xy > 0:
        vol_coll = np.empty_like(vol_atn, dtype=np.float32)
        for j in range(ny):
            vol_coll[:, j, :] = gaussian_filter(
                vol_atn[:, j, :],
                sigma=coll_sigma_xy,
                mode="nearest",
            )
    else:
        vol_coll = vol_atn

    # (4) Projektion: Summe über y -> koronales Bild (z,x)
    proj = np.sum(vol_coll, axis=1)  # shape (nz, nx)
    return proj


def gamma_camera_forward_zyx(
    act_zyx: np.ndarray,
    mu_zyx: np.ndarray,
    scatter_sigma_xy: float = 2.0,
    coll_sigma_xy: float = 2.0,
    use_scatter: bool = True,
    use_attenuation: bool = True,
    use_collimator: bool = True,
    spacing_y_mm: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    AP/PA-Projektion aus Volumen (nz, ny, nx) mit Integration entlang y.
    Ergebnis: proj_AP, proj_PA jeweils (nz, nx) = koronal.
    """
    assert act_zyx.shape == mu_zyx.shape

    # AP: entlang +y
    proj_ap = _forward_single_view_zyx(
        act_zyx=act_zyx,
        mu_zyx=mu_zyx,
        scatter_sigma_xy=scatter_sigma_xy,
        coll_sigma_xy=coll_sigma_xy,
        use_scatter=use_scatter,
        use_attenuation=use_attenuation,
        use_collimator=use_collimator,
        spacing_y_mm=spacing_y_mm,
    )

    # PA: entlang -y (y-Achse umdrehen)
    proj_pa = _forward_single_view_zyx(
        act_zyx=act_zyx[:, ::-1, :],
        mu_zyx=mu_zyx[:, ::-1, :],
        scatter_sigma_xy=scatter_sigma_xy,
        coll_sigma_xy=coll_sigma_xy,
        use_scatter=use_scatter,
        use_attenuation=use_attenuation,
        use_collimator=use_collimator,
        spacing_y_mm=spacing_y_mm,
    )

    return proj_ap, proj_pa


# -------------------------------------------------------------
# Plot-Helper (einfach, wie in der „guten“ Version)
# -------------------------------------------------------------
def save_projection_png(proj: np.ndarray, path: str, title: str,
                        spacing_x: float, spacing_z: float):
    """
    proj hat Shape (nz, nx) = (z, x) im PET-Space.
    Wir stellen z (kranio-kaudal) vertikal, x (links-rechts) horizontal dar.
    Die physikalische Geometrie wird über 'extent' in mm korrekt abgebildet.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    nz, nx = proj.shape  # (z, x)

    # OPTIONAL: Kopf nach oben, Füße nach unten – je nach Konvention
    # Wenn dir die bisherige Orientierung lieber war, kannst du hier
    # mit np.flipud / np.fliplr arbeiten.
    proj_disp = proj  # oder z.B. proj[::-1, :] falls nötig

    x_min, x_max = 0.0, nx * spacing_x
    z_min, z_max = 0.0, nz * spacing_z

    plt.figure(figsize=(5, 7))
    im = plt.imshow(
        proj_disp,
        cmap="inferno",
        origin="lower",
        extent=[x_min, x_max, z_min, z_max],  # in mm
        aspect="equal",  # gleiche Skalierung in x- und z-Richtung (in mm)
    )
    plt.title(title)
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# -------------------------------------------------------------
# Haupt-Pipeline (Baseline-Version: CT -> PET-Space)
# -------------------------------------------------------------
def run_scintigraphy_simulation(
    pet_dicom_dir: str,
    ct_dicom_dir: str,
    output_dir: str,
    scatter_sigma_xy: float = 2.0,
    coll_sigma_xy: float = 2.0,
    use_scatter: bool = True,
    use_attenuation: bool = True,
    use_collimator: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Lese PET-DICOM aus: {pet_dicom_dir}")
    pet_img = read_dicom_series(pet_dicom_dir)
    # PET-Voxelabstände (SimpleITK: (x,y,z) in mm)
    spacing_x, spacing_y, spacing_z = pet_img.GetSpacing()

    print(f"[INFO] Lese CT-DICOM aus:  {ct_dicom_dir}")
    ct_img = read_dicom_series(ct_dicom_dir)

    print("[INFO] Resample CT -> PET-Space ...")
    ct_resampled = resample_to_reference(
        ct_img,
        reference=pet_img,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,  # Luft
    )

    # Arrays im PET-Space (SimpleITK: [z, y, x])
    pet_arr_zyx = sitk.GetArrayFromImage(pet_img).astype(np.float32)
    ct_arr_zyx = sitk.GetArrayFromImage(ct_resampled).astype(np.float32)
    

    # HU -> µ
    print("[INFO] Erzeuge µ-Map aus CT-HU ...")
    mu_zyx = ct_hu_to_mu(ct_arr_zyx, energy_keV=140.0)

    # Forward-Modell direkt auf (z,y,x)
    print("[INFO] Starte Gamma-Kamera-Vorwärtsmodell (AP/PA, koronal) ...")
    proj_ap, proj_pa = gamma_camera_forward_zyx(
        act_zyx=pet_arr_zyx,
        mu_zyx=mu_zyx,
        scatter_sigma_xy=scatter_sigma_xy,
        coll_sigma_xy=coll_sigma_xy,
        use_scatter=use_scatter,
        use_attenuation=use_attenuation,
        use_collimator=use_collimator,
        spacing_y_mm=spacing_y,
    )

    # Speichern als NPY
    np.save(os.path.join(output_dir, "projection_AP.npy"), proj_ap.astype(np.float32))
    np.save(os.path.join(output_dir, "projection_PA.npy"), proj_pa.astype(np.float32))
    print(f"[INFO] NPYs gespeichert in: {output_dir}")

    # PNG-Heatmaps
    save_projection_png(
        proj_ap,
        os.path.join(output_dir, "projection_AP.png"),
        "Simulated Scintigraphy – AP",
        spacing_x=spacing_x,
        spacing_z=spacing_z,
    )
    save_projection_png(
        proj_pa,
        os.path.join(output_dir, "projection_PA.png"),
        "Simulated Scintigraphy – PA",
        spacing_x=spacing_x,
        spacing_z=spacing_z,
    )
    print(f"[INFO] PNGs gespeichert in: {output_dir}")

    return proj_ap, proj_pa


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Simulierte Szintigrafie aus PET+CT mittels einfachem Gamma-Kamera-Modell."
    )
    parser.add_argument(
        "--pet_dicom_dir",
        required=True,
        help="Pfad zum DICOM-Ordner der PET-Serie (wird als Aktivitätsvolumen genutzt).",
    )
    parser.add_argument(
        "--ct_dicom_dir",
        required=True,
        help="Pfad zum DICOM-Ordner der CT-Serie (wird als Abschwächungskarte genutzt).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Ausgabeordner für Projektionen (NPY + PNG).",
    )
    parser.add_argument(
        "--scatter_sigma_xy",
        type=float,
        default=2.0,
        help="Sigma der Gauß-Streuung in der (z,x)-Ebene (in Pixeln). 0 = aus.",
    )
    parser.add_argument(
        "--coll_sigma_xy",
        type=float,
        default=2.0,
        help="Sigma der Gauß-Kollimatorunschärfe in der (z,x)-Ebene (in Pixeln). 0 = aus.",
    )
    parser.add_argument(
        "--no_scatter",
        action="store_true",
        help="Streuung abschalten.",
    )
    parser.add_argument(
        "--no_attenuation",
        action="store_true",
        help="Abschwächung abschalten.",
    )
    parser.add_argument(
        "--no_collimator",
        action="store_true",
        help="Kollimatorunschärfe abschalten.",
    )

    args = parser.parse_args()

    run_scintigraphy_simulation(
        pet_dicom_dir=args.pet_dicom_dir,
        ct_dicom_dir=args.ct_dicom_dir,
        output_dir=args.output_dir,
        scatter_sigma_xy=args.scatter_sigma_xy,
        coll_sigma_xy=args.coll_sigma_xy,
        use_scatter=not args.no_scatter,
        use_attenuation=not args.no_attenuation,
        use_collimator=not args.no_collimator,
    )


if __name__ == "__main__":
    main()