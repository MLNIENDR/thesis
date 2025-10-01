import numpy as np
import nibabel as nib
import subprocess
import os
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

try:
    from skimage.exposure import match_histograms, equalize_adapthist
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

def gaussian_psf(img, fwhm_mm, voxel_mm):
    if fwhm_mm <= 0:
        return img
    sig_mm = fwhm_mm / 2.3548
    sigma = [sig_mm / voxel_mm[i] for i in range(3)]
    return gaussian_filter(img, sigma=sigma, mode="nearest")

def add_gaussian_noise(img, std_hu):
    if std_hu <= 0:
        return img
    rng = np.random.default_rng()
    return img + rng.normal(0.0, std_hu, size=img.shape).astype(img.dtype)

def add_poisson_like(img, scale_hu=1.0):
    if scale_hu <= 0:
        return img
    rng = np.random.default_rng()
    var = (np.clip((img + 1000.0)/1000.0, 0.1, 2.0) * scale_hu)**2
    noise = rng.normal(0.0, np.sqrt(var), size=img.shape)
    return img + noise.astype(img.dtype)

def add_bias_field(img, amp_hu=15.0, fwhm_mm=120.0, voxel_mm=(1.0,1.0,1.0)):
    if amp_hu <= 0:
        return img
    rng = np.random.default_rng()
    low = rng.normal(0.0, 1.0, size=img.shape).astype(np.float32)
    sig_mm = fwhm_mm / 2.3548
    sigma = [sig_mm / voxel_mm[i] for i in range(3)]
    field = gaussian_filter(low, sigma=sigma, mode="nearest")
    field = field / (np.std(field) + 1e-6) * amp_hu
    return img + field.astype(img.dtype)

def hist_match_to_ref(img_hu, ref_path):
    if not _HAS_SKIMAGE:
        print("WARNUNG: scikit-image nicht installiert -> Histogramm-Matching übersprungen.")
        return img_hu
    ref = nib.load(ref_path).get_fdata().astype(np.float32)
    mask_img = img_hu > -900
    mask_ref = ref > -900
    matched = img_hu.copy()
    if mask_img.sum() > 1000 and mask_ref.sum() > 1000:
        matched[mask_img] = match_histograms(img_hu[mask_img], ref[mask_ref], channel_axis=None)
    else:
        print("WARNUNG: zu wenig Voxels für Histogramm-Matching.")
    return matched.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(
        description="XCAT μ(mm^-1) → HU → CT-like Vorverarbeitung → NIfTI → TotalSegmentator"
    )
    parser.add_argument('-i','--input', required=True)
    parser.add_argument('-o','--output', required=True)
    parser.add_argument('--shape', nargs=3, type=int, default=[256,256,651])
    parser.add_argument('--voxel', nargs=3, type=float, default=[1.5,1.5,1.5])
    parser.add_argument('--mu_water_mm', type=float, default=0.0190)
    parser.add_argument('--auto_mu', action='store_true')
    parser.add_argument('--clip', nargs=2, type=int, default=[-200,300])
    parser.add_argument('--int16_out', action='store_true')

    # CT-like Optionen
    parser.add_argument('--psf_fwhm_mm', type=float, default=2.0)
    parser.add_argument('--noise_gauss_hu', type=float, default=8.0)
    parser.add_argument('--noise_poisson_hu', type=float, default=0.0)
    parser.add_argument('--bias_amp_hu', type=float, default=12.0)
    parser.add_argument('--bias_fwhm_mm', type=float, default=140.0)
    parser.add_argument('--hist_ref', type=str, default=None)
    parser.add_argument('--plot_hist', action='store_true')

    # TS-Settings
    parser.add_argument('--nr', type=int, default=16)
    parser.add_argument('--ns', type=int, default=16)
    parser.add_argument('--device', choices=['gpu','cpu','mps'], default='gpu')
    parser.add_argument('--ts_body_crop', action='store_true')
    parser.add_argument('--roi_subset', nargs='*', default=[])
    parser.add_argument('--skip_seg', action='store_true')
    args = parser.parse_args()

    # ---- .bin laden
    print(f"Lade {args.input} ...")
    raw = np.fromfile(args.input, dtype=np.float32)
    expected = np.prod(args.shape)
    if raw.size != expected:
        raise ValueError(f"Elementanzahl passt nicht zu shape {args.shape}: {raw.size} vs {expected}")
    mu_mm = raw.reshape(args.shape, order='F').astype(np.float32)

    # ---- µ_Wasser
    if args.auto_mu:
        vals = mu_mm[mu_mm > 0.001].ravel()
        hist, bin_edges = np.histogram(vals, bins=500)
        mu_water = float((bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist)+1]) / 2.0)
    else:
        mu_water = float(args.mu_water_mm)

    print(f"µ_Wasser = {mu_water:.6f} mm^-1")

    # ---- HU
    hu = 1000.0 * (mu_mm - mu_water) / mu_water
    hu[mu_mm == 0.0] = -1000.0
    hu = hu.astype(np.float32)

    voxel = tuple(args.voxel)

    # CT-like: PSF, Bias, Noise
    if args.psf_fwhm_mm > 0:
        hu = gaussian_psf(hu, args.psf_fwhm_mm, voxel)
        print(f"PSF angewendet: FWHM={args.psf_fwhm_mm} mm")

    if args.bias_amp_hu > 0:
        hu = add_bias_field(hu, amp_hu=args.bias_amp_hu, fwhm_mm=args.bias_fwhm_mm, voxel_mm=voxel)
        print(f"Bias-Feld: amp={args.bias_amp_hu} HU, FWHM={args.bias_fwhm_mm} mm")

    if args.noise_poisson_hu > 0:
        hu = add_poisson_like(hu, scale_hu=args.noise_poisson_hu)
        print(f"Poisson-ähnliches Rauschen: scale={args.noise_poisson_hu} HU")

    if args.noise_gauss_hu > 0:
        hu = add_gaussian_noise(hu, std_hu=args.noise_gauss_hu)
        print(f"Gauss-Rauschen: std={args.noise_gauss_hu} HU")

    # ---- CLAHE (lokaler Kontrast)
    if _HAS_SKIMAGE:
        from skimage import exposure
        body = hu > -900
        lo, hi = -300.0, 300.0
        hu_win = np.clip((hu - lo) / (hi - lo), 0, 1)
        hu_eq = hu_win
        if body.sum() > 0:
            hu_eq = hu_win
            hu_eq[body] = exposure.equalize_adapthist(hu_win[body], clip_limit=0.02, nbins=256)
        hu = hu_eq * (hi - lo) + lo
        print("CLAHE angewendet.")
    else:
        print("CLAHE übersprungen (scikit-image fehlt).")

    # ---- Unsharp Mask
    blurred = gaussian_filter(hu, sigma=[0.8/voxel[0], 0.8/voxel[1], 0.8/voxel[2]], mode="nearest")
    hu = (hu + 0.3*(hu - blurred)).astype(np.float32)
    print("Unsharp Mask angewendet.")

    # ---- Histogramm-Matching
    if args.hist_ref:
        hu = hist_match_to_ref(hu, args.hist_ref)
        print(f"Histogramm-Matching gegen: {args.hist_ref}")

    # ---- Clip & Save
    hu = np.clip(hu, args.clip[0], args.clip[1])
    print(f"HU nach CT-like: min={hu.min():.1f}, max={hu.max():.1f}, mean={hu.mean():.1f}")

    if args.plot_hist:
        plt.figure(figsize=(8,4))
        plt.hist(hu.ravel(), bins=200)
        plt.xlabel("HU"); plt.ylabel("Voxelanzahl")
        plt.title("HU-Histogramm nach CT-like")
        plt.savefig("ctlike_hist.png", dpi=150, bbox_inches='tight')

    affine = np.diag(list(voxel) + [1.0])
    nii_path = os.path.splitext(os.path.basename(args.input))[0] + "_ctlike.nii.gz"
    data = np.round(hu).astype(np.int16) if args.int16_out else hu.astype(np.float32)
    img = nib.Nifti1Image(data, affine); img.set_qform(affine); img.set_sform(affine)
    nib.save(img, nii_path)
    print(f"NIfTI gespeichert: {nii_path}")

    if args.skip_seg:
        print("Segmentierung übersprungen (--skip_seg).")
        return

    ts = ["TotalSegmentator","-i",nii_path,"-o",args.output,
          "-nr",str(args.nr),"-ns",str(args.ns),"-d",args.device]
    if args.ts_body_crop:
        ts.append("-bs")
    if args.roi_subset:
        ts.extend(["-rs"] + args.roi_subset)
    print("Starte TotalSegmentator:", " ".join(ts))
    subprocess.run(ts, check=True)
    print(f"Segmentierung fertig in {args.output}")

if __name__ == "__main__":
    main()