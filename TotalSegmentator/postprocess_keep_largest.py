import os
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import label

def keep_largest_component(arr):
    arr_bin = (arr > 0).astype(np.uint8)
    if arr_bin.sum() == 0:
        return arr
    lab, n = label(arr_bin, structure=np.ones((3,3,3), dtype=np.uint8))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes.argmax()
    return np.where(lab == keep, arr, 0).astype(arr.dtype)

def main(out_dir):
    files = [f for f in os.listdir(out_dir) if f.endswith(".nii.gz")]
    for f in files:
        p = os.path.join(out_dir, f)
        img = nib.load(p)
        data = img.get_fdata()
        cleaned = keep_largest_component(data)
        if np.any(cleaned != data):
            nib.save(nib.Nifti1Image(cleaned.astype(data.dtype), img.affine, img.header), p)
            print(f"cleaned: {f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python postprocess_keep_largest.py <output_dir>")
        sys.exit(1)
    main(sys.argv[1])