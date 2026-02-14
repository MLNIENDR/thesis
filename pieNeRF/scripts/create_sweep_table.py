import csv
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

KEEP_PHANTOMS = ["phantom_16", "phantom_24", "phantom_30"]

METRICS = [
    "vol_mae_vol",
    "vol_rmse_vol",
    "activity_rel_abs_error_vol",
    "organ_rel_error_total_activity_active_mean",
    "proj_poisson_dev_counts",
    "pred_outside_mask_frac",
]

def robust_read_csv(path):
    header = next(csv.reader(open(path)))
    n_expected = len(header)

    rows_ok = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) == n_expected:
                rows_ok.append(row)

    df = pd.DataFrame(rows_ok, columns=header)
    return df


def compute_run_stats(metrics_path):
    df = robust_read_csv(metrics_path)

    df = df[df["phantom_id"].isin(KEEP_PHANTOMS)]
    df = df.sort_values("timestamp").groupby("phantom_id").tail(1)

    df[METRICS] = df[METRICS].astype(float)

    means = df[METRICS].mean()
    stds = df[METRICS].std()

    return means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", required=True,
                        help="Path to results_spect_sweep")
    parser.add_argument("--tag-prefix", required=True,
                        help="Prefix like actW_ or projW_")
    args = parser.parse_args()

    sweep_root = Path(args.sweep_root)

    cleaned_prefix = args.tag_prefix.rstrip("_").strip("_") or "tag"
    output_dir = sweep_root / f"{cleaned_prefix}_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_dirs = sorted(
        [d for d in sweep_root.iterdir()
         if d.is_dir() and d.name.startswith(args.tag_prefix)]
    )

    table = {}

    for tag_dir in tag_dirs:
        metrics_path = tag_dir / "postproc" / "metrics.csv"
        if not metrics_path.exists():
            print(f"[WARN] missing {metrics_path}")
            continue

        means, stds = compute_run_stats(metrics_path)

        param_value = tag_dir.name.replace(args.tag_prefix, "")
        col_name = f"{args.tag_prefix}{param_value}"

        table[col_name] = [
            f"{means[m]:.4g} ± {stds[m]:.2g}" for m in METRICS
        ]

    df_table = pd.DataFrame(table, index=METRICS)

    print("\n=== Sweep Table ===\n")
    print(df_table)

    # Save CSV
    out_path = output_dir / f"{args.tag_prefix.rstrip('_')}_sweep_table.csv"
    df_table.to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
