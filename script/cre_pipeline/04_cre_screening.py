#!/usr/bin/env python3
"""Step 3: screen infection-specific CRE candidates from ISM scores."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from common import ensure_dir, flatten_numeric, save_json, write_bed


CLUSTER_NAMES = {
    "early": "early_response",
    "middle": "transient_response",
    "late": "late_response",
    "sustained": "sustained_response",
}


def determine_threshold(delta: np.ndarray, background: np.ndarray | None, percentile: float) -> Dict[str, float]:
    if background is not None:
        bg = flatten_numeric(background)
        threshold = float(np.nanmean(bg) + 2.0 * np.nanstd(bg))
        return {
            "delta_importance_threshold": threshold,
            "method": "background_mean_plus_2std",
            "background_mean": float(np.nanmean(bg)),
            "background_std": float(np.nanstd(bg)),
            "percentile": percentile,
        }
    values = flatten_numeric(delta)
    threshold = float(np.nanpercentile(values, percentile))
    return {
        "delta_importance_threshold": threshold,
        "method": "delta_percentile",
        "percentile": percentile,
        "background_mean": float("nan"),
        "background_std": float("nan"),
    }


def connected_true_regions(mask: np.ndarray, merge_gap: int) -> List[Tuple[int, int]]:
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return []
    regions = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if idx - prev <= merge_gap + 1:
            prev = idx
            continue
        regions.append((start, prev + 1))
        start = idx
        prev = idx
    regions.append((start, prev + 1))
    return regions


def trim_region_to_max_width(scores: np.ndarray, start: int, end: int, max_width: int) -> Tuple[int, int]:
    width = end - start
    if width <= max_width:
        return start, end
    local = scores[start:end]
    center = int(np.nanargmax(local)) + start
    new_start = max(start, center - max_width // 2)
    new_end = min(end, new_start + max_width)
    new_start = max(start, new_end - max_width)
    return new_start, new_end


def interpret_cluster(center: np.ndarray) -> str:
    if len(center) == 0:
        return "unknown"
    finite = np.nan_to_num(center, nan=0.0)
    max_idx = int(np.argmax(finite))
    if np.all(finite > 0) and finite.min() >= 0.5 * max(finite.max(), 1e-8):
        return CLUSTER_NAMES["sustained"]
    if max_idx <= max(0, len(finite) // 3 - 1):
        return CLUSTER_NAMES["early"]
    if max_idx >= 2 * len(finite) // 3:
        return CLUSTER_NAMES["late"]
    return CLUSTER_NAMES["middle"]


def cluster_temporal(delta_npz: Path, n_clusters: int) -> pd.DataFrame:
    if not delta_npz.exists():
        return pd.DataFrame(columns=["peak_id", "cluster_id", "cluster_label", "cluster_importance"])
    data = np.load(delta_npz)
    timepoints = list(data.files)
    if not timepoints:
        return pd.DataFrame(columns=["peak_id", "cluster_id", "cluster_label", "cluster_importance"])
    features = np.vstack([np.nanmean(data[tp], axis=1) for tp in timepoints]).T
    features = np.nan_to_num(features, nan=0.0)
    n_regions = features.shape[0]
    if n_regions == 0:
        return pd.DataFrame(columns=["peak_id", "cluster_id", "cluster_label", "cluster_importance"])
    k = max(1, min(n_clusters, n_regions))
    try:
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(features)
        centers = model.cluster_centers_
    except Exception:
        labels = np.argmax(features, axis=1) % k
        centers = np.vstack([features[labels == i].mean(axis=0) for i in range(k)])
    label_names = {idx: interpret_cluster(centers[idx]) for idx in range(k)}
    return pd.DataFrame(
        {
            "peak_id": [f"peak_{i:06d}" for i in range(n_regions)],
            "cluster_id": labels.astype(int),
            "cluster_label": [label_names[int(x)] for x in labels],
            "cluster_importance": features.mean(axis=1),
        }
    )


def define_cre_boundaries(
    delta: np.ndarray,
    metadata: pd.DataFrame,
    cluster_df: pd.DataFrame,
    threshold: float,
    *,
    min_width: int,
    max_width: int,
    merge_gap: int,
) -> pd.DataFrame:
    columns = [
        "chr",
        "start",
        "end",
        "cre_id",
        "importance_score",
        "peak_id",
        "peak_region_id",
        "cluster_id",
        "cluster_label",
        "cluster_importance",
    ]
    cluster_by_peak = cluster_df.set_index("peak_id").to_dict(orient="index") if len(cluster_df) else {}
    records = []
    cre_counter = 1
    for i, meta in metadata.iterrows():
        peak_id = f"peak_{i:06d}"
        length = int(meta.get("length", int(meta["end"]) - int(meta["start"])))
        scores = delta[i, :length]
        mask = np.isfinite(scores) & (scores > threshold)
        for rel_start, rel_end in connected_true_regions(mask, merge_gap):
            rel_start, rel_end = trim_region_to_max_width(scores, rel_start, rel_end, max_width)
            width = rel_end - rel_start
            if width < min_width or width > max_width:
                continue
            genomic_start = int(meta["start"]) + rel_start
            genomic_end = int(meta["start"]) + rel_end
            cluster = cluster_by_peak.get(peak_id, {})
            records.append(
                {
                    "chr": meta["chrom"],
                    "start": genomic_start,
                    "end": genomic_end,
                    "cre_id": f"cre_{cre_counter:06d}",
                    "importance_score": float(np.nanmean(scores[rel_start:rel_end])),
                    "peak_id": peak_id,
                    "peak_region_id": meta.get("region_id", peak_id),
                    "cluster_id": cluster.get("cluster_id", -1),
                    "cluster_label": cluster.get("cluster_label", "unknown"),
                    "cluster_importance": cluster.get("cluster_importance", float(np.nanmean(scores))),
                }
            )
            cre_counter += 1
    return pd.DataFrame(records, columns=columns)


def plot_distribution(delta: np.ndarray, threshold: float, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        values = flatten_numeric(delta)
        plt.figure(figsize=(6, 4))
        plt.hist(values, bins=80, color="steelblue", alpha=0.85)
        plt.axvline(threshold, color="firebrick", linestyle="--", linewidth=1)
        plt.xlabel("Delta importance")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output, dpi=300)
        plt.close()
    except Exception as exc:
        print(f"Skipping distribution plot: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ism-dir", default="results/ism")
    parser.add_argument("--output-dir", default="results/cre_candidates")
    parser.add_argument("--delta-file", default=None)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--background-file", default=None)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--min-width", type=int, default=50)
    parser.add_argument("--max-width", type=int, default=500)
    parser.add_argument("--merge-gap", type=int, default=10)
    parser.add_argument("--n-clusters", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ism_dir = Path(args.ism_dir)
    output_dir = ensure_dir(args.output_dir)
    delta_path = Path(args.delta_file) if args.delta_file else ism_dir / "delta_importance.npy"
    metadata_path = Path(args.metadata_file) if args.metadata_file else ism_dir / "region_metadata.csv"
    delta = np.load(delta_path)
    metadata = pd.read_csv(metadata_path)
    background = np.load(args.background_file) if args.background_file else None

    threshold_info = determine_threshold(delta, background, args.percentile)
    threshold = threshold_info["delta_importance_threshold"]
    save_json(threshold_info, output_dir / "thresholds.json")
    plot_distribution(delta, threshold, output_dir / "delta_importance_distribution.png")

    cluster_df = cluster_temporal(ism_dir / "delta_importance_per_timepoint.npz", args.n_clusters)
    cluster_df.to_csv(output_dir / "cluster_assignments.csv", index=False)
    cre_df = define_cre_boundaries(
        delta,
        metadata,
        cluster_df,
        threshold,
        min_width=args.min_width,
        max_width=args.max_width,
        merge_gap=args.merge_gap,
    )
    cre_df.to_csv(output_dir / "infection_specific_cre.csv", index=False)
    bed_cols = ["chr", "start", "end", "cre_id", "importance_score", "cluster_id"]
    write_bed(cre_df, output_dir / "infection_specific_cre.bed", bed_cols)
    print(f"Saved {len(cre_df)} CRE candidates to {output_dir}")


if __name__ == "__main__":
    main()
