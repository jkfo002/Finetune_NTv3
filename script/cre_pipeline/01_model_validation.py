#!/usr/bin/env python3
"""Step 1: validate model prediction quality on genomic regions."""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common import (
    add_dataloader_args,
    add_model_args,
    add_track_args,
    build_delta_pairs,
    build_score_groups,
    ensure_dir,
    load_regions,
    load_model_for_inference,
    load_project_config,
    make_dataloader,
    predict_tokens,
    resolve_atac_timepoints,
    resolve_track_design,
    save_json,
)
from model.analysis import visualization_channels_means


def pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_np(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr

        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 2:
            return float("nan")
        return float(spearmanr(x[mask], y[mask]).correlation)
    except Exception:
        return pearson_np(pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy())


def auc_np(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, scores))
    except Exception:
        order = np.argsort(scores)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(scores) + 1)
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def region_pearson_per_track(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute Pearson r per track for one region. pred/target: (seq_len, num_tracks)."""
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)
    pred_centered = pred - pred.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    cov = (pred_centered * target_centered).mean(axis=0)
    pred_std = pred_centered.std(axis=0, ddof=0)
    target_std = target_centered.std(axis=0, ddof=0)
    corr = cov / (pred_std * target_std + 1e-8)
    return np.clip(corr, -1.0, 1.0)


def region_metrics_for_sample(
    pred: np.ndarray,
    target: np.ndarray,
    score_groups: Mapping[str, Sequence[int]],
    *,
    peak_target_percentile: float,
) -> Tuple[float, Dict[int, float], Dict[str, float], Dict[str, float], Dict[str, Optional[float]]]:
    """计算单个 region 的预测-目标 track 相关性及分组指标。

    Args:
        pred: 模型预测值，shape (seq_len, num_tracks)
        target: 真实目标值，shape (seq_len, num_tracks)
        score_groups: track 分组映射，如 {"ATAC::T3": [0,1], "RNA::T3::infect": [2,3]}
        peak_target_percentile: 用于计算 peak AUC 的目标值分位数阈值

    Returns:
        pearson_global: 所有 track 的 Pearson 相关系数均值
        per_track: 每个 track 的 Pearson 相关系数，{track_idx: corr}
        per_group: 每组 track 均值后的 Pearson 相关系数，{group_key: corr}
        per_group_spearman: 每组 track 均值后的 Spearman 相关系数，{group_key: corr}
        per_group_peak_auc: 每组 peak AUC（基于目标值分位数阈值），{group_key: auc}
    """
    track_corr = region_pearson_per_track(pred, target)
    pearson_global = float(np.nanmean(track_corr))

    per_track = {int(idx): float(track_corr[idx]) for idx in range(len(track_corr))}
    per_group: Dict[str, float] = {}
    per_group_spearman: Dict[str, float] = {}
    per_group_peak_auc: Dict[str, Optional[float]] = {}
    for group_key, indices in score_groups.items():
        idx_list = list(indices)
        pred_group = pred[:, idx_list].mean(axis=1)
        target_group = target[:, idx_list].mean(axis=1)
        per_group[group_key] = pearson_np(pred_group, target_group)
        per_group_spearman[group_key] = spearman_np(pred_group, target_group)
        threshold = np.nanpercentile(target_group, peak_target_percentile)
        per_group_peak_auc[group_key] = auc_np(pred_group, target_group >= threshold)

    return pearson_global, per_track, per_group, per_group_spearman, per_group_peak_auc


def summarize_series(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
    }


def plot_region_tracks(
    target: np.ndarray,
    pred: np.ndarray,
    mean_order: Mapping[str, Sequence[int]],
    save_path: Path,
    *,
    plot_ylim: Optional[float] = None,
) -> str:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ylim = (0.0, float(plot_ylim)) if plot_ylim is not None else None
    visualization_channels_means(
        np.expand_dims(target, axis=0),
        np.expand_dims(pred, axis=0),
        dict(mean_order),
        save_path=str(save_path),
        ylim=ylim,
    )
    return str(save_path)


def _plot_region_task(args: Tuple[Any, ...]) -> str:
    target, pred, mean_order, save_path, plot_ylim = args
    return plot_region_tracks(
        target,
        pred,
        mean_order,
        Path(save_path),
        plot_ylim=plot_ylim,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    add_track_args(parser)
    add_dataloader_args(parser)
    parser.add_argument("--regions-bed", default=None, help="Optional BED regions. Defaults to config gene_bed.")
    parser.add_argument("--output-dir", default="results/model_performance")
    parser.add_argument("--peak-target-percentile", type=float, default=90.0)
    parser.add_argument("--plot", action="store_true", help="Save per-region track plots.")
    parser.add_argument(
        "--plot-limit-regions",
        type=float,
        default=None,
        help="Upper Y-axis limit for track plots (lower bound fixed at 0).",
    )
    parser.add_argument("--plot-workers", type=int, default=8, help="Worker processes for plotting.")
    return parser.parse_args()


def main() -> None:

    # Parse args
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    plots_dir = output_dir / "plots"
    config = load_project_config(args.config)
    design = resolve_track_design(config, args.track_map)
    score_groups = build_score_groups(design)
    mean_order = score_groups # for plot func 聚合 track
    baseline, infection = resolve_atac_timepoints(design, args.baseline_timepoint, args.infection_timepoints) # resolve timepoint
    delta_pairs = build_delta_pairs(design, baseline, infection) # build_delta
    num_tracks = int(config["num_tracks"])

    # load module
    model, tokenizer, device = load_model_for_inference(
        config,
        args.ckpt,
        args.device,
        compile_model=args.compile_model,
        strict=args.strict,
    )

    # load region
    if args.regions_bed is None:
        print(f"Trying to load region bed from config toml {args.config['gene_bed']}")
        if args.config['gene_bed'] is None:
            raise ValueError(f"No region were detected from both {args.regions_bed} and {args.config}")
    regions = load_regions(config, bed_path=args.regions_bed, limit=args.limit_regions)

    # load dataloader
    dataloader = make_dataloader(
        config,
        tokenizer,
        regions,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    region_rows: List[Dict[str, Any]] = []
    region_track_rows: List[Dict[str, Any]] = []
    region_group_rows: List[Dict[str, Any]] = []
    region_delta_rows: List[Dict[str, Any]] = []
    plot_pool: Optional[Pool] = None
    if args.plot:
        plot_pool = Pool(processes=max(1, int(args.plot_workers))) # multiprocessing pool

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            targets = batch["bigwig_targets"].to(device)
            preds = predict_tokens(model, batch["tokens"], device)

            pred_cpu = preds.detach().float().cpu().numpy()
            target_cpu = targets.detach().float().cpu().numpy()
            batch_plot_jobs: List[Tuple[Any, ...]] = []

            for i, chrom in enumerate(batch["chrom"]):
                start = int(batch["start"][i])
                end = int(batch["end"][i])
                region_id = f"{chrom}_{start}_{end}"
                pred_i = pred_cpu[i]
                target_i = target_cpu[i]

                # pearson 计算
                pearson_global, per_track, per_group, per_group_spearman, per_group_peak_auc = region_metrics_for_sample(
                    pred_i,
                    target_i,
                    score_groups,
                    peak_target_percentile=args.peak_target_percentile,
                )

                """
                记录 region 级指标
                float(np.nanmean(pred_i)) 记录track聚合均值
                """
                region_rows.append(
                    {
                        "region_id": region_id,
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "mean_pred": float(np.nanmean(pred_i)),
                        "mean_target": float(np.nanmean(target_i)),
                        "pearson_global": pearson_global,
                    }
                )

                """
                记录 track 级指标
                """
                for track_idx, value in per_track.items():
                    region_track_rows.append(
                        {
                            "region_id": region_id,
                            "track_idx": track_idx,
                            "pearson": value,
                        }
                    )

                """
                记录 group 级指标
                """
                for group_key, value in per_group.items():
                    region_group_rows.append(
                        {
                            "region_id": region_id,
                            "group_key": group_key,
                            "pearson": value,
                            "spearman": per_group_spearman[group_key],
                            "peak_auc": per_group_peak_auc[group_key],
                        }
                    )
                
                # delta 相关性
                for delta_key, (lhs_key, rhs_key) in delta_pairs.items():
                    lhs = score_groups.get(lhs_key, [])
                    rhs = score_groups.get(rhs_key, [])
                    if not lhs or not rhs:
                        continue
                    pred_delta = pred_i[:, lhs].mean(axis=1) - pred_i[:, rhs].mean(axis=1)
                    true_delta = target_i[:, lhs].mean(axis=1) - target_i[:, rhs].mean(axis=1)
                    region_delta_rows.append(
                        {
                            "region_id": region_id,
                            "delta_key": delta_key,
                            "lhs_group": lhs_key,
                            "rhs_group": rhs_key,
                            "pearson_delta": pearson_np(pred_delta, true_delta),
                            "spearman_delta": spearman_np(pred_delta, true_delta),
                        }
                    )

                if args.plot:
                    chrom_safe = str(chrom).replace("/", "_")
                    save_path = plots_dir / chrom_safe / f"{region_id}.png"
                    batch_plot_jobs.append(
                        (
                            target_i.copy(),
                            pred_i.copy(),
                            mean_order,
                            str(save_path),
                            args.plot_limit_regions,
                        )
                    )

            if plot_pool is not None and batch_plot_jobs:
                list(plot_pool.imap(_plot_region_task, batch_plot_jobs))

    if plot_pool is not None:
        plot_pool.close()
        plot_pool.join()

    region_group_df = pd.DataFrame(region_group_rows)
    region_track_df = pd.DataFrame(region_track_rows)
    region_delta_df = pd.DataFrame(region_delta_rows)

    group_summary: Dict[str, Dict[str, Any]] = {}
    for group_key, indices in score_groups.items():
        group_df = region_group_df[region_group_df["group_key"] == group_key]
        peak_auc_values = [
            float(v)
            for v in group_df["peak_auc"].tolist()
            if v is not None and np.isfinite(v)
        ]
        entry: Dict[str, Any] = {
            "track_indices": list(indices),
            "pearson": summarize_series(group_df["pearson"].tolist()),
            "spearman": summarize_series(group_df["spearman"].tolist()),
        }
        if peak_auc_values:
            entry["peak_auc"] = summarize_series(peak_auc_values)
        group_summary[group_key] = entry

    delta_summary: Dict[str, Dict[str, Any]] = {}
    for delta_key in delta_pairs:
        delta_df = region_delta_df[region_delta_df["delta_key"] == delta_key]
        if delta_df.empty:
            continue
        lhs_key, rhs_key = delta_pairs[delta_key]
        delta_summary[delta_key] = {
            "lhs_group": lhs_key,
            "rhs_group": rhs_key,
            "pearson_delta": summarize_series(delta_df["pearson_delta"].tolist()),
            "spearman_delta": summarize_series(delta_df["spearman_delta"].tolist()),
        }

    track_summary: Dict[str, Dict[str, float]] = {}
    for track_idx in range(num_tracks):
        track_df = region_track_df[region_track_df["track_idx"] == track_idx]
        track_summary[f"track{track_idx}"] = summarize_series(track_df["pearson"].tolist())

    global_metrics = {
        "num_regions": len(region_rows),
        "num_tracks": num_tracks,
        "track_design": design,
        "score_groups": score_groups,
        "atac_baseline_timepoint": baseline,
        "atac_infection_timepoints": infection,
        "group_summary": group_summary,
        "delta_summary": delta_summary,
        "track_summary": track_summary,
        "plot_enabled": bool(args.plot),
        "plot_y_max": args.plot_limit_regions,
    }

    save_json(global_metrics, output_dir / "global_metrics.json")
    pd.DataFrame(region_rows).to_csv(output_dir / "region_prediction_summary.csv", index=False)
    region_track_df.to_csv(output_dir / "region_pearson_per_track.csv", index=False)
    region_group_df.to_csv(output_dir / "region_pearson_per_group.csv", index=False)
    region_delta_df.to_csv(output_dir / "region_delta_metrics.csv", index=False)
    pd.DataFrame.from_dict(track_summary, orient="index").to_csv(output_dir / "track_summary.csv")
    print(f"Saved validation outputs to {output_dir}")


if __name__ == "__main__":
    main()
