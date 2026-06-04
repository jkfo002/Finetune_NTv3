#!/usr/bin/env python3
"""Step 2: run N-mask ISM over 51.2 kb gene regions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import (
    add_model_args,
    add_track_args,
    build_delta_pairs,
    build_score_groups,
    ensure_dir,
    load_gene_regions_from_config,
    load_mask_intervals_json,
    load_model_for_inference,
    load_project_config,
    predict_tokens,
    resolve_atac_timepoints,
    resolve_track_design,
    save_json,
    sliding_mask_windows,
    tokenize_sequences,
)
from model.dataset import GenomeBigWigMaskISMDataset


def compute_prediction_scores(
    logits: torch.Tensor,
    score_groups: Mapping[str, Sequence[int]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Aggregate logits (batch, seq_len, num_tracks) into scalar scores."""
    per_track = logits.mean(dim=1).detach().float().cpu().numpy()
    overall = per_track.mean(axis=1)
    groups = {
        key: logits[..., list(indices)].mean(dim=(1, 2)).detach().float().cpu().numpy()
        for key, indices in score_groups.items()
    }
    return overall, per_track, groups


def compute_delta_scores(
    group_scores: Mapping[str, float],
    delta_pairs: Mapping[str, Tuple[str, str]],
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for delta_key, (lhs_key, rhs_key) in delta_pairs.items():
        if lhs_key not in group_scores or rhs_key not in group_scores:
            continue
        deltas[delta_key] = float(group_scores[lhs_key]) - float(group_scores[rhs_key])
    return deltas


def resolve_mask_windows(
    scheme: str,
    *,
    seq_length: int,
    region_start_genome: int,
    scan_start: int,
    scan_end: int,
    window_size: int,
    window_step: int,
    max_windows: int | None,
    region_id: str,
    fixed_intervals: Mapping[str, Sequence[Tuple[int, int]]] | None,
) -> List[Tuple[int, int]]:
    if scheme == "sliding":
        return sliding_mask_windows(
            scan_start,
            scan_end,
            window_size,
            window_step,
            max_windows=max_windows,
        )
    if scheme == "fixed":
        if fixed_intervals is None:
            raise ValueError("Fixed ISM scheme requires --mask-intervals-json.")
        absolute_windows = list(fixed_intervals.get(region_id, []))
        if not absolute_windows:
            raise ValueError(f"No fixed mask intervals found for region id {region_id!r}.")
        windows: List[Tuple[int, int]] = []
        region_end_genome = region_start_genome + seq_length
        for abs_start, abs_end in absolute_windows:
            if abs_start < region_start_genome or abs_end > region_end_genome or abs_start >= abs_end:
                raise ValueError(
                    f"Fixed interval [{abs_start}, {abs_end}) for {region_id!r} is outside "
                    f"region genome interval [{region_start_genome}, {region_end_genome})."
                )
            windows.append((abs_start - region_start_genome, abs_end - region_start_genome))
        if max_windows is not None:
            windows = windows[:max_windows]
        return windows
    raise ValueError(f"Unsupported ISM scheme: {scheme}")


def run_region_ism(
    model,
    tokenizer,
    device: str,
    row: pd.Series,
    mask_windows: Sequence[Tuple[int, int]],
    score_groups: Mapping[str, Sequence[int]],
    delta_pairs: Mapping[str, Tuple[str, str]],
    *,
    fasta_path: str,
    sequence_length: int,
    inference_batch_size: int,
    scheme: str,
    region_index: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    chrom = str(row["chrom"])
    region_start = int(row["region_start"])
    region_end = int(row["region_end"])
    region_start_genome = region_start - 1
    region_id = str(row["id"])
    tss_start = int(row["start"])
    tss_end = int(row["end"])

    dataset = GenomeBigWigMaskISMDataset(
        fasta_path=fasta_path,
        chrom=chrom,
        region_start=region_start,
        region_end=region_end,
        mask_windows=list(mask_windows),
        sequence_length=sequence_length,
        tokenizer=tokenizer,
    )
    base_sequence = dataset.base_sequence

    original_tokens = tokenize_sequences(tokenizer, [base_sequence], sequence_length)
    with torch.no_grad():
        original_logits = predict_tokens(model, original_tokens, device)
    orig_overall, orig_tracks, orig_groups = compute_prediction_scores(original_logits, score_groups)
    orig_overall_score = float(orig_overall[0])
    orig_track_scores = {idx: float(orig_tracks[0, idx]) for idx in range(orig_tracks.shape[1])}
    orig_group_scores = {key: float(values[0]) for key, values in orig_groups.items()}
    orig_delta_scores = compute_delta_scores(orig_group_scores, delta_pairs)

    overall_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []

    loader = DataLoader(
        dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=0,
    )

    base_meta = {
        "region_index": region_index,
        "region_id": region_id,
        "chrom": chrom,
        "tss_start": tss_start,
        "tss_end": tss_end,
        "region_start": region_start,
        "region_end": region_end,
        "sequence_length": len(base_sequence),
        "scheme": scheme,
        "original_overall_score": orig_overall_score,
    }

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            logits = predict_tokens(model, tokens, device)
            masked_overall, masked_tracks, masked_groups = compute_prediction_scores(logits, score_groups)

            for batch_idx in range(tokens.shape[0]):
                window_idx = int(batch["window_idx"][batch_idx])
                mask_start = int(batch["mask_start"][batch_idx])
                mask_end = int(batch["mask_end"][batch_idx])
                # Convert region-relative mask offsets to absolute genome coordinates.
                mask_start_genome = region_start_genome + mask_start
                mask_end_genome = region_start_genome + mask_end
                row_meta = {
                    **base_meta,
                    "window_idx": window_idx,
                    "mask_start": mask_start,
                    "mask_end": mask_end,
                    "mask_start_genome": mask_start_genome,
                    "mask_end_genome": mask_end_genome,
                    "masked_overall_score": float(masked_overall[batch_idx]),
                    "importance_overall": orig_overall_score - float(masked_overall[batch_idx]),
                }
                overall_rows.append(row_meta)

                # Per-track scores
                masked_track_scores = {
                    track_idx: float(masked_tracks[batch_idx, track_idx])
                    for track_idx in range(masked_tracks.shape[1])
                }
                for track_idx, masked_score in masked_track_scores.items():
                    track_rows.append(
                        {
                            **base_meta,
                            "window_idx": window_idx,
                            "mask_start": mask_start,
                            "mask_end": mask_end,
                            "track_idx": track_idx,
                            "original_score": orig_track_scores[track_idx],
                            "masked_score": masked_score,
                            "importance": orig_track_scores[track_idx] - masked_score,
                        }
                    )

                # Per-group scores
                masked_group_scores = {
                    key: float(values[batch_idx]) for key, values in masked_groups.items()
                }
                masked_delta_scores = compute_delta_scores(masked_group_scores, delta_pairs)
                for group_key, masked_score in masked_group_scores.items():
                    group_rows.append(
                        {
                            **base_meta,
                            "window_idx": window_idx,
                            "mask_start": mask_start,
                            "mask_end": mask_end,
                            "group_key": group_key,
                            "original_score": orig_group_scores[group_key],
                            "masked_score": masked_score,
                            "importance": orig_group_scores[group_key] - masked_score,
                        }
                    )

                # Delta scores
                for delta_key in delta_pairs:
                    if delta_key not in orig_delta_scores or delta_key not in masked_delta_scores:
                        continue
                    delta_rows.append(
                        {
                            **base_meta,
                            "window_idx": window_idx,
                            "mask_start": mask_start,
                            "mask_end": mask_end,
                            "delta_key": delta_key,
                            "lhs_group": delta_pairs[delta_key][0],
                            "rhs_group": delta_pairs[delta_key][1],
                            "original_delta": orig_delta_scores[delta_key],
                            "masked_delta": masked_delta_scores[delta_key],
                            "importance_delta": orig_delta_scores[delta_key] - masked_delta_scores[delta_key],
                        }
                    )

    region_meta = {
        **base_meta,
        "num_windows": len(mask_windows),
    }
    return overall_rows, track_rows, group_rows, delta_rows, region_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    add_track_args(parser)
    parser.add_argument("--output-dir", default="results/ism")
    parser.add_argument(
        "--ism-scheme", choices=["sliding", "fixed"], required=True, 
        help="ISM scan scheme. sliding: slide a fixed-length window with step `s` inside `[scan_start, scan_end)`, replace window with `N`. fixed: use predefined intervals from `--mask-intervals-json`."
    )
    parser.add_argument("--scan-start", type=int, default=0, help="Sliding scheme: scan interval start within region.")
    parser.add_argument("--scan-end", type=int, default=None, help="Sliding scheme: scan interval end within region.")
    parser.add_argument("--window-size", type=int, default=200, help="Sliding scheme: N-mask window length.")
    parser.add_argument("--window-step", type=int, default=100, help="Sliding scheme: step between windows.")
    parser.add_argument(
        "--mask-intervals-json",
        default=None,
        help="Fixed scheme: JSON dict keyed by gene_bed id with [[genome_start, genome_end), ...].",
    )
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--limit-regions", type=int, default=None)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Limit mask windows per region for smoke tests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    expected_outputs = [
        output_dir / "ism_window_overall.csv",
        output_dir / "ism_window_per_track.csv",
        output_dir / "ism_window_per_group.csv",
        output_dir / "ism_window_delta.csv",
    ]
    if not args.overwrite and all(path.exists() for path in expected_outputs):
        raise FileExistsError(f"ISM outputs already exist in {output_dir}; pass --overwrite to replace them.")

    # project config
    config = load_project_config(args.config)
    design = resolve_track_design(config, args.track_map)
    baseline, infection = resolve_atac_timepoints(
        design,
        baseline_timepoint=args.baseline_timepoint,
        infection_timepoints=args.infection_timepoints,
    )
    score_groups = build_score_groups(design)
    delta_pairs = build_delta_pairs(design, baseline, infection)
    sequence_length = int(config["sequence_length"])
    scan_end = int(args.scan_end) if args.scan_end is not None else sequence_length

    # model checks
    fixed_intervals = None
    if args.ism_scheme == "fixed":
        if not args.mask_intervals_json:
            raise ValueError("Fixed ISM scheme requires --mask-intervals-json.")
        fixed_intervals = load_mask_intervals_json(args.mask_intervals_json)
    elif args.ism_scheme == "sliding":
        if args.scan_start < 0 or scan_end > sequence_length or args.scan_start >= scan_end:
            raise ValueError(
                f"Sliding scan interval must satisfy 0 <= scan_start < scan_end <= {sequence_length}, "
                f"got [{args.scan_start}, {scan_end})."
            )
        if args.window_size <= 0:
            raise ValueError(f"Sliding window size must be positive, got {args.window_size}.")
        if args.window_step <= 0:
            raise ValueError(f"Sliding window step must be positive, got {args.window_step}.")
        if args.max_windows is not None and args.max_windows <= 0:
            raise ValueError(f"Sliding max windows must be positive, got {args.max_windows}.")
    else:
        raise ValueError(f"Unsupported ISM scheme: {args.ism_scheme}")

    # model loading
    model, tokenizer, device = load_model_for_inference(
        config,
        args.ckpt,
        args.device,
        compile_model=args.compile_model,
        strict=args.strict,
    )
    regions = load_gene_regions_from_config(config, limit=args.limit_regions)

    overall_rows: List[Dict[str, Any]] = []
    track_rows: List[Dict[str, Any]] = []
    group_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    region_metadata: List[Dict[str, Any]] = []

    for region_index, row in tqdm(regions.iterrows(), total=len(regions), desc="ISM regions"):
        region_id = str(row["id"])
        seq_len = int(row["region_end"]) - int(row["region_start"])
        region_start_genome = int(row["region_start"]) - 1
        mask_windows = resolve_mask_windows(
            args.ism_scheme,
            seq_length=seq_len,
            region_start_genome=region_start_genome,
            scan_start=args.scan_start,
            scan_end=min(scan_end, seq_len),
            window_size=args.window_size,
            window_step=args.window_step,
            max_windows=args.max_windows,
            region_id=region_id,
            fixed_intervals=fixed_intervals,
        )
        if not mask_windows:
            continue

        region_overall, region_track, region_group, region_delta, region_meta = run_region_ism(
            model,
            tokenizer,
            device,
            row,
            mask_windows,
            score_groups,
            delta_pairs,
            fasta_path=str(config["fasta_path"]),
            sequence_length=sequence_length,
            inference_batch_size=args.inference_batch_size,
            scheme=args.ism_scheme,
            region_index=int(region_index),
        )
        overall_rows.extend(region_overall)
        track_rows.extend(region_track)
        group_rows.extend(region_group)
        delta_rows.extend(region_delta)
        region_metadata.append(region_meta)

    pd.DataFrame(overall_rows).to_csv(output_dir / "ism_window_overall.csv", index=False)
    pd.DataFrame(track_rows).to_csv(output_dir / "ism_window_per_track.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "ism_window_per_group.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(output_dir / "ism_window_delta.csv", index=False)
    pd.DataFrame(region_metadata).to_csv(output_dir / "region_metadata.csv", index=False)
    save_json(
        {
            "track_design": design,
            "score_groups": score_groups,
            "delta_pairs": {k: list(v) for k, v in delta_pairs.items()},
            "atac_baseline_timepoint": baseline,
            "atac_infection_timepoints": infection,
            "ism_scheme": args.ism_scheme,
            "scan_start": args.scan_start,
            "scan_end": scan_end,
            "window_size": args.window_size,
            "window_step": args.window_step,
            "mask_intervals_json": args.mask_intervals_json,
            "inference_batch_size": args.inference_batch_size,
            "max_windows": args.max_windows,
            "sequence_length": sequence_length,
        },
        output_dir / "ism_parameters.json",
    )
    print(f"Saved ISM outputs to {output_dir}")


if __name__ == "__main__":
    main()
