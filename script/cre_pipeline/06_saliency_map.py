#!/usr/bin/env python3
"""Step 6: compute gradient saliency maps for CRE regions (MoE-compatible)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyfaidx
from tqdm import tqdm

from common import (
    add_model_args,
    add_track_args,
    build_score_groups,
    ensure_dir,
    load_bed_regions,
    load_gene_regions_from_config,
    load_model_for_inference,
    load_project_config,
    resolve_track_design,
    sanitize_sequence,
    save_json,
)
from model.head import SaliencyComputer


def parse_track_indices(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    out: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out if out else None


def parse_region(raw: Optional[str]) -> Optional[Tuple[int, int]]:
    if not raw:
        return None
    if ":" not in raw:
        raise ValueError("--saliency-region must be formatted as start:end")
    start_str, end_str = raw.split(":", 1)
    start, end = int(start_str), int(end_str)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid saliency region [{start}, {end}).")
    return start, end


def resolve_saliency_tracks(
    config: Dict,
    track_map: Optional[str],
    score_group: Optional[str],
    track_indices_raw: Optional[str],
) -> Optional[List[int]]:
    if score_group and track_indices_raw:
        raise ValueError("Use either --score-group or --track-indices, not both.")
    if score_group:
        design = resolve_track_design(config, track_map)
        groups = build_score_groups(design)
        if score_group not in groups:
            raise ValueError(f"Unknown score group {score_group!r}. Available: {sorted(groups)}")
        return list(groups[score_group])
    return parse_track_indices(track_indices_raw)


def extract_region_sequence(fasta, chrom: str, region_start: int, region_end: int) -> str:
    # Keep coordinate semantics consistent with GenomeBigWigDataset (1-based region fields).
    seq = fasta[chrom][region_start - 1 : region_end - 1].seq
    return sanitize_sequence(str(seq))


def flush_chunk(
    output_dir: Path,
    chunk_id: int,
    gradients: List[np.ndarray],
    onehots: List[np.ndarray],
    rows: List[Dict],
    save_onehot: bool,
) -> None:
    if not gradients:
        return
    grad_concat = np.concatenate(gradients, axis=0)
    np.save(output_dir / f"gradients_ordered_{chunk_id}.npy", grad_concat)
    if save_onehot:
        onehot_concat = np.concatenate(onehots, axis=0)
        np.save(output_dir / f"embeddings_ordered_{chunk_id}.npy", onehot_concat)
    pd.DataFrame(rows).to_csv(output_dir / f"saliency_chunk_{chunk_id}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    add_track_args(parser)
    parser.add_argument("--regions-bed", default=None, help="Optional BED regions. Defaults to config gene_bed.")
    parser.add_argument("--output-dir", default="results/saliency")
    parser.add_argument("--limit-regions", type=int, default=None)
    parser.add_argument(
        "--track-indices",
        default=None,
        help="Comma-separated track indices used for saliency score (e.g. 0,1,2).",
    )
    parser.add_argument(
        "--score-group",
        default=None,
        help="Use indices from a score group key (e.g. RNA::24::infect).",
    )
    parser.add_argument(
        "--saliency-region",
        default=None,
        help="Optional logits subregion as start:end before score aggregation.",
    )
    parser.add_argument("--chunk-size", type=int, default=8000, help="Regions per saved chunk.")
    parser.add_argument(
        "--save-onehot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one-hot embeddings alongside gradients.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    if not args.overwrite and any(output_dir.glob("gradients_ordered_*.npy")):
        raise FileExistsError(f"Saliency outputs already exist in {output_dir}; pass --overwrite to replace them.")

    config = load_project_config(args.config)
    selected_tracks = resolve_saliency_tracks(
        config,
        args.track_map,
        args.score_group,
        args.track_indices,
    )
    saliency_region = parse_region(args.saliency_region)

    model, tokenizer, device = load_model_for_inference(
        config,
        args.ckpt,
        args.device,
        compile_model=args.compile_model,
        strict=args.strict,
    )
    saliency_computer = SaliencyComputer(
        model=model,
        tokenizer=tokenizer,
        sequence_length=int(config["sequence_length"]),
        track_indices=selected_tracks,
        device=device,
        region=saliency_region,
    )

    if args.regions_bed:
        regions = load_bed_regions(args.regions_bed, limit=args.limit_regions, add_one_for_dataset=True)
    else:
        regions = load_gene_regions_from_config(config, limit=args.limit_regions)

    fasta = pyfaidx.Fasta(str(config["fasta_path"]))
    gradients_chunk: List[np.ndarray] = []
    onehots_chunk: List[np.ndarray] = []
    rows_chunk: List[Dict] = []
    row_cursor = 0
    chunk_id = 0

    try:
        for idx, row in tqdm(regions.iterrows(), total=len(regions), desc="Saliency regions"):
            chrom = str(row["chrom"])
            region_start = int(row["region_start"])
            region_end = int(row["region_end"])
            region_id = str(row.get("id", f"region_{idx:06d}"))
            sequence = extract_region_sequence(fasta, chrom, region_start, region_end)
            if not sequence:
                continue

            gradients, one_hots = saliency_computer.compute_saliency(sequence=sequence)
            # Keep compatibility with historical script channel convention.
            gradients_nt = gradients[:, 6:10]
            one_hots_nt = one_hots[:, 6:10][:, [0, 2, 3, 1]]

            gradients_chunk.append(gradients_nt)
            if args.save_onehot:
                onehots_chunk.append(one_hots_nt)

            rows_chunk.append(
                {
                    "chunk_id": chunk_id,
                    "region_index": int(idx),
                    "region_id": region_id,
                    "chrom": chrom,
                    "region_start": region_start,
                    "region_end": region_end,
                    "row_start": row_cursor,
                    "row_end": row_cursor + gradients_nt.shape[0],
                }
            )
            row_cursor += gradients_nt.shape[0]

            if len(gradients_chunk) >= args.chunk_size:
                flush_chunk(output_dir, chunk_id, gradients_chunk, onehots_chunk, rows_chunk, args.save_onehot)
                gradients_chunk = []
                onehots_chunk = []
                rows_chunk = []
                row_cursor = 0
                chunk_id += 1
    finally:
        fasta.close()

    flush_chunk(output_dir, chunk_id, gradients_chunk, onehots_chunk, rows_chunk, args.save_onehot)

    save_json(
        {
            "config": args.config,
            "ckpt": args.ckpt,
            "track_map": args.track_map,
            "score_group": args.score_group,
            "track_indices": selected_tracks,
            "saliency_region": list(saliency_region) if saliency_region else None,
            "sequence_length": int(config["sequence_length"]),
            "chunk_size": args.chunk_size,
            "save_onehot": bool(args.save_onehot),
            "num_regions": int(len(regions)),
            "output_dir": str(output_dir),
        },
        output_dir / "saliency_run_config.json",
    )
    print(f"Saved saliency outputs to {output_dir}")


if __name__ == "__main__":
    main()
