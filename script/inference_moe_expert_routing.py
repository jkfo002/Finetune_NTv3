#!/usr/bin/env python3
"""Batch inference for per-nucleotide MoE expert routing visualization."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyfaidx
import torch
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.analysis import plot_moe_expert_routing
from model.dataset import GenomeBigWigDataset
from model.moe import HFModelWithMoE_Infer, load_moe_config
from model.utils import (
    init_config,
    init_moe_model,
    load_Data,
    load_ckpt_with_compile,
    load_config,
    transform_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize MoE expert routing (topk_idx) at nucleotide resolution.",
    )
    parser.add_argument("--regions-bed", required=True, help="Bed file with regions to infer.")
    parser.add_argument("--config", default="fineturn_my_moe.toml", help="MoE TOML config when training.")
    parser.add_argument("--ckpt", required=True, help="Lightning checkpoint or .pth path.")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs.")
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers.")
    parser.add_argument("--limit-regions", type=int, default=None, help="Limit number of regions.")
    parser.add_argument("--region-index", type=int, default=None, help="Run a single bed row index.")
    parser.add_argument("--downsample", type=int, default=100, help="Plot downsample factor.")
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG generation.")
    parser.add_argument("--compile-model", action="store_true", help="torch.compile when loading ckpt.")
    parser.add_argument("--strict", action="store_true", help="Strict checkpoint loading.")
    return parser.parse_args()


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def crop_center_sequence(seq: str, keep_target_center_fraction: float) -> tuple[str, int]:
    seq_len = len(seq)
    target_offset = int(seq_len * (1 - keep_target_center_fraction) // 2)
    target_length = seq_len - 2 * target_offset
    return seq[target_offset : target_offset + target_length], target_offset


def region_prefix(chrom: str, region_start: int, region_end: int) -> str:
    chrom_safe = str(chrom).replace("/", "_")
    return f"{chrom_safe}_{region_start}_{region_end}"


def fetch_region_sequence(
    fasta_path: str,
    chrom: str,
    region_start: int,
    region_end: int,
) -> str:
    faidx = pyfaidx.Fasta(fasta_path)
    seq = str(faidx[chrom][region_start - 1 : region_end - 1]).upper()
    faidx.close()
    return seq


def build_routing_table(
    topk_idx: np.ndarray,
    topk_probs: np.ndarray,
    bases: str,
    region_start: int,
    crop_offset: int,
) -> pd.DataFrame:
    seq_len, top_k = topk_idx.shape
    if len(bases) != seq_len:
        raise ValueError(
            f"Sequence length ({len(bases)}) != topk_idx length ({seq_len})."
        )

    data: dict[str, Any] = {
        "local_idx": np.arange(seq_len, dtype=np.int64),
        "genomic_pos": region_start + crop_offset + np.arange(seq_len, dtype=np.int64),
        "base": list(bases),
    }

    for rank in range(top_k):
        data[f"top{rank + 1}_idx"] = topk_idx[:, rank].astype(np.int64)
        data[f"top{rank + 1}_prob"] = topk_probs[:, rank].astype(np.float64)

    return pd.DataFrame(data)


def _num_experts_from_routing(
    topk_idx: np.ndarray,
    router_probs: np.ndarray | None = None,
) -> int:
    if router_probs is not None:
        if router_probs.ndim == 2:
            return int(router_probs.shape[1])
        return int(router_probs.shape[0])
    return int(topk_idx.max()) + 1


def export_region_outputs(
    output_dir: Path,
    region_id: str,
    topk_idx: np.ndarray,
    topk_probs: np.ndarray,
    routing_df: pd.DataFrame,
    *,
    router_probs: np.ndarray | None = None,
    plot: bool = True,
    downsample: int = 100,
) -> dict[str, Any]:
    region_dir = output_dir / region_id
    region_dir.mkdir(parents=True, exist_ok=True)

    np.save(region_dir / f"{region_id}_topk_idx.npy", topk_idx.astype(np.int32))
    np.save(region_dir / f"{region_id}_topk_probs.npy", topk_probs.astype(np.float32))
    routing_df.to_csv(region_dir / f"{region_id}_routing.tsv", sep="\t", index=False)

    if router_probs is not None:
        np.save(region_dir / f"{region_id}_router_probs.npy", router_probs.astype(np.float32))

    if plot:
        num_experts = _num_experts_from_routing(topk_idx, router_probs)
        expert_labels = [str(i) for i in range(num_experts)]
        plot_moe_expert_routing(
            topk_idx,
            expert_labels,
            topk_probs=topk_probs,
            router_probs=router_probs,
            region_label=region_id,
            downsample=downsample,
            save_path=str(region_dir / f"{region_id}_expert_routing.png"),
        )

    top1_counts = Counter(int(x) for x in topk_idx[:, 0])
    return {
        "region_id": region_id,
        "seq_len": int(topk_idx.shape[0]),
        "top_k": int(topk_idx.shape[1]),
        "top1_expert_counts": {int(k): v for k, v in sorted(top1_counts.items())},
    }


def main() -> None:
    args = parse_args()
    config = init_config(load_config(args.config))
    if config.get("moe_config_path") is not None:
        with open(config["moe_config_path"], "r") as f:
            moe_config = json.load(f)
    else:
        raise ValueError("MoE config not found in config file.")

    # load model
    device = resolve_device(args.device)
    model, tokenizer = init_moe_model(config, HFModelWithMoE_Infer)
    model = load_ckpt_with_compile(
        model,
        args.ckpt,
        device,
        compile=args.compile_model,
        strict=args.strict,
    )
    model = model.to(device)
    model.eval()

    # load data
    if args.regions_bed is not None:
        gene_bed_path = args.regions_bed
    else:
        print(f"Using default gene bed: {os.path.join(config['training_data_dir'], config['gene_bed'])}")
        gene_bed_path = os.path.join(config["training_data_dir"], config["gene_bed"])
    gene_bed = pd.read_csv(
        gene_bed_path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "id", "type"],
    )
    faidx = pyfaidx.Fasta(config["fasta_path"])
    infer_bed = load_Data(gene_bed, faidx, config["TSS_up"], config["TSS_down"])
    faidx.close()

    if args.limit_regions is not None:
        infer_bed = infer_bed.iloc[: args.limit_regions].reset_index(drop=True)

    # build dataset and dataloader
    num_workers = args.num_workers if args.num_workers is not None else config.get("num_workers", 4)
    infer_dataset = GenomeBigWigDataset(
        fasta_path=config["fasta_path"],
        bigwig_path_list=[
            os.path.join(config["training_data_dir"], f) for f in config["bigwig_files"]
        ],
        chrom_regions=infer_bed,
        sequence_length=config["sequence_length"],
        tokenizer=tokenizer,
        transform_fn=transform_fn,
        keep_target_center_fraction=config["keep_target_center_fraction"],
        track_label_list=config.get("track_label_list"),
        num_samples=len(infer_bed),
    )
    infer_dataloader = DataLoader(
        infer_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # build output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_fraction = config["keep_target_center_fraction"]
    region_summaries: list[dict[str, Any]] = []
    global_top1 = Counter()

    for batch_idx, batch in enumerate(tqdm(infer_dataloader, desc="MoE routing inference")):
        tokens = batch["tokens"].to(device)
        with autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            with torch.no_grad():
                outputs = model(tokens=tokens, mode="infer", return_dict=True)

        topk_idx_batch = outputs["topk_idx"].detach().cpu().numpy()
        topk_probs_batch = outputs["topk_probs"].detach().cpu().numpy()
        router_probs_batch = outputs.get("router_probs")
        if router_probs_batch is not None:
            router_probs_batch = router_probs_batch.detach().cpu().numpy()

        batch_size = topk_idx_batch.shape[0]
        for i in range(batch_size):
            dataset_idx = batch_idx * args.batch_size + i
            row = infer_bed.iloc[dataset_idx]
            chrom = row["chrom"]
            region_start = int(row["region_start"])
            region_end = int(row["region_end"])
            region_id = region_prefix(chrom, region_start, region_end)

            raw_seq = fetch_region_sequence(
                config["fasta_path"],
                chrom,
                region_start,
                region_end,
            )
            topk_idx = topk_idx_batch[i]
            topk_probs = topk_probs_batch[i]
            router_probs = router_probs_batch[i] if router_probs_batch is not None else None

            cropped_seq, crop_offset = crop_center_sequence(raw_seq, keep_fraction)
            if len(cropped_seq) != topk_idx.shape[0]:
                """长度不等的时候的处理"""
                if len(cropped_seq) > topk_idx.shape[0]:
                    cropped_seq = cropped_seq[: topk_idx.shape[0]]
                else:
                    cropped_seq = cropped_seq + ("N" * (topk_idx.shape[0] - len(cropped_seq)))

            routing_df = build_routing_table(
                topk_idx,
                topk_probs,
                cropped_seq,
                region_start,
                crop_offset
            )

            summary = export_region_outputs(
                output_dir,
                region_id,
                topk_idx,
                topk_probs,
                routing_df,
                router_probs=router_probs,
                plot=not args.no_plot,
                downsample=args.downsample,
            )
            region_summaries.append(summary)
            for expert_id, count in summary["top1_expert_counts"].items():
                global_top1[expert_id] += count

    summary_payload = {
        "num_regions": len(region_summaries),
        "top_k": moe_config.get("top_k"),
        "num_experts": moe_config.get("num_experts"),
        "routing": moe_config.get("routing"),
        "keep_target_center_fraction": keep_fraction,
        "global_top1_expert_counts": dict(global_top1),
        "regions": region_summaries,
    }
    with (output_dir / "summary.json").open("w") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)

    print(f"Saved outputs for {len(region_summaries)} regions to {output_dir}")


if __name__ == "__main__":
    main()
