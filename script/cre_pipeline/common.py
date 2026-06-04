#!/usr/bin/env python3
"""Shared helpers for the CRE inference pipeline.

The scripts in this folder are intentionally independent from the existing
ad-hoc inference examples. They reuse the project model/dataset APIs, while
keeping all runtime choices configurable from CLI arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import toml
import torch
from torch.amp.autocast_mode import autocast


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.dataset import GenomeBigWigDataset
from model.head import HFModelWithHead_Infer
from model.utils import (
    init_model,
    init_moe_model,
    load_Data,
    load_ckpt_with_compile,
    load_config,
    init_config,
    transform_fn,
)


DNA_ALPHABET = np.array(list("ACGT"))


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/fineturn_my.toml", help="Project TOML config.")
    parser.add_argument("--ckpt", required=True, help="Model checkpoint path.")
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda if available.")
    parser.add_argument("--compile-model", action="store_true", help="Compile model when loading checkpoint.")
    parser.add_argument("--strict", action="store_true", help="Strict checkpoint loading.")


def add_track_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track-map",
        default=None,
        help="JSON/TOML mapping of timepoint to track indices. Defaults to ATAC tracks inferred from config.",
    )
    parser.add_argument("--baseline-timepoint", default=None, help="Baseline timepoint for delta importance.")
    parser.add_argument(
        "--infection-timepoints",
        nargs="*",
        default=None,
        help="Infection timepoints. Defaults to all non-baseline timepoints in the inferred mapping.",
    )


def add_dataloader_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit-regions", type=int, default=None, help="Limit regions for smoke tests.")


def load_project_config(config_path: str) -> Dict[str, Any]:
    config = load_config(config_path)
    config = init_config(config)
    return config


def resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_for_inference(
    config: Mapping[str, Any],
    ckpt_path: str,
    device: Optional[str] = None,
    *,
    compile_model: bool = False,
    strict: bool = False,
    saliency: bool = False,
):
    config_dict = dict(config)
    if config_dict.get("use_moe", False):
        from model.moe import HFModelWithMoE_Infer  # noqa: WPS433

        model, tokenizer = init_moe_model(config_dict, HFModelWithMoE_Infer)
    else:
        model, tokenizer = init_model(config_dict, HFModelWithHead_Infer)
    device = resolve_device(device)
    model = load_ckpt_with_compile(
        model,
        ckpt_path,
        device,
        compile=compile_model,
        strict=strict,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_mapping_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        with path.open() as handle:
            raw = json.load(handle)
    else:
        raw = toml.load(path)
    return raw


def infer_atac_timepoint_tracks(config: Mapping[str, Any]) -> Dict[str, List[int]]:
    bigwig_files = list(config.get("bigwig_files", []))
    track_labels = config.get("track_label_list")
    if track_labels is None:
        atac_indices = list(range(len(bigwig_files)))
    else:
        atac_indices = [idx for idx, label in enumerate(track_labels) if int(label) == 1]

    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx in atac_indices:
        if idx >= len(bigwig_files):
            continue
        name = os.path.basename(str(bigwig_files[idx]))
        match = re.match(r"([^._-]+)", name)
        timepoint = match.group(1) if match else f"track{idx}"
        grouped[timepoint].append(idx)

    if not grouped:
        raise ValueError("No ATAC tracks could be inferred from config['track_label_list'].")

    def sort_key(item: Tuple[str, List[int]]) -> Tuple[int, Any]:
        key = item[0]
        return (0, int(key)) if key.isdigit() else (1, key)

    return dict(sorted(grouped.items(), key=sort_key))


def infer_rna_design(config: Mapping[str, Any]) -> Dict[str, Dict[str, List[int]]]:
    bigwig_files = list(config.get("bigwig_files", []))
    track_labels = config.get("track_label_list")
    if track_labels is None:
        rna_indices = list(range(len(bigwig_files)))
    else:
        rna_indices = [idx for idx, label in enumerate(track_labels) if int(label) == 0]

    design: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    for idx in rna_indices:
        if idx >= len(bigwig_files):
            continue
        name = os.path.basename(str(bigwig_files[idx]))
        root = re.split(r"[-_.]", name)[0]
        match = re.match(r"([A-Za-z]+)(\d+)?", root)
        if not match:
            continue
        prefix = match.group(1)
        number = match.group(2)
        if number:
            if prefix.upper() == "C":
                design[number]["CK"].append(idx)
            elif prefix.upper() == "T":
                design[number]["infect"].append(idx)
            else:
                design[prefix.lower()]["CK"].append(idx)
        else:
            design[prefix.lower()]["CK"].append(idx)

    def sort_key(item: Tuple[str, Dict[str, List[int]]]) -> Tuple[int, Any]:
        key = item[0]
        return (0, int(key)) if key.isdigit() else (1, key)

    return {k: dict(v) for k, v in sorted(design.items(), key=sort_key)}


def normalize_track_design(raw: Mapping[str, Any]) -> Dict[str, Any]:
    if "RNA" in raw or "ATAC" in raw:
        rna_raw = raw.get("RNA", {})
        atac_raw = raw.get("ATAC", {})
    else:
        # Backward compatible flat mapping -> ATAC only
        rna_raw = {}
        atac_raw = raw

    rna: Dict[str, Dict[str, List[int]]] = {}
    for timepoint, groups in rna_raw.items():
        if isinstance(groups, list):
            rna[str(timepoint)] = {"CK": [int(x) for x in groups]}
            continue
        if not isinstance(groups, Mapping):
            raise ValueError(f"RNA mapping for {timepoint!r} must be list or dict, got {type(groups).__name__}.")
        normalized_groups: Dict[str, List[int]] = {}
        for group_name, indices in groups.items():
            if isinstance(indices, list):
                normalized_groups[str(group_name)] = [int(x) for x in indices]
            else:
                raise ValueError(
                    f"RNA mapping {timepoint!r}/{group_name!r} must be list of indices, got {type(indices).__name__}."
                )
        rna[str(timepoint)] = normalized_groups

    atac: Dict[str, List[int]] = {}
    for timepoint, indices in atac_raw.items():
        if not isinstance(indices, list):
            raise ValueError(f"ATAC mapping for {timepoint!r} must be list, got {type(indices).__name__}.")
        atac[str(timepoint)] = [int(x) for x in indices]

    return {"RNA": rna, "ATAC": atac}


def resolve_track_design(
    config: Mapping[str, Any],
    track_map_path: Optional[str] = None,
) -> Dict[str, Any]:
    if track_map_path:
        raw = load_mapping_file(track_map_path)
        design = normalize_track_design(raw)
    else:
        design = {"RNA": infer_rna_design(config), "ATAC": infer_atac_timepoint_tracks(config)}
    validate_track_design(design, int(config["num_tracks"]))
    return design


def validate_track_map(mapping: Mapping[str, Sequence[int]], num_tracks: int) -> None:
    for timepoint, indices in mapping.items():
        if not indices:
            raise ValueError(f"Track mapping for {timepoint!r} is empty.")
        for idx in indices:
            if idx < 0 or idx >= num_tracks:
                raise ValueError(f"Track index {idx} for {timepoint!r} outside [0, {num_tracks}).")


def validate_track_design(design: Mapping[str, Any], num_tracks: int) -> None:
    atac = design.get("ATAC", {})
    rna = design.get("RNA", {})
    validate_track_map(atac, num_tracks)
    for timepoint, groups in rna.items():
        if not isinstance(groups, Mapping):
            raise ValueError(f"RNA groups for {timepoint!r} must be mapping.")
        validate_track_map({f"{timepoint}:{k}": v for k, v in groups.items()}, num_tracks)


def resolve_timepoints(
    mapping: Mapping[str, Sequence[int]],
    baseline_timepoint: Optional[str] = None,
    infection_timepoints: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    keys = list(mapping.keys())
    if not keys:
        raise ValueError("Empty track mapping.")
    baseline = baseline_timepoint or keys[0]
    if baseline not in mapping:
        raise ValueError(f"Baseline timepoint {baseline!r} not found in track mapping: {keys}")
    infection = list(infection_timepoints) if infection_timepoints else [tp for tp in keys if tp != baseline]
    missing = [tp for tp in infection if tp not in mapping]
    if missing:
        raise ValueError(f"Infection timepoints not found in track mapping: {missing}")
    return baseline, infection


def resolve_atac_timepoints(
    design: Mapping[str, Any],
    baseline_timepoint: Optional[str] = None,
    infection_timepoints: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    atac = design.get("ATAC", {})
    return resolve_timepoints(atac, baseline_timepoint=baseline_timepoint, infection_timepoints=infection_timepoints)


def build_score_groups(design: Mapping[str, Any]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for timepoint, indices in design.get("ATAC", {}).items():
        groups[f"ATAC::{timepoint}"] = list(indices)
    for timepoint, conditions in design.get("RNA", {}).items():
        for condition, indices in conditions.items():
            groups[f"RNA::{timepoint}::{condition}"] = list(indices)
    return groups


def build_delta_pairs(
    design: Mapping[str, Any],
    atac_baseline: str,
    atac_infection: Sequence[str],
) -> Dict[str, Tuple[str, str]]:
    pairs: Dict[str, Tuple[str, str]] = {}
    for tp in atac_infection:
        pairs[f"ATAC::{tp}"] = (f"ATAC::{tp}", f"ATAC::{atac_baseline}")
    for tp, conditions in design.get("RNA", {}).items():
        if "infect" in conditions and "CK" in conditions:
            pairs[f"RNA::{tp}"] = (f"RNA::{tp}::infect", f"RNA::{tp}::CK")
    return pairs


def load_gene_regions_from_config(config: Mapping[str, Any], limit: Optional[int] = None) -> pd.DataFrame:
    import pyfaidx

    gene_bed = str(config["gene_bed"])
    if not os.path.isabs(gene_bed):
        gene_bed = os.path.join(str(config["training_data_dir"]), gene_bed)
    df = pd.read_csv(gene_bed, sep="\t", header=None, names=["chrom", "start", "end", "id", "type"])
    faidx = pyfaidx.Fasta(str(config["fasta_path"]))
    try:
        regions = load_Data(df, faidx, int(config["TSS_up"]), int(config["TSS_down"]))
    finally:
        faidx.close()
    if limit is not None:
        regions = regions.iloc[:limit].copy()
    return regions.reset_index(drop=True)


def load_bed_regions(
    bed_path: str | Path,
    *,
    limit: Optional[int] = None,
    add_one_for_dataset: bool = False,
) -> pd.DataFrame:
    cols = ["chrom", "start", "end", "id", "score", "strand"]
    bed = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    bed = bed.iloc[:, : min(bed.shape[1], len(cols))]
    bed.columns = cols[: bed.shape[1]]
    if "id" not in bed:
        bed["id"] = [f"region_{i:06d}" for i in range(len(bed))]
    if "score" not in bed:
        bed["score"] = "."
    bed["type"] = bed["score"].astype(str)
    offset = 1 if add_one_for_dataset else 0
    bed["region_start"] = bed["start"].astype(int) + offset
    bed["region_end"] = bed["end"].astype(int) + offset
    out = bed[["chrom", "start", "end", "id", "type", "region_start", "region_end"]].copy()
    if limit is not None:
        out = out.iloc[:limit].copy()
    return out.reset_index(drop=True)


def make_dataloader(
    config: Mapping[str, Any],
    tokenizer,
    regions: pd.DataFrame,
    *,
    batch_size: int,
    num_workers: Optional[int] = None,
):
    from torch.utils.data import DataLoader

    bigwig_paths = [
        path if os.path.isabs(path) else os.path.join(str(config["training_data_dir"]), path)
        for path in config["bigwig_files"]
    ]
    dataset = GenomeBigWigDataset(
        fasta_path=str(config["fasta_path"]),
        bigwig_path_list=bigwig_paths,
        chrom_regions=regions,
        sequence_length=int(config["sequence_length"]),
        tokenizer=tokenizer,
        transform_fn=transform_fn,
        keep_target_center_fraction=float(config["keep_target_center_fraction"]),
        track_label_list=config.get("track_label_list"),
        num_samples=len(regions),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("num_workers", 0) if num_workers is None else num_workers),
    )


@torch.no_grad()
def predict_tokens(
    model,
    tokens: torch.Tensor,
    device: str,
    *,
    use_bfloat16: bool = True,
    return_outputs: bool = False,
) -> torch.Tensor | Dict[str, Any]:
    """
    return model all outputs when return_outputs=True, 
    otherwise return bigwig tracks logits.
    """
    tokens = tokens.to(device)
    if device.startswith("cuda") and use_bfloat16:
        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(tokens, mode="infer", return_dict=True)
    else:
        outputs = model(tokens, mode="infer", return_dict=True)
    if return_outputs:
        return outputs
    return outputs["bigwig_tracks_logits"]


def track_mean(logits: torch.Tensor, track_indices: Sequence[int]) -> torch.Tensor:
    return logits[..., list(track_indices)].mean(dim=-1)


def sanitize_sequence(sequence: str) -> str:
    sequence = sequence.upper()
    return re.sub(r"[^ACGT]", "N", sequence)


def mutate_window(
    sequence: str,
    start: int,
    end: int,
    *,
    strategy: str,
    rng: np.random.Generator,
) -> str:
    start = max(0, start)
    end = min(len(sequence), end)
    if start >= end:
        return sequence
    chars = list(sequence)
    window = chars[start:end]
    if strategy == "shuffle":
        rng.shuffle(window)
        chars[start:end] = window
    elif strategy == "mask_n":
        chars[start:end] = ["N"] * (end - start)
    elif strategy == "random":
        chars[start:end] = rng.choice(DNA_ALPHABET, size=end - start).tolist()
    else:
        raise ValueError(f"Unsupported mutation strategy: {strategy}")
    return "".join(chars)


def tokenize_sequences(tokenizer, sequences: Sequence[str], sequence_length: int) -> torch.Tensor:
    batch = tokenizer(
        list(sequences),
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )
    return batch["input_ids"]


def read_fasta_sequence(fasta, chrom: str, start: int, end: int) -> str:
    start = max(0, int(start))
    end = max(start, int(end))
    return sanitize_sequence(fasta[chrom][start:end].seq)


def flatten_numeric(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def minmax(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = vals.min()
    hi = vals.max()
    if hi <= lo:
        return pd.Series(np.zeros(len(vals)), index=series.index)
    return (vals - lo) / (hi - lo)


def write_bed(df: pd.DataFrame, path: str | Path, columns: Sequence[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.loc[:, list(columns)].to_csv(path, sep="\t", header=False, index=False)


def parse_weights(raw: Optional[str], default: Mapping[str, float]) -> Dict[str, float]:
    if raw is None:
        return dict(default)
    if os.path.exists(raw):
        path = Path(raw)
        if path.suffix.lower() == ".json":
            with path.open() as handle:
                data = json.load(handle)
        else:
            data = toml.load(path)
        return {str(k): float(v) for k, v in data.items()}
    weights = {}
    for item in raw.split(","):
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value)
    return weights


def iter_chunks(items: Sequence[Any], chunk_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def sliding_mask_windows(
    scan_start: int,
    scan_end: int,
    window_size: int,
    step: int,
    *,
    max_windows: Optional[int] = None,
) -> List[Tuple[int, int]]:
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive.")
    if scan_start < 0 or scan_end <= scan_start:
        raise ValueError(f"Invalid scan interval [{scan_start}, {scan_end}).")
    windows: List[Tuple[int, int]] = []
    start = scan_start
    while start + window_size <= scan_end:
        windows.append((start, start + window_size))
        if max_windows is not None and len(windows) >= max_windows:
            break
        start += step
    return windows


def load_mask_intervals_json(path: str | Path) -> Dict[str, List[Tuple[int, int]]]:
    raw = load_mapping_file(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Mask interval JSON must be a dict keyed by region id, got {type(raw).__name__}.")
    parsed: Dict[str, List[Tuple[int, int]]] = {}
    for region_id, intervals in raw.items():
        if not isinstance(intervals, list):
            raise ValueError(f"Intervals for {region_id!r} must be a list, got {type(intervals).__name__}.")
        region_windows: List[Tuple[int, int]] = []
        for item in intervals:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(f"Interval for {region_id!r} must be [start, end), got {item!r}.")
            start, end = int(item[0]), int(item[1])
            if start >= end:
                raise ValueError(f"Interval for {region_id!r} must satisfy start < end, got [{start}, {end}).")
            region_windows.append((start, end))
        parsed[str(region_id)] = region_windows
    return parsed
