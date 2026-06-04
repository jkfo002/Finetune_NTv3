#!/usr/bin/env python3
"""Step 4: annotate CRE candidates and rank them for follow-up."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from common import ensure_dir, minmax, save_json, write_bed


DEFAULT_WEIGHTS = {
    "importance_score": 0.30,
    "motif_match": 0.20,
    "gene_foldchange": 0.25,
    "conservation": 0.15,
    "temporal_cluster": 0.10,
}


def parse_attrs(raw: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for part in str(raw).strip().split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
        elif " " in part:
            key, value = part.split(" ", 1)
            value = value.strip('"')
        else:
            continue
        attrs[key.strip()] = value.strip().strip('"')
    return attrs


def read_cre(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".bed":
        cols = ["chr", "start", "end", "cre_id", "importance_score", "cluster_id"]
        df = pd.read_csv(path, sep="\t", header=None, names=cols[: len(pd.read_csv(path, sep="\t", header=None, nrows=1).columns)])
        if "cre_id" not in df:
            df["cre_id"] = [f"cre_{i + 1:06d}" for i in range(len(df))]
        if "importance_score" not in df:
            df["importance_score"] = 0.0
        return df
    df = pd.read_csv(path)
    if "chr" not in df and "chrom" in df:
        df = df.rename(columns={"chrom": "chr"})
    if "cre_id" not in df:
        df["cre_id"] = [f"cre_{i + 1:06d}" for i in range(len(df))]
    return df


def read_gene_annotation(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".bed", ".tsv"}:
        cols = ["chr", "start", "end", "gene_id", "gene_name", "strand"]
        df = pd.read_csv(path, sep="\t", header=None, comment="#")
        df = df.iloc[:, : min(df.shape[1], len(cols))]
        df.columns = cols[: df.shape[1]]
        if "gene_name" not in df:
            df["gene_name"] = df["gene_id"]
        return df[["chr", "start", "end", "gene_id", "gene_name"]].copy()

    rows = []
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            chrom, _, feature, start, end, _, _, _, attrs_raw = fields
            if feature not in {"gene", "mRNA", "transcript"}:
                continue
            attrs = parse_attrs(attrs_raw)
            gene_id = attrs.get("ID") or attrs.get("gene_id") or attrs.get("Parent")
            gene_name = attrs.get("Name") or attrs.get("gene_name") or gene_id
            if not gene_id:
                continue
            rows.append(
                {
                    "chr": chrom,
                    "start": int(start) - 1,
                    "end": int(end),
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                }
            )
    return pd.DataFrame(rows)


def assign_target_genes(cre: pd.DataFrame, genes: pd.DataFrame, max_distance: int) -> pd.DataFrame:
    if genes.empty:
        return pd.DataFrame(
            {
                "cre_id": cre["cre_id"],
                "target_gene": "unknown",
                "gene_name": "unknown",
                "distance": np.nan,
                "assignment_method": "none",
            }
        )
    grouped = {chrom: group.reset_index(drop=True) for chrom, group in genes.groupby("chr")}
    records = []
    for _, row in cre.iterrows():
        chrom_genes = grouped.get(row["chr"])
        if chrom_genes is None or chrom_genes.empty:
            records.append(
                {
                    "cre_id": row["cre_id"],
                    "target_gene": "unknown",
                    "gene_name": "unknown",
                    "distance": np.nan,
                    "assignment_method": "none",
                }
            )
            continue
        center = (int(row["start"]) + int(row["end"])) / 2.0
        gene_centers = (chrom_genes["start"].astype(float) + chrom_genes["end"].astype(float)) / 2.0
        distances = (gene_centers - center).abs()
        best_idx = int(distances.idxmin())
        best = chrom_genes.loc[best_idx]
        distance = float(distances.loc[best_idx])
        records.append(
            {
                "cre_id": row["cre_id"],
                "target_gene": best["gene_id"] if distance <= max_distance else "unknown",
                "gene_name": best["gene_name"] if distance <= max_distance else "unknown",
                "distance": distance if distance <= max_distance else np.nan,
                "assignment_method": "nearest" if distance <= max_distance else "none",
            }
        )
    return pd.DataFrame(records)


def read_motif_positions(path: str | Path) -> pd.DataFrame:
    cols = ["chr", "start", "end", "motif_id", "score", "strand", "motif_name"]
    df = pd.read_csv(path, sep="\t", header=None, comment="#")
    df = df.iloc[:, : min(df.shape[1], len(cols))]
    df.columns = cols[: df.shape[1]]
    if "motif_id" not in df:
        df["motif_id"] = "motif"
    if "motif_name" not in df:
        df["motif_name"] = df["motif_id"]
    return df


def overlap_motifs(cre: pd.DataFrame, motifs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    grouped = {chrom: group for chrom, group in motifs.groupby("chr")}
    for _, c in cre.iterrows():
        chrom_motifs = grouped.get(c["chr"], pd.DataFrame())
        if chrom_motifs.empty:
            continue
        hits = chrom_motifs[
            (chrom_motifs["start"].astype(int) < int(c["end"]))
            & (chrom_motifs["end"].astype(int) > int(c["start"]))
        ]
        for _, hit in hits.iterrows():
            records.append(
                {
                    "chr": hit["chr"],
                    "start": int(hit["start"]),
                    "end": int(hit["end"]),
                    "cre_id": c["cre_id"],
                    "motif_id": hit["motif_id"],
                    "motif_name": hit["motif_name"],
                }
            )
    pos = pd.DataFrame(records)
    if pos.empty:
        summary = pd.DataFrame(columns=["cre_id", "motif_match", "motif_count"])
    else:
        summary = (
            pos.groupby("cre_id")
            .agg(
                motif_match=("motif_name", lambda x: ";".join(sorted(set(map(str, x))))),
                motif_count=("motif_name", "size"),
            )
            .reset_index()
        )
    return summary, pos


def motif_enrichment_from_positions(motif_positions: pd.DataFrame, cre_count: int) -> pd.DataFrame:
    if motif_positions.empty:
        return pd.DataFrame(columns=["motif_name", "matches", "expected", "p_value", "fold_enrichment"])
    counts = motif_positions.groupby("motif_name").size().reset_index(name="matches")
    counts["expected"] = max(1.0, counts["matches"].mean())
    counts["p_value"] = np.nan
    counts["fold_enrichment"] = counts["matches"] / counts["expected"]
    counts["cre_count"] = cre_count
    return counts.sort_values("matches", ascending=False)


def merge_optional_table(cre: pd.DataFrame, path: str | None, key: str = "cre_id") -> pd.DataFrame:
    if not path:
        return cre
    table = pd.read_csv(path)
    if key not in table.columns:
        raise ValueError(f"{path} must contain {key!r}.")
    return cre.merge(table, on=key, how="left")


def attach_expression(cre: pd.DataFrame, expression_path: str | None) -> pd.DataFrame:
    if not expression_path:
        cre["gene_foldchange"] = cre.get("gene_foldchange", 0.0)
        return cre
    expr = pd.read_csv(expression_path)
    if "target_gene" not in expr.columns:
        if "gene_id" in expr.columns:
            expr = expr.rename(columns={"gene_id": "target_gene"})
        else:
            raise ValueError("Expression table must contain target_gene or gene_id.")
    if "gene_foldchange" not in expr.columns:
        fold_cols = [c for c in expr.columns if re.search("fold|log2fc|fc", c, re.I)]
        if not fold_cols:
            raise ValueError("Expression table must contain gene_foldchange or a fold-change-like column.")
        expr = expr.rename(columns={fold_cols[0]: "gene_foldchange"})
    return cre.merge(expr[["target_gene", "gene_foldchange"]], on="target_gene", how="left")


def compute_priority_score(cre: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = cre.copy()
    out["motif_match"] = out.get("motif_match", pd.Series("", index=out.index, dtype=object)).fillna("")
    motif_count = out["motif_count"] if "motif_count" in out else pd.Series(0.0, index=out.index)
    gene_foldchange = out["gene_foldchange"] if "gene_foldchange" in out else pd.Series(0.0, index=out.index)
    cluster_importance = (
        out["cluster_importance"]
        if "cluster_importance" in out
        else out.get("importance_score", pd.Series(0.0, index=out.index))
    )
    out["motif_count"] = pd.to_numeric(motif_count, errors="coerce").fillna(0.0)
    out["gene_foldchange"] = pd.to_numeric(gene_foldchange, errors="coerce").fillna(0.0)
    out["cluster_importance"] = pd.to_numeric(cluster_importance, errors="coerce").fillna(0.0)
    if "conservation_rank" not in out:
        out["conservation_rank"] = "unknown"
    out["conservation_norm"] = out["conservation_rank"].map({"low": 0.2, "medium": 0.5, "high": 1.0}).fillna(0.0)
    out["importance_norm"] = minmax(out["importance_score"])
    out["motif_norm"] = (out["motif_count"] > 0).astype(float)
    out["foldchange_norm"] = minmax(out["gene_foldchange"].abs())
    out["cluster_norm"] = minmax(out["cluster_importance"])
    out["priority_score"] = (
        weights["importance_score"] * out["importance_norm"]
        + weights["motif_match"] * out["motif_norm"]
        + weights["gene_foldchange"] * out["foldchange_norm"]
        + weights["conservation"] * out["conservation_norm"]
        + weights["temporal_cluster"] * out["cluster_norm"]
    )
    out = out.sort_values("priority_score", ascending=False).reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)
    return out


def parse_weights(raw: str | None) -> Dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if not raw:
        return weights
    for item in raw.split(","):
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value)
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cre-file", default="results/cre_candidates/infection_specific_cre.csv")
    parser.add_argument("--gene-annotation", default=None, help="GFF/GTF/BED gene annotation for nearest-gene assignment.")
    parser.add_argument("--motif-positions", default=None, help="Optional motif hits BED.")
    parser.add_argument("--motif-enrichment", default=None, help="Optional precomputed motif enrichment CSV.")
    parser.add_argument("--conservation-table", default=None, help="Optional CSV with cre_id and conservation columns.")
    parser.add_argument("--expression-table", default=None, help="Optional CSV with target_gene/gene_id and gene_foldchange.")
    parser.add_argument("--output-dir", default="results/annotation")
    parser.add_argument("--final-dir", default="results/final")
    parser.add_argument("--max-distance", type=int, default=50000)
    parser.add_argument("--weights", default=None, help="Comma list overriding priority weights.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    final_dir = ensure_dir(args.final_dir)
    cre = read_cre(args.cre_file)

    genes = read_gene_annotation(args.gene_annotation) if args.gene_annotation else pd.DataFrame()
    targets = assign_target_genes(cre, genes, args.max_distance)
    targets.to_csv(output_dir / "cre_target_genes.csv", index=False)
    annotated = cre.merge(targets, on="cre_id", how="left")

    if args.motif_positions:
        motif_summary, motif_positions = overlap_motifs(annotated, read_motif_positions(args.motif_positions))
        motif_summary.to_csv(output_dir / "cre_motif_summary.csv", index=False)
        if motif_positions.empty:
            motif_positions.to_csv(output_dir / "motif_positions.bed", sep="\t", header=False, index=False)
        else:
            write_bed(motif_positions, output_dir / "motif_positions.bed", ["chr", "start", "end", "cre_id", "motif_name"])
        annotated = annotated.merge(motif_summary, on="cre_id", how="left")
        motif_enrichment = motif_enrichment_from_positions(motif_positions, len(annotated))
    elif args.motif_enrichment:
        motif_enrichment = pd.read_csv(args.motif_enrichment)
    else:
        motif_enrichment = pd.DataFrame(columns=["motif_name", "matches", "expected", "p_value", "fold_enrichment"])
    motif_enrichment.to_csv(output_dir / "motif_enrichment.csv", index=False)

    annotated = merge_optional_table(annotated, args.conservation_table)
    if "conservation_rank" not in annotated:
        if "mean_phastcons" in annotated:
            annotated["conservation_rank"] = pd.cut(
                pd.to_numeric(annotated["mean_phastcons"], errors="coerce").fillna(0.0),
                bins=[-np.inf, 0.3, 0.6, np.inf],
                labels=["low", "medium", "high"],
            ).astype(str)
        else:
            annotated["conservation_rank"] = "unknown"
    conservation_cols = [c for c in ["cre_id", "mean_phylop", "mean_phastcons", "conservation_rank"] if c in annotated]
    annotated[conservation_cols].drop_duplicates("cre_id").to_csv(output_dir / "cre_conservation.csv", index=False)

    annotated = attach_expression(annotated, args.expression_table)
    ranked = compute_priority_score(annotated, parse_weights(args.weights))
    ranked.to_csv(final_dir / "cre_ranked.csv", index=False)
    save_json({"weights": parse_weights(args.weights), "max_distance": args.max_distance}, final_dir / "ranking_parameters.json")
    print(f"Saved annotations to {output_dir} and ranked CREs to {final_dir}")


if __name__ == "__main__":
    main()
