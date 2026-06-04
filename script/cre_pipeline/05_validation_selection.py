#!/usr/bin/env python3
"""Step 5: select CRE validation candidates and create an experiment design."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from common import ensure_dir


def sample_pool(pool: pd.DataFrame, n: int, tier: str, seed: int) -> pd.DataFrame:
    if pool.empty or n <= 0:
        return pd.DataFrame(columns=list(pool.columns) + ["priority_tier", "validation_type"])
    selected = pool.sample(n=min(n, len(pool)), random_state=seed).copy()
    selected["priority_tier"] = tier
    selected["validation_type"] = np.where(
        (tier == "high") & (selected["priority_rank"].astype(int) <= 10),
        "LUC+CRISPR",
        "LUC",
    )
    return selected


def select_validation_candidates(
    ranked: pd.DataFrame,
    *,
    n_high: int,
    n_medium: int,
    n_low: int,
    high_pool: int,
    medium_start: int,
    medium_end: int,
    low_start: int,
    low_end: int,
    seed: int,
) -> pd.DataFrame:
    ranked = ranked.sort_values("priority_rank").reset_index(drop=True)
    high = sample_pool(ranked.head(high_pool), n_high, "high", seed)
    medium = sample_pool(
        ranked[(ranked["priority_rank"] >= medium_start) & (ranked["priority_rank"] <= medium_end)],
        n_medium,
        "medium",
        seed + 1,
    )
    low = sample_pool(
        ranked[(ranked["priority_rank"] >= low_start) & (ranked["priority_rank"] <= low_end)],
        n_low,
        "low",
        seed + 2,
    )
    candidates = pd.concat([high, medium, low], ignore_index=True)
    if candidates.empty:
        return candidates
    tier_order = pd.CategoricalDtype(["high", "medium", "low"], ordered=True)
    candidates["priority_tier"] = candidates["priority_tier"].astype(tier_order)
    return candidates.sort_values(["priority_tier", "priority_rank"], ascending=[True, True]).reset_index(drop=True)


def value(row: pd.Series, key: str, default: str = "unknown") -> str:
    val = row.get(key, default)
    if pd.isna(val) or val == "":
        return default
    return str(val)


def write_experiment_design(candidates: pd.DataFrame, path: Path) -> None:
    lines: List[str] = [
        "# CRE 验证实验设计",
        "",
        "本设计表由 CRE priority ranking 自动生成，候选分为 high、medium、low 三个优先级，用于覆盖不同置信度区间的验证成功率。",
        "",
    ]
    tier_titles = {
        "high": "高优先级候选",
        "medium": "中优先级候选",
        "low": "低优先级候选",
    }
    for tier in ["high", "medium", "low"]:
        subset = candidates[candidates["priority_tier"] == tier]
        lines.extend([f"## {tier_titles[tier]} ({len(subset)}个)", ""])
        if subset.empty:
            lines.extend(["暂无候选。", ""])
            continue
        for _, row in subset.iterrows():
            cre_id = value(row, "cre_id")
            motif = value(row, "motif_match")
            target = value(row, "target_gene")
            lines.extend(
                [
                    f"### {cre_id} ({motif} motif, target: {target})",
                    f"- **位置**: {value(row, 'chr')}:{int(row['start'])}-{int(row['end'])}",
                    f"- **优先级排名**: {int(row['priority_rank'])}",
                    f"- **重要性分数**: {float(row.get('importance_score', 0.0)):.4f}",
                    f"- **综合评分**: {float(row.get('priority_score', 0.0)):.4f}",
                    f"- **验证实验**: {value(row, 'validation_type', 'LUC')}",
                    "- **LUC 报告基因**: 克隆 CRE 区域到最小启动子报告载体，比较侵染与对照处理。",
                    "- **预期结果**: 侵染条件下报告活性与目标基因诱导趋势一致。",
                ]
            )
            if value(row, "validation_type", "LUC") == "LUC+CRISPR":
                lines.extend(
                    [
                        "- **CRISPR 删除**: 设计 sgRNA 靶向 CRE 两侧，验证删除后目标基因响应是否下降。",
                        "- **表型观察**: 根据目标基因功能记录抗病或发育相关表型。",
                    ]
                )
            lines.append("")
    lines.extend(
        [
            "## 实验时间线",
            "- Week 1-2: 引物设计、片段扩增与载体构建。",
            "- Week 3-4: 原生质体或瞬时表达体系中完成 LUC 初筛。",
            "- Week 5-8: 高优先级候选推进稳定转化或 CRISPR 材料构建。",
            "- Week 9-12: 基因表达、报告活性和表型联动验证。",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked-file", default="results/final/cre_ranked.csv")
    parser.add_argument("--output-dir", default="results/validation")
    parser.add_argument("--n-high", type=int, default=5)
    parser.add_argument("--n-medium", type=int, default=3)
    parser.add_argument("--n-low", type=int, default=2)
    parser.add_argument("--high-pool", type=int, default=10)
    parser.add_argument("--medium-start", type=int, default=20)
    parser.add_argument("--medium-end", type=int, default=50)
    parser.add_argument("--low-start", type=int, default=100)
    parser.add_argument("--low-end", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    ranked = pd.read_csv(args.ranked_file)
    candidates = select_validation_candidates(
        ranked,
        n_high=args.n_high,
        n_medium=args.n_medium,
        n_low=args.n_low,
        high_pool=args.high_pool,
        medium_start=args.medium_start,
        medium_end=args.medium_end,
        low_start=args.low_start,
        low_end=args.low_end,
        seed=args.seed,
    )
    candidates.to_csv(output_dir / "validation_candidates.csv", index=False)
    write_experiment_design(candidates, output_dir / "experiment_design.md")
    print(f"Saved {len(candidates)} validation candidates to {output_dir}")


if __name__ == "__main__":
    main()
