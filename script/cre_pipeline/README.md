# CRE Inference Pipeline

This folder contains a regenerated, standalone implementation of the CRE analysis pipeline described in `CRE_Analysis_Pipeline_Specification.md`. It does not modify the existing inference examples in `script/`.

## Scripts

1. `01_model_validation.py`: validate model predictions on gene/BED regions; compute per-region Pearson, RNA/ATAC group summaries, optional track plots.
2. `02_ism_scan.py`: run N-mask ISM on 51.2 kb gene regions (sliding or fixed mask windows).
3. `03_cre_screening.py`: convert delta importance scores into CRE candidate intervals.
4. `04_annotation.py`: assign target genes, merge optional motif/conservation/expression annotations, and rank CREs.
5. `05_validation_selection.py`: select validation candidates and generate an experiment design document.
6. `06_saliency_map.py`: compute gradient-based saliency maps (supports both standard and MoE checkpoints).

Shared helpers live in `common.py`. Model loading in `common.py` supports both standard heads and MoE (`use_moe = true` in the TOML config).

## Step 1: Model Validation (`01_model_validation.py`)

This script evaluates prediction quality at **region level** first, then summarizes RNA/ATAC performance across regions. It does **not** pool all batches/regions into a single global Pearson.

### Metrics

For each gene region:

- **`pearson_global`**: mean Pearson across all tracks (sequence-level correlation per track, same idea as `script/inference_pred_track.py`).
- **Per-track Pearson**: one value per track index.
- **Per-group Pearson / Spearman / peak AUC**: groups come from `--track-map` (or auto-inferred design), e.g. `RNA::3::CK`, `RNA::3::infect`, `ATAC::24`. Replicate tracks within a group are averaged before correlation.
- **Delta metrics per region**: e.g. `RNA::3` (`infect - CK`), `ATAC::12` (`12 - baseline`).

Across all regions, `global_metrics.json` reports **`group_summary`** and **`delta_summary`** (mean / median / std over regions) for each RNA/ATAC timepoint or contrast.

### Outputs

Written to `--output-dir` (default `results/model_performance`):

| File | Description |
|------|-------------|
| `region_prediction_summary.csv` | One row per region: coordinates, `mean_pred`, `mean_target`, `pearson_global` |
| `region_pearson_per_track.csv` | Per-region, per-track Pearson |
| `region_pearson_per_group.csv` | Per-region, per-group Pearson / Spearman / peak AUC |
| `region_delta_metrics.csv` | Per-region delta Pearson / Spearman |
| `track_summary.csv` | Per-track Pearson summarized across regions |
| `global_metrics.json` | Track design, `group_summary`, `delta_summary`, `track_summary` |
| `plots/{chrom}/{region_id}.png` | Optional track plots (only with `--plot`) |

Plots use `visualization_channels_means` with groups expanded from the design file (replicate tracks averaged per group).

### Useful CLI flags

Shared model/data flags come from `common.py`: `--config`, `--ckpt`, `--track-map`, `--batch-size`, `--limit-regions`, etc.

Step-1 specific flags:

- `--regions-bed`: optional BED instead of config `gene_bed`
- `--output-dir`: result directory
- `--peak-target-percentile`: threshold for per-region peak AUC (default `90.0`)
- `--plot`: save per-region track plots
- `--plot-limit-regions`: upper Y-axis limit for plots (lower bound fixed at `0`)
- `--plot-workers`: multiprocessing workers for plotting (default `8`)

Example with design file and plots:

```bash
python script/cre_pipeline/01_model_validation.py \
  --config fineturn_my_moe.toml \
  --ckpt "$CKPT" \
  --track-map config/expert_design.json \
  --limit-regions 2 \
  --batch-size 1 \
  --plot \
  --plot-limit-regions 10 \
  --output-dir results/model_performance_smoke
```

## Step 2: ISM Scan (`02_ism_scan.py`)

This script runs **N-mask in silico mutagenesis** on gene regions defined the same way as step 1 (`gene_bed` + `TSS_up` / `TSS_down`, typically 51.2 kb). Each mask window is inferred separately; importance is `original_score - masked_score`.

### Mask schemes

Choose one scheme per run with `--ism-scheme`:

| Scheme | Description |
|--------|-------------|
| `sliding` | Slide a fixed-length window with step `s` inside `[scan_start, scan_end)`, replace window with `N` |
| `fixed` | Use predefined intervals from `--mask-intervals-json` |

Fixed-interval JSON format (keys must match the **`id` column** in `gene_bed`):

```json
{
  "geneA": [[1000, 1200], [5000, 5200]],
  "geneB": [[0, 200]]
}
```

Intervals are **0-based half-open genomic absolute coordinates**; the script converts them to region-relative offsets internally.

Example: `window_size=200`, `window_step=100`, `scan_start=0`, `scan_end=51200` → **511 windows** per region (+ 1 original inference).

### ISM scores

For each mask window the script writes scalar scores:

- **Overall track**: mean over all tracks (sequence mean, then track mean)
- **Per-track**: one importance value per track index
- **Per-group**: one importance per design group (e.g. `RNA::24::infect`)
- **Delta**: group contrasts (`infect - CK`, ATAC `timepoint - baseline`) computed from masked vs original group scores

### Outputs

Written to `--output-dir` (default `results/ism`):

| File | Description |
|------|-------------|
| `ism_window_overall.csv` | One row per region × mask window, overall score/importance |
| `ism_window_per_track.csv` | Long format: region, window, track_idx, importance |
| `ism_window_per_group.csv` | Long format: region, window, group_key, importance |
| `ism_window_delta.csv` | Long format: region, window, delta_key, importance_delta |
| `region_metadata.csv` | One row per scanned region |
| `ism_parameters.json` | Run parameters and track design |

**Note:** step 2 now writes CSV window scores instead of `delta_importance.npy`. `03_cre_screening.py` still expects the old `.npy` format and will need to be updated before running the full pipeline end-to-end.

### Useful CLI flags

- `--ism-scheme sliding|fixed`
- `--scan-start`, `--scan-end` (sliding; default end = config `sequence_length`)
- `--window-size`, `--window-step` (sliding)
- `--mask-intervals-json` (fixed)
- `--inference-batch-size` (default `16`)
- `--max-windows` (smoke-test limit per region)
- `--limit-regions`, `--track-map`, `--overwrite`

Sliding example:

```bash
python script/cre_pipeline/02_ism_scan.py \
  --config fineturn_my_moe.toml \
  --ckpt "$CKPT" \
  --track-map config/expert_design.json \
  --ism-scheme sliding \
  --scan-start 0 \
  --scan-end 51200 \
  --window-size 200 \
  --window-step 100 \
  --limit-regions 1 \
  --max-windows 4 \
  --inference-batch-size 2 \
  --output-dir results/ism_smoke \
  --overwrite
```

Fixed-interval example:

```bash
python script/cre_pipeline/02_ism_scan.py \
  --ckpt "$CKPT" \
  --track-map config/expert_design.json \
  --ism-scheme fixed \
  --mask-intervals-json config/ism_mask_intervals.json \
  --output-dir results/ism_fixed \
  --overwrite
```

## Step 6: Saliency Map (`06_saliency_map.py`)

This script computes gradient saliency maps for selected regions with the same model-loading path used by the pipeline (`common.load_model_for_inference`), so `use_moe = true` checkpoints are supported automatically.

### Inputs

- Regions: `--regions-bed` or default config gene regions
- Scoring tracks:
  - `--track-indices 0,1,2` (explicit list), or
  - `--score-group RNA::24::infect` (resolved from `--track-map`)
- Optional logits sub-window: `--saliency-region start:end`

### Outputs

Written to `--output-dir` (default `results/saliency`):

| File | Description |
|------|-------------|
| `gradients_ordered_<chunk>.npy` | Concatenated saliency gradients for each chunk |
| `embeddings_ordered_<chunk>.npy` | Concatenated one-hot inputs for each chunk (`--save-onehot`) |
| `saliency_chunk_<chunk>.csv` | Region-to-row mapping for each saved chunk |
| `saliency_run_config.json` | Run configuration and selected tracks/groups |

### Example

```bash
python script/cre_pipeline/06_saliency_map.py \
  --config fineturn_my_moe.toml \
  --ckpt "$CKPT" \
  --track-map config/expert_design.json \
  --score-group RNA::24::infect \
  --limit-regions 100 \
  --chunk-size 200 \
  --output-dir results/saliency_moe \
  --overwrite
```

## Track Mapping

`--track-map` now supports a nested design file (JSON/TOML) with both RNA and ATAC sections.

Recommended format (same as your `expert_design.json`):

```json
{
  "RNA": {
    "3": { "CK": [0, 1, 2], "infect": [18, 19, 20] },
    "6": { "CK": [3, 4, 5], "infect": [21, 22, 23] },
    "12": { "CK": [6, 7, 8], "infect": [24, 25, 26] },
    "24": { "CK": [9, 10, 11], "infect": [27, 28, 29] },
    "36": { "CK": [12, 13, 14], "infect": [30, 31, 32] },
    "48": { "CK": [15, 16, 17], "infect": [33, 34, 35] }
  },
  "ATAC": {
    "0": [54, 55],
    "12": [56, 57],
    "24": [58, 59],
    "48": [60, 61]
  }
}
```

When no `--track-map` is provided, the scripts auto-infer:

- RNA groups from file prefixes (`C*` -> `CK`, `T*` -> `infect`)
- ATAC timepoints from ATAC-labeled tracks (`track_label_list == 1`)

Backward-compatible flat mapping is still accepted and treated as ATAC-only.

## Delta Rules

Difference computation follows your experiment design:

- RNA-Seq: per-timepoint contrast `infect - CK` (for each of `3, 6, 12, 24, 36, 48`; only when both groups exist in the design)
- ATAC-Seq: baseline contrast `timepoint - baseline`, default baseline is `0`

`01_model_validation.py` applies the same contrasts per region and writes `region_delta_metrics.csv` / `delta_summary` in `global_metrics.json`.

`02_ism_scan.py` computes the same contrasts per mask window and writes `ism_window_per_group.csv` / `ism_window_delta.csv`.

## Example Run

Set the checkpoint once:

```bash
CKPT=/path/to/model.ckpt
```

Run a small smoke test first:

```bash
python script/cre_pipeline/01_model_validation.py \
  --ckpt "$CKPT" \
  --track-map config/expert_design.json \
  --limit-regions 2 \
  --batch-size 1 \
  --output-dir results/model_performance_smoke
python script/cre_pipeline/02_ism_scan.py \
  --ckpt "$CKPT" \
  --track-map config/expert_design.json \
  --ism-scheme sliding \
  --limit-regions 1 \
  --max-windows 4 \
  --window-size 200 \
  --window-step 100 \
  --inference-batch-size 2 \
  --output-dir results/ism_smoke \
  --overwrite
python script/cre_pipeline/03_cre_screening.py --ism-dir results/ism_smoke --output-dir results/cre_candidates_smoke --min-width 1 --max-width 500
python script/cre_pipeline/04_annotation.py --cre-file results/cre_candidates_smoke/infection_specific_cre.csv --output-dir results/annotation_smoke --final-dir results/final_smoke
python script/cre_pipeline/05_validation_selection.py --ranked-file results/final_smoke/cre_ranked.csv --output-dir results/validation_smoke
```

Run the full ISM step on all gene regions:

```bash
python script/cre_pipeline/02_ism_scan.py \
  --ckpt "$CKPT" \
  --track-map config/expert_design.json \
  --ism-scheme sliding \
  --scan-start 0 \
  --scan-end 51200 \
  --window-size 200 \
  --window-step 100 \
  --inference-batch-size 16 \
  --output-dir results/ism \
  --overwrite
```

Then continue with scripts `03` to `05` after updating step 3 for the new CSV outputs.

## Optional Annotation Inputs

`04_annotation.py` can run with only CRE coordinates. Additional evidence can be merged when available:

- `--gene-annotation genes.gff3`: nearest gene assignment.
- `--motif-positions motif_hits.bed`: motif overlaps per CRE and motif summary.
- `--conservation-table cre_conservation.csv`: columns such as `cre_id`, `mean_phylop`, `mean_phastcons`, `conservation_rank`.
- `--expression-table gene_foldchange.csv`: columns `target_gene` or `gene_id`, plus `gene_foldchange`.

Missing optional inputs are kept as `unknown` or zero-valued ranking features so the core pipeline still completes.
