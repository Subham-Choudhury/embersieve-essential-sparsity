# EmberSieve — Essential Sparsity in Transformer Compression

**A Benchmarking Study on BERT-base and DistilBERT across GLUE**

Summer research internship project, Indian Institute of Technology Bhubaneswar.
Implements and benchmarks the **one-shot magnitude pruning (OMP)** methodology from
[VITA-Group's *Essential Sparsity*](https://github.com/VITA-Group/essential_sparsity)
across the full GLUE benchmark suite.

---

## Overview

This project investigates how much of a pretrained transformer's weights can be
removed via one-shot magnitude pruning before task performance meaningfully
degrades — the "essential sparsity" of the model. Two models (BERT-base,
DistilBERT) are fine-tuned on all 9 GLUE tasks, then pruned in a single shot at
12 sparsity levels (0% to 99%) with **no further fine-tuning**, and re-evaluated.

**Full experimental grid:** 2 models × 9 GLUE tasks × 12 sparsity levels = **216 runs**,
each measured across 9 metrics (accuracy, precision, recall, F1, memory size,
latency, throughput, bit-width, energy consumption).

## Methodology

- **Pruning method:** Global unstructured L1 magnitude pruning, applied jointly
  across all `nn.Linear` layers (not per-layer thresholds) — matching the
  Essential Sparsity paper's one-shot definition.
- **Models:** `bert-base-uncased`, `distilbert-base-uncased`
- **Tasks:** SST-2, QNLI, QQP, WNLI, MNLI, MRPC, STS-B, RTE, CoLA
- **Sparsity levels:** 0%, 10%, 20%, ..., 90%, 95%, 99%
- **Fixed hyperparameters:** AdamW, weight decay 0.01, batch size 16, LR 2e-5,
  10% warmup, linear decay, 5 epochs, max sequence length 128, seed 42
- **Framework:** PyTorch + HuggingFace Transformers/Trainer, run on Google Colab (T4 GPU)

## Key Findings

- **Consistent "essential sparsity" threshold across most tasks:** both models
  tolerate **40–50% one-shot pruning** while staying within 5 percentage points
  of dense-model accuracy (SST-2, QQP, QNLI reach 50%; CoLA, MRPC, STS-B, MNLI,
  RTE reach 40%).
- **Latency is largely unaffected by unstructured sparsity** on standard GPU
  hardware — despite the theoretical bit-width reduction, dense matrix
  multiplies still execute the same shape. This illustrates the well-known gap
  between theoretical and *realized* compression without specialized sparse
  kernels or structured pruning.
- **STS-B (regression) shows a genuine sparsity cliff:** at ≥90% sparsity the
  regression head collapses to a constant prediction, making Pearson/Spearman
  correlation mathematically undefined.
- **WNLI is excluded from sparsity-tolerance conclusions.** Its dense baseline
  itself underperforms the majority-class heuristic (a well-documented
  characteristic of WNLI in the BERT/GLUE literature), so its "flat" accuracy
  curve across sparsity levels reflects the task's known instability, not a
  genuine pruning-robustness finding.

See [`results/dense_baseline_table.csv`](results/dense_baseline_table.csv) for
the full dense-model comparison table, and
[`results/sparsity_cliff_summary.csv`](results/sparsity_cliff_summary.csv) for
the per-task/model sparsity-tolerance summary.

## Repository Structure

```
├── notebooks/
│   └── essential_sparsity_pipeline.ipynb   # Full pipeline: setup, pruning, training, evaluation, analysis
├── results/
│   ├── essential_sparsity_results.csv      # Full 216-row result set (all sparsity levels)
│   ├── dense_baseline_table.csv            # Per-task, per-model dense-baseline comparison
│   ├── sparsity_cliff_summary.csv          # Max "safe" sparsity per task/model
│   └── figures/                            # Sparsity-vs-accuracy plots, cliff charts, trade-off plots
├── docs/
│   └── methodology_notes.md                # Extended notes on design decisions and known limitations
├── requirements.txt
└── README.md
```

> **Note on data:** Model checkpoints (~400 MB–1.7 GB each × 18 dense models,
> plus pruned variants) are excluded from this repository due to size. The
> notebook is fully self-contained and will regenerate all checkpoints and
> results from scratch when run end-to-end.

## Reproduction

1. Open `notebooks/essential_sparsity_pipeline.ipynb` in Google Colab (GPU runtime recommended).
2. Run all cells top to bottom. The pipeline is fully resumable — if a Colab
   session disconnects mid-run, re-running the notebook picks up exactly where
   it left off (checkpoints and results persist to Google Drive).
3. Full reproduction of all 216 runs takes approximately 20–25 GPU-hours in
   total, dominated by the two largest GLUE tasks (QQP, MNLI: ~360k+ training
   examples each).

Locally (outside Colab), adjust `PROJECT_ROOT` in Cell 2 to a local directory —
Google Drive mounting is skipped automatically if `google.colab` is unavailable.

## Known Limitations

- Unstructured pruning does not yield measured inference speedups on standard
  (non-sparse-optimized) GPU hardware — see Key Findings above.
- Energy consumption figures reflect short inference-time profiling passes via
  [CodeCarbon](https://codecarbon.io/), not full training-run energy (which was
  not captured during the original training passes).
- WNLI results should not be interpreted as genuine sparsity-tolerance findings
  (see Key Findings above).
- Reproducing results from scratch on different hardware may show minor
  (sub-percentage-point) accuracy variation due to non-deterministic GPU
  floating-point operations under mixed precision (`fp16`), even with a fixed
  seed.

## Acknowledgements

Methodology based on VITA-Group's [Essential Sparsity](https://github.com/VITA-Group/essential_sparsity)
research. Conducted as part of a summer research internship at the Indian
Institute of Technology Bhubaneswar, under the supervision of Dr. Devashree
Tripathy, with mentorship from Sugyan Mishra.

## License

MIT — see [`LICENSE`](LICENSE).
