# HWRFormer — Anonymous Code Release

Anonymous code release accompanying the ECML PKDD 2026 workshop submission
*"Mitigating Exposure Bias in IMU Handwriting Recognition."*

This repository contains the model code, training/evaluation entry points,
experiment configurations, and the table/figure regeneration scripts used
in the paper. It is released for review; identifying information has been
removed.

## What is HWRFormer?

HWRFormer is an encoder–decoder recognizer for inertial-measurement-unit
(IMU) handwriting recognition: a 1D CNN encoder feeds an autoregressive (AR)
Transformer decoder with scaled-dot-product-attention (SDPA) output gating.
The paper studies two training-time interventions that mitigate exposure
bias in the AR decoder — **input corruption** and **hybrid CTC–AR training**
— and contrasts them against the recurrent CNN-BiLSTM-CTC baseline.

## Repository layout

```
main.py                 # Train / evaluate one fold
evaluate.py             # 5-fold cross-validation aggregation + params/MACs
hwrformer/              # Core library
  model/                #   encoders (1D CNN), AR Transformer decoder,
                        #   CTC Transformer, hybrid dual-head, gating
  dataset/              #   IMU loaders, augmentations, collation
  training/             #   train/eval loops (CTC, AR, hybrid)
  analysis/             #   metrics + encoder-feature analysis
  ctc_decoder.py        #   CTC best-path decoder
configs/                # YAML experiment configs (see below)
analysis/scripts/       # Table + figure regeneration scripts
HWRFormer.json          # Pre-aggregated 5-fold means for all paper tables/figures
LICENSE.txt             # MIT
requirements.txt        # Python dependencies
```

## Configs (public OnHW experiments only)

All configs train on the **public** OnHW-words500 splits (writer-independent
and writer-dependent). Configs for the private dataset are not included
because that data cannot be redistributed.

| Directory | Experiment |
|---|---|
| `configs/Baseline-REWI/` | CNN-BiLSTM-CTC baseline (REWI) |
| `configs/AR-Baseline/` | HWRFormer: AR (no gating / elementwise / headwise) + parameter-matched Transformer-CTC |
| `configs/AR-Baseline-WD/` | Writer-dependent OnHW variants |
| `configs/AR-InputCorruption/` | The five input-corruption modes (uniform, bigram-right, bigram-left, self-confusion, adjacent-swap) |
| `configs/AR-InputCorruption-Sweep/` | Corruption-rate (`p_ic`) sweep |
| `configs/hybrid/` | Hybrid CTC–AR `λ_ctc` sweep + weight-tying ablation |
| `configs/HybridInputCorruption/` | Hybrid + corruption combination (λ_ctc = 0.1) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export REPO=$(pwd)        # configs reference ${REPO} for dataset / output paths
```

## Data

The paper's primary benchmark is the **public** OnHW-words500 dataset.
Download it from the Fraunhofer IIS OnHW release and the conversion notebook
referenced therein, then place the per-sample CSV + JSON layout under:

```
${REPO}/data/onhw_wi_word_rh/   # writer-independent words
${REPO}/data/onhw_wd_word_rh/   # writer-dependent words
```

The private IMU pen dataset used for cross-dataset confirmation in the paper
**cannot be redistributed** and is therefore not included; the corresponding
configs are omitted.

## Reproducing the paper

Train one fold:

```bash
python main.py -c configs/AR-Baseline/train-ar-baseline-onhw-word.yaml
```

(Set `idx_fold` 0–4 in the YAML, or override on the command line, to run the
five writer-disjoint folds. Training uses 300 epochs, AdamW, cosine schedule,
30 warmup epochs, batch size 64.)

Aggregate cross-validation results (computes 5-fold mean CER/WER, parameter
counts, and MACs via `FlopCounterMode`):

```bash
python evaluate.py -c configs/AR-Baseline/train-ar-baseline-onhw-word.yaml
```

Regenerate paper figures from the pre-aggregated numbers without re-running
training:

```bash
python analysis/scripts/plot_corruption_modes_bars.py   # corruption-mode bar chart
python analysis/scripts/plot_corruption_p_sweep.py      # p_ic sweep curves
python analysis/scripts/plot_lambda_sweep.py            # lambda_ctc sweep
python analysis/scripts/ctc_posterior_lambda_analysis.py # CTC posterior diagnostics
python analysis/scripts/compare_ar_hybrid.py            # PCA + cosine diagnostics
```

`HWRFormer.json` contains the 5-fold means underlying every table and figure
in the paper, so the plots can be rebuilt without access to the training
checkpoints.

## License

MIT — see `LICENSE.txt`.
