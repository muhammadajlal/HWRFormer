# HWRFormer — Anonymous Code Release

Anonymous code release accompanying the ECML PKDD 2026 workshop submission
*"Mitigating Exposure Bias in IMU Handwriting Recognition."* It is released for
review; identifying information has been removed.

## Introduction

HWRFormer is an encoder–decoder recognizer for inertial-measurement-unit (IMU)
handwriting recognition: a 1D CNN encoder feeds an autoregressive (AR)
Transformer decoder with scaled-dot-product-attention (SDPA) output gating. The
paper studies two training-time interventions that mitigate exposure bias in the
AR decoder — **input corruption** and **hybrid CTC–AR training** — and contrasts
them against the recurrent CNN-BiLSTM-CTC baseline.

![HWRFormer architecture](figures/architecture.png)

*HWRFormer architecture and hybrid training. The solid horizontal path is used at
inference: a shared 1D CNN encoder feeds an AR Transformer decoder with SDPA
output gating. During hybrid training, an auxiliary CTC head on the same encoder
adds a training-only loss summed with the AR loss; the dashed connector denotes
the optional weight-tying ablation.*

## Installation

```bash
conda create -n hwrformer python=3.10
conda activate hwrformer
pip install -r requirements.txt
export REPO=$(pwd)        # configs reference ${REPO} for dataset / output paths
```

## Dataset

The paper's primary benchmark is the **public** OnHW-words500 dataset. Download
it from the Fraunhofer IIS OnHW release and convert it to the MSCOCO-like layout
(`train.json` / `val.json` + per-sample CSVs) used here, then place it under:

```
${REPO}/data/onhw_wi_word_rh/   # writer-independent words
${REPO}/data/onhw_wd_word_rh/   # writer-dependent words
```

The private IMU pen dataset used for cross-dataset confirmation in the paper
**cannot be redistributed** and is therefore not included.

## Training

Edit `configs/train.yaml` (it documents every knob inline) and run 5-fold
cross-validation with a single command — `train_cv.py` generates one config per
fold, runs `main.py` on each sequentially, and cleans up afterwards:

```bash
python train_cv.py -c configs/train.yaml
```

To train a single fold instead, set `idx_fold` to `0..4` in the config and call
`main.py` directly:

```bash
python main.py -c configs/train.yaml
```

Training uses 300 epochs, AdamW, a linear-warmup/cosine schedule (30 warmup
epochs), and batch size 64. Each fold writes to `<dir_work>/<fold>/`.

`configs/others/` holds ready-to-run examples for the other regimes:

| Config | Model |
|---|---|
| `configs/train.yaml` | HWRFormer (AR + elementwise SDPA gating) — the headline model |
| `configs/others/train_rewi_ctc.yaml` | CNN-BiLSTM-CTC baseline (REWI) |
| `configs/others/train_transformer_ctc.yaml` | Parameter-matched Transformer-CTC |
| `configs/others/train_corruption.yaml` | AR + input corruption |
| `configs/others/train_hybrid.yaml` | Hybrid CTC–AR |

### Reproducing paper experiments

Every paper result is a knob change on one of the configs above. Start from the
listed base config, apply the override, set `dir_work` to the group shown (so
results land where the figure scripts expect them), and run `train_cv.py`. Swap
`onhw_wi_word_rh` → `onhw_wd_word_rh` in `dir_dataset` and `dir_work` for the
writer-dependent split.

| Experiment | Base config | Knob override | `dir_work` group (under `${REPO}/results/hwr2/`) |
|---|---|---|---|
| AR baseline (elementwise) | `train.yaml` | *(defaults)* | `Baseline-AR-blconv_b/ar_transformer__onhw_wi_word_rh` |
| AR, no gating | `train.yaml` | `use_gated_attention: false` | `Baseline-AR-Ungated/ar_transformer__onhw_wi_word_rh` |
| AR, headwise gating | `train.yaml` | `gating_type: headwise` | `Baseline-AR-HeadwiseGating/ar_transformer__onhw_wi_word_rh` |
| Transformer-CTC | `others/train_transformer_ctc.yaml` | *(as given)* | `Baseline-Transformer-CTC-Matched/transformer__onhw_wi_word_rh` |
| CNN-BiLSTM-CTC (REWI) | `others/train_rewi_ctc.yaml` | *(as given)* | `blconv_bilstm_wide_no_tokenizer/bilstm_wide__onhw_wi_word_rh` |
| Corruption mode | `others/train_corruption.yaml` | `input_corruption.mode:` `uniform` \| `bigram_left` \| `bigram_right` \| `self_confusion` \| `adjacent_swap` | `Baseline-AR-InputCorruption-<mode>/ar_transformer__onhw_wi_word_rh` |
| Corruption-rate sweep | `others/train_corruption.yaml` | `input_corruption.p_replace:` `0.05` \| `0.10` \| `0.15` \| `0.20` \| `0.30` | `Baseline-AR-InputCorruption-Sweep-blconv_b/ar_transformer__onhw_wi_word_rh__p0p<NN>` |
| Hybrid λ sweep | `others/train_hybrid.yaml` | `dual_head.lambda_ctc:` `0.1 … 1.0` | `train_element_word_hybrid_<NN>_onhw_wi/ar_transformer__onhw_wi_word_rh` |
| Hybrid weight-tying | `others/train_hybrid.yaml` | `dual_head.tie.ctc_to_ar_outproj: true` | `train_element_word_hybrid_<NN>_onhw_wi_ctc_to_ar_outproj/ar_transformer__onhw_wi_word_rh` |
| Hybrid + corruption | `others/train_hybrid.yaml` + `input_corruption` block | both blocks (λ=0.1) | `HybridInputCorruption_<mode>/ar_transformer__onhw_wi_word_rh` |

`self_confusion` additionally needs per-fold confusion matrices at
`input_corruption.confusion_path`; the other corruption modes are self-contained
(the bigram table is built from the training labels at runtime).

## Evaluation

The 5-fold means are produced directly by training. To aggregate them (5-fold
mean ± std CER/WER plus parameter counts and MACs via `FlopCounterMode`):

```bash
python evaluate.py -c configs/train.yaml
```

To re-score a single trained checkpoint, edit `configs/test.yaml` (`checkpoint`,
`idx_fold`) and run `python main.py -c configs/test.yaml`.

The paper's figures can be regenerated from the pre-aggregated numbers in
`HWRFormer.json` without re-running training:

```bash
python analysis/scripts/plot_corruption_modes_bars.py    # corruption-mode bar chart
python analysis/scripts/plot_corruption_p_sweep.py       # p_ic sweep curves
python analysis/scripts/plot_lambda_sweep.py             # lambda_ctc sweep
```

## Repository layout

```
main.py                 # train / evaluate one fold
train_cv.py             # run all cross-validation folds sequentially
evaluate.py             # 5-fold aggregation + params/MACs
hwrformer/              # core library
  model/                #   1D CNN encoder, AR Transformer decoder, Transformer-CTC,
                        #   hybrid dual-head, SDPA gating
  dataset/              #   IMU loaders, augmentations, collation
  training/             #   train/eval loops (CTC, AR, hybrid, input corruption)
  analysis/             #   metrics + encoder-feature analysis
  ctc_decoder.py        #   CTC best-path decoder
configs/                # train.yaml, test.yaml, others/ (see Training)
analysis/scripts/       # table + figure regeneration scripts
figures/                # architecture figure used in this README
HWRFormer.json          # pre-aggregated 5-fold means for the paper's tables/figures
requirements.txt        # Python dependencies
LICENSE.txt             # MIT
```

## License

MIT — see `LICENSE.txt`.

## Citation

Citation details are omitted for anonymous review.
