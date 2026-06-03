# HWRFormer — Anonymous Code Release

Anonymous code release accompanying the ECML PKDD 2026 workshop submission
*"Mitigating Exposure Bias in IMU Handwriting Recognition."* It is released for
review; identifying information has been removed.

## Introduction

HWRFormer is an encoder–decoder recognizer for inertial-measurement-unit (IMU)
handwriting recognition: a 1D CNN encoder feeds an autoregressive (AR)
Transformer decoder with scaled-dot-product-attention (SDPA) output gating. The
paper studies two training-time interventions that mitigate exposure bias in the
AR decoder — **noise injection** and **hybrid CTC–AR training** — and contrasts
them against the recurrent CNN-BiLSTM-CTC baseline.

![HWRFormer architecture](figures/architecture.png)

*HWRFormer architecture and hybrid training. The solid horizontal path is used at
inference: a shared 1D CNN encoder feeds an AR Transformer decoder with SDPA
output gating. During hybrid training, an auxiliary CTC head on the same encoder
adds a training-only loss summed with the AR loss; the dashed connector denotes
the optional weight-tying ablation.*

## Results

5-fold cross-validation on the **public** OnHW-words500 splits (writer-independent
and writer-dependent). CER/WER in %, lower is better; **bold** = best per column.
MACs are reported at greedy AR decoding `len_max = 6`.

**Architecture migration — CTC baseline → HWRFormer.** Cells show 5-fold mean ± across-fold sample std (n=5).

| Model | Obj. | Gating | WI CER | WI WER | WD CER | WD WER | #Params | MACs |
|---|---|---|---|---|---|---|---|---|
| REWI (CNN-BiLSTM) | CTC | — | 7.30 ± 6.85 | 15.16 ± 10.05 | **14.81** ± 3.84 | 44.77 ± 7.84 | 4.64M | **413M** |
| HWRFormer | AR | — | 7.10 ± 7.15 | **10.39** ± 8.08 | 16.47 ± 5.73 | 32.07 ± 9.21 | 4.57M | 653M |
| HWRFormer | AR | elementwise | **6.94** ± 7.10 | 10.50 ± 8.26 | 16.31 ± 6.20 | 31.87 ± 9.31 | 4.64M | 669M |
| HWRFormer | AR | headwise | 6.99 ± 7.19 | 10.47 ± 8.42 | 15.70 ± 5.91 | **31.08** ± 9.25 | 4.57M | 667M |

**Exposure-bias interventions** (CER / WER, mean ± std), HWRFormer = elementwise-gated AR.

| Dataset | REWI | HWRFormer | + Noise inj. | + Hybrid | + Hybrid + Noise inj. |
|---|---|---|---|---|---|
| OnHW-words500 (WI) | 7.30 / 15.16<br>±6.85 / ±10.05 | 6.94 / 10.50<br>±7.10 / ±8.26 | 6.86 / 13.04<br>±6.66 / ±8.82 | 6.83 / **10.17**<br>±7.10 / ±8.20 | **6.70** / 12.93<br>±6.64 / ±9.21 |
| OnHW-words500 (WD) | 14.81 / 44.77<br>±3.84 / ±7.84 | 16.31 / 31.87<br>±6.20 / ±9.31 | 13.52 / 35.98<br>±4.06 / ±6.17 | 13.39 / **27.42**<br>±5.70 / ±9.28 | **11.49** / 31.72<br>±3.99 / ±6.42 |

The large OnHW WI deviations reflect across-writer difficulty variation rather
than training instability (individual folds range from ~1 to ~15 % CER on WI),
so sub-percentage-point differences on WI are within fold noise. See the
"Reproducing paper experiments" table below to regenerate any cell.

**Direct exposure-bias probe** (CER, %; mean over 5 folds). Each saved checkpoint
is re-evaluated under two decoding regimes: **w/ TF** feeds the ground-truth
prefix at every step (per-position argmax); **w/o TF** is the greedy decoder
that feeds its own previous prediction back in (the inference path used in
the tables above). The jump from w/ TF to w/o TF is the prefix-drift cost
and is the direct exposure-bias signal.

| Training | OnHW (WI) w/ TF | OnHW (WI) w/o TF | OnHW (WD) w/ TF | OnHW (WD) w/o TF |
|---|---|---|---|---|
| HWRFormer (AR) | 2.76 | 6.95 | 10.91 | 16.31 |
| + Noise inj. | 5.23 | 6.86 | 11.76 | 13.52 |
| + Hybrid | 2.77 | 6.83 | 9.86 | 13.38 |
| + Hybrid + Noise inj. | 5.33 | 6.70 | 10.33 | 11.48 |

Noise injection compresses the w/ TF → w/o TF jump on both public splits and on
the private subsets reported in the paper. Hybrid CTC–AR alone barely changes
the jump, consistent with its role as an encoder-side regularizer. Reproduce
with `eval_tf_gap.py` (see the Evaluation section).

## Installation

```bash
conda create -n hwrformer python=3.10
conda activate hwrformer
pip install -r requirements.txt
export REPO=$(pwd)        # configs reference ${REPO} for dataset / output paths
```

## Dataset

For commercial reasons, the private IMU-pen datasets used for cross-dataset
confirmation in the paper are not published. Alternatively, the **public**
OnHW-words500 dataset can be used for training and evaluation; the paper's
primary benchmark is its right-handed writer-independent subset. Download it
from the Fraunhofer IIS OnHW release:
<https://www.iis.fraunhofer.de/de/ff/lv/dataanalytics/anwproj/schreibtrainer/onhw-dataset.html>

We use a MSCOCO-like structure (`train.json` / `val.json` + per-sample CSVs).
After downloading, convert the original dataset to this structure with the
`onhw.ipynb` notebook, adjusting the `dir_raw`, `dir_out`, and `writer_indep`
variables accordingly. Then place the converted dataset under:

```
${REPO}/data/onhw_wi_word_rh/   # writer-independent words
${REPO}/data/onhw_wd_word_rh/   # writer-dependent words
```

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
| `configs/others/train_noise_injection.yaml` | AR + noise injection |
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
| Noise-injection mode | `others/train_noise_injection.yaml` | `noise_injection.mode:` `uniform` \| `bigram_left` \| `bigram_right` \| `self_confusion` \| `adjacent_swap` | `Baseline-AR-NoiseInjection-<mode>/ar_transformer__onhw_wi_word_rh` |
| Noise-injection rate sweep | `others/train_noise_injection.yaml` | `noise_injection.p_replace:` `0.05` \| `0.10` \| `0.15` \| `0.20` \| `0.30` | `Baseline-AR-NoiseInjection-Sweep-blconv_b/ar_transformer__onhw_wi_word_rh__p0p<NN>` |
| Hybrid λ sweep | `others/train_hybrid.yaml` | `dual_head.lambda_ctc:` `0.1 … 1.0` | `train_element_word_hybrid_<NN>_onhw_wi/ar_transformer__onhw_wi_word_rh` |
| Hybrid weight-tying | `others/train_hybrid.yaml` | `dual_head.tie.ctc_to_ar_outproj: true` | `train_element_word_hybrid_<NN>_onhw_wi_ctc_to_ar_outproj/ar_transformer__onhw_wi_word_rh` |
| Hybrid + noise injection | `others/train_hybrid.yaml` + `noise_injection` block | both blocks (λ=0.1) | `HybridNoiseInjection_<mode>/ar_transformer__onhw_wi_word_rh` |
| Exposure-bias probe (Table 3) | any saved fold checkpoint | *(no training; run `eval_tf_gap.py`, see Evaluation)* | writes `eval_tf_gap.json` next to the fold's `train.yaml` |

`self_confusion` additionally needs per-fold confusion matrices at
`noise_injection.confusion_path`; the other noise-injection modes are
self-contained (the bigram table is built from the training labels at runtime).

## Evaluation

The 5-fold means are produced directly by training. To aggregate them (5-fold
mean ± std CER/WER plus parameter counts and MACs via `FlopCounterMode`):

```bash
python evaluate.py -c configs/train.yaml
```

To re-score a single trained checkpoint, edit `configs/test.yaml` (`checkpoint`,
`idx_fold`) and run `python main.py -c configs/test.yaml`.

**Direct exposure-bias probe (Table 3).** For a saved fold checkpoint, run

```bash
python eval_tf_gap.py -c <dir_work>/<fold>/<fold>/train.yaml \
    --checkpoint <dir_work>/<fold>/<fold>/checkpoints/best_cer.pth
```

to compute teacher-forced (w/ TF) vs.\ greedy decoding without teacher forcing
(w/o TF) on the same checkpoint. The script writes `eval_tf_gap.json` next to
the YAML, with per-condition CER/WER and the w/ TF → w/o TF gap. Sweeping
across all folds and training conditions reproduces Table 3 of the paper.

The paper's figures can be regenerated from the pre-aggregated numbers in
`HWRFormer.json` without re-running training:

```bash
python analysis/scripts/plot_noise_injection_modes_bars.py    # noise-injection mode bar chart
python analysis/scripts/plot_noise_injection_p_sweep.py       # p_ic sweep curves
python analysis/scripts/plot_lambda_sweep.py             # lambda_ctc sweep
```

## Repository layout

```
main.py                 # train / evaluate one fold
train_cv.py             # run all cross-validation folds sequentially
evaluate.py             # 5-fold aggregation + params/MACs
eval_tf_gap.py          # direct exposure-bias probe (w/ TF vs. w/o TF, Table 3)
onhw.ipynb              # OnHW -> MSCOCO-like dataset conversion
hwrformer/              # core library
  model/                #   1D CNN encoder, AR Transformer decoder, Transformer-CTC,
                        #   hybrid dual-head, SDPA gating
  dataset/              #   IMU loaders, augmentations, collation
  training/             #   train/eval loops (CTC, AR, hybrid, noise injection)
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
