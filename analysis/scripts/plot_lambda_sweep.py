#!/usr/bin/env python3
"""Generate the lambda_ctc sweep figure (paper Fig. 4).

Two-panel figure: downstream AR CER (left) and WER (right) vs lambda_ctc,
5-fold mean +/- standard error of the mean. Dashed vertical line marks the
selected lambda_ctc=0.1. A horizontal AR-only reference is drawn as a
dotted line.

Data source: trained results under results/hwr2/ when present
(train_element_word_hybrid_{01..10}_onhw_wi + the AR-only
Baseline-AR-blconv_b run); otherwise the pre-aggregated figure4 block in
HWRFormer.json, so the paper figure can be regenerated without re-running
training.

Run from anywhere:
    python plot_lambda_sweep.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, no Type3 (Springer)
import matplotlib.pyplot as plt

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
RESULTS = REPO / "results" / "hwr2"
OUT_PDF = REPO / "figures" / "lambda_sweep.pdf"

LAMBDAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SELECTED = 0.1


def read_per_fold(model_dir: Path) -> tuple[list[float], list[float]]:
    cers, wers = [], []
    for k in range(5):
        files = sorted(glob.glob(str(model_dir / f"{k}/train_*.json")))
        if not files:
            continue
        try:
            with open(files[-1]) as f:
                d = json.load(f)
            cers.append(float(d["best"]["character_error_rate"][1]) * 100.0)
            wers.append(float(d["best"]["word_error_rate"][1]) * 100.0)
        except (KeyError, ValueError, json.JSONDecodeError):
            pass
    return cers, wers


def lambda_dir(k: int) -> Path:
    return (
        RESULTS
        / f"train_element_word_hybrid_{k:02d}_onhw_wi"
        / "ar_transformer__onhw_wi_word_rh"
    )


def load_data() -> tuple[list[float], list[float], list[float], list[float], float, float]:
    """Return (cer_means, cer_sems, wer_means, wer_sems, ar_cer, ar_wer).

    5-fold means and standard error of the mean (SEM = std / sqrt(n)).
    SEM is the right error bar for "is the mean different across lambdas?"
    The between-fold std spans ~7pp (some writers are harder than others)
    and would dwarf the <1pp between-lambda signal; SEM gives the
    uncertainty in the mean estimate itself.
    """
    if RESULTS.exists():
        cer_per_lambda: list[list[float]] = []
        wer_per_lambda: list[list[float]] = []
        for lam in LAMBDAS:
            c, w = read_per_fold(lambda_dir(int(round(lam * 10))))
            cer_per_lambda.append(c)
            wer_per_lambda.append(w)
        ar_only_dir = RESULTS / "Baseline-AR-blconv_b" / "ar_transformer__onhw_wi_word_rh"
        ar_cers, ar_wers = read_per_fold(ar_only_dir)
        if ar_cers and all(len(c) > 1 for c in cer_per_lambda):
            return (
                [statistics.mean(c) for c in cer_per_lambda],
                [statistics.stdev(c) / math.sqrt(len(c)) for c in cer_per_lambda],
                [statistics.mean(w) for w in wer_per_lambda],
                [statistics.stdev(w) / math.sqrt(len(w)) for w in wer_per_lambda],
                statistics.mean(ar_cers),
                statistics.mean(ar_wers),
            )
        print(f"results under {RESULTS} incomplete; falling back to HWRFormer.json")
    json_path = REPO / "HWRFormer.json"
    d = json.load(open(json_path))
    print(f"using pre-aggregated numbers from {json_path}")
    sweep = d["figure4_lambda_sweep"]
    cer_means, cer_sems, wer_means, wer_sems = [], [], [], []
    for lam in LAMBDAS:
        e = sweep[f"lambda_{lam:.1f}"]
        n = int(e.get("n_folds", 5))
        cer_means.append(float(e["cer_mean"]))
        cer_sems.append(float(e["cer_std"]) / math.sqrt(n))
        wer_means.append(float(e["wer_mean"]))
        wer_sems.append(float(e["wer_std"]) / math.sqrt(n))
    ar_only = d["table2_interventions"]["onhw_wi_word_rh"]["HWRFormer"]
    return cer_means, cer_sems, wer_means, wer_sems, float(ar_only["cer_mean"]), float(ar_only["wer_mean"])


def main() -> None:
    cer_means, cer_sems, wer_means, wer_sems, ar_cer_mean, ar_wer_mean = load_data()

    fig, (ax_cer, ax_wer) = plt.subplots(1, 2, figsize=(9.0, 3.2), sharex=True)

    ax_cer.errorbar(LAMBDAS, cer_means, yerr=cer_sems, color="#1f77b4", linewidth=2,
                    marker="o", markersize=6, capsize=4, zorder=5)
    ax_wer.errorbar(LAMBDAS, wer_means, yerr=wer_sems, color="#ff7f0e", linewidth=2,
                    marker="s", markersize=6, capsize=4, zorder=5)

    from matplotlib.lines import Line2D

    # AR-only reference lines and selected-lambda highlight.
    ax_cer.axhline(ar_cer_mean, color="black", linestyle=":", linewidth=1.5, zorder=3)
    ax_wer.axhline(ar_wer_mean, color="black", linestyle=":", linewidth=1.5, zorder=3)

    # The selected-lambda indicator: a translucent vertical band (not a
    # thin line) so it stays visible even where the orange/blue error
    # bars cross it. A bright color is used to maximize contrast against
    # the data colors.
    HIGHLIGHT = "#9467bd"  # purple, distinct from CER blue and WER orange
    for ax in (ax_cer, ax_wer):
        ax.axvspan(SELECTED - 0.015, SELECTED + 0.015, color=HIGHLIGHT,
                   alpha=0.22, zorder=1)
        ax.axvline(SELECTED, color=HIGHLIGHT, linestyle="--", linewidth=1.6,
                   alpha=0.9, zorder=2)
        # Downward arrow on the top axis edge pointing at the selected value.
        ax.annotate(
            "", xy=(SELECTED, 1.0), xytext=(SELECTED, 1.06),
            xycoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="-|>", color=HIGHLIGHT, lw=1.8),
        )
        ax.set_xlabel(r"$\lambda_{\mathrm{ctc}}$", fontsize=12)
        ax.grid(True, alpha=0.3)

    ax_cer.set_ylabel("CER (%)", fontsize=12)
    ax_wer.set_ylabel("WER (%)", fontsize=12)
    ax_cer.set_title(r"Downstream AR CER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)
    ax_wer.set_title(r"Downstream AR WER vs $\lambda_{\mathrm{ctc}}$", fontsize=12)

    # Tight y-limits around the mean+/-SEM envelope so the small between-
    # lambda signal is actually visible.
    cer_envelope = [(m - s, m + s) for m, s in zip(cer_means, cer_sems)]
    wer_envelope = [(m - s, m + s) for m, s in zip(wer_means, wer_sems)]
    cer_lo = min(lo for lo, _ in cer_envelope + [(ar_cer_mean, ar_cer_mean)]) - 0.4
    cer_hi = max(hi for _, hi in cer_envelope + [(ar_cer_mean, ar_cer_mean)]) + 0.4
    wer_lo = min(lo for lo, _ in wer_envelope + [(ar_wer_mean, ar_wer_mean)]) - 0.4
    wer_hi = max(hi for _, hi in wer_envelope + [(ar_wer_mean, ar_wer_mean)]) + 0.4
    ax_cer.set_ylim(cer_lo, cer_hi)
    ax_wer.set_ylim(wer_lo, wer_hi)

    # Tiny inline tag at the right end of each AR-only reference line
    # showing the numeric baseline value.
    bbox_small = dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="0.7", alpha=0.95)
    ax_cer.text(0.99, ar_cer_mean, f"AR-only {ar_cer_mean:.2f}",
                transform=ax_cer.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, bbox=bbox_small, zorder=6)
    ax_wer.text(0.99, ar_wer_mean, f"AR-only {ar_wer_mean:.2f}",
                transform=ax_wer.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, bbox=bbox_small, zorder=6)

    # Single FIGURE-LEVEL legend below both panels. Three generic entries
    # describe the visual encoding; numeric AR-only values are shown
    # inline above. Proxy handles so we control exactly what shows.
    fig.legend(
        handles=[
            Line2D([0], [0], color="#1f77b4", marker="o", markersize=6,
                   linewidth=2, label="5-fold mean $\\pm$ SEM (CER)"),
            Line2D([0], [0], color="#ff7f0e", marker="s", markersize=6,
                   linewidth=2, label="5-fold mean $\\pm$ SEM (WER)"),
            Line2D([0], [0], color="black", linestyle=":", linewidth=1.5,
                   label="AR-only baseline"),
            Line2D([0], [0], color=HIGHLIGHT, linestyle="--", linewidth=1.6,
                   label=f"selected $\\lambda_{{\\mathrm{{ctc}}}}={SELECTED}$"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=4,
        fontsize=10,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.8,
    )

    # Reserve room at the bottom for the single fig-level legend.
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
