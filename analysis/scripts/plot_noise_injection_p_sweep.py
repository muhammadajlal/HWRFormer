#!/usr/bin/env python3
"""Generate corruption_p_sweep.pdf for the paper.

Per-dataset panel: CER and WER vs noise-injection rate p_ic, 5-fold mean.
CER on left axis (blue), WER on right axis (orange). Dotted vertical line
marks the recommended default p_ic=0.15. The no-noise-injection baseline is
the left-most x=0.00 point.

Four panels (2x2): OnHW-WI (small-vocabulary, writer-independent), OnHW-WD
(small-vocabulary, writer-dependent), private words (large-vocabulary,
short), and private sentences (large-vocabulary, long). Export ``REPO``
first (``export REPO=$(pwd)``) so the result paths resolve.

Run from anywhere:
    python plot_noise_injection_p_sweep.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
RESULTS = REPO / "results" / "hwr2"
OUT_PDF = REPO / "publications" / "paper2_lncs_overleaf" / "figures" / "corruption_p_sweep.pdf"

PANELS = [
    ("onhw_wi_word_rh", "OnHW-Words500 (WI)"),
    ("onhw_wd_word_rh", "OnHW-Words500 (WD)"),
    ("wi_word_hw6_meta", "Private (words)"),
    ("wi_sent_hw6_meta", "Private (sentences)"),
]

P_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
DEFAULT_P = 0.15


def read_5fold(model_dir: Path) -> tuple[float | None, float | None]:
    cers, wers = [], []
    for k in range(5):
        files = sorted(glob.glob(str(model_dir / f"{k}/train_*.json")))
        if not files:
            return None, None
        try:
            with open(files[-1]) as f:
                d = json.load(f)
            cers.append(float(d["best"]["character_error_rate"][1]) * 100.0)
            wers.append(float(d["best"]["word_error_rate"][1]) * 100.0)
        except (KeyError, ValueError, json.JSONDecodeError):
            return None, None
    if len(cers) != 5:
        return None, None
    return statistics.mean(cers), statistics.mean(wers)


def cell_path(dataset_key: str, p: float) -> Path:
    if p == 0.00:
        return RESULTS / "Baseline-AR-blconv_b" / f"ar_transformer__{dataset_key}"
    if abs(p - 0.15) < 1e-9:
        return RESULTS / "Baseline-AR-NoiseInjection-uniform" / f"ar_transformer__{dataset_key}"
    pstr = f"p0p{int(round(p * 100)):02d}"
    return (
        RESULTS
        / "Baseline-AR-NoiseInjection-Sweep-blconv_b"
        / f"ar_transformer__{dataset_key}__{pstr}"
    )


def gather(dataset_key: str) -> tuple[list[float], list[float], list[float]]:
    """Return (p_values, cers, wers) for the points that have 5-fold data."""
    ps, cers, wers = [], [], []
    for p in P_VALUES:
        cer, wer = read_5fold(cell_path(dataset_key, p))
        if cer is not None and wer is not None:
            ps.append(p)
            cers.append(cer)
            wers.append(wer)
    return ps, cers, wers


# Grid layout chosen by how many datasets actually have data present. This lets
# a public reproducer without the private subsets get a clean OnHW-only figure
# instead of empty private panels; the full four-dataset run yields the 2x2.
_LAYOUT = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2)}
_FIGSIZE = {(1, 1): (5.0, 3.2), (1, 2): (9.5, 3.2), (1, 3): (11.5, 3.2), (2, 2): (9.5, 6.0)}


def main() -> None:
    panels = []
    for dataset_key, label in PANELS:
        ps, cers, wers = gather(dataset_key)
        if len(ps) >= 2:
            panels.append((label, ps, cers, wers))
        else:
            print(f"skip {label}: only {len(ps)} sweep point(s) found under {RESULTS}")

    if not panels:
        print(f"no datasets with >=2 sweep points found under {RESULTS}; nothing to plot")
        return

    n = len(panels)
    nrows, ncols = _LAYOUT.get(n, (2, 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=_FIGSIZE[(nrows, ncols)], squeeze=False)
    axs = list(axes.flat)
    legend_handles = None

    for ax, (label, ps, cers, wers) in zip(axs, panels):
        cer_color = "#1f77b4"  # blue
        wer_color = "#ff7f0e"  # orange

        l1 = ax.plot(ps, cers, marker="o", color=cer_color, linewidth=2, label="CER")
        ax.set_ylabel("CER (%)", color=cer_color)
        ax.tick_params(axis="y", labelcolor=cer_color)
        ax.set_xlabel(r"$p_{\mathrm{ni}}$")
        ax.set_title(label, fontsize=11)
        ax.axvline(DEFAULT_P, linestyle=":", color="gray", linewidth=1.2)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        l2 = ax2.plot(ps, wers, marker="s", color=wer_color, linewidth=2, linestyle="--", label="WER")
        ax2.set_ylabel("WER (%)", color=wer_color)
        ax2.tick_params(axis="y", labelcolor=wer_color)

        if legend_handles is None:
            legend_handles = l1 + l2

    # Hide any unused axes (e.g. 3 datasets in a 2x2 fallback).
    for ax in axs[n:]:
        ax.set_visible(False)

    fig.legend(
        legend_handles,
        ["CER (left axis)", "WER (right axis)"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"saved: {OUT_PDF} ({n} panel{'s' if n != 1 else ''})")


if __name__ == "__main__":
    main()
