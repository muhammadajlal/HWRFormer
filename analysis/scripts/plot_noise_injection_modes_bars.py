#!/usr/bin/env python3
"""Generate the noise-injection mode bar chart (paper Fig. 2).

Grouped bar chart: one cluster per dataset. Each cluster uses a fixed
left-to-right order: baseline, uniform, bigram-right, bigram-left,
self-confusion, adjacent-swap. The fixed order keeps the mode comparison
visually stable across datasets, and the no-noise-injection baseline is the
leftmost (grey) bar in every cluster.

Data source: trained results under results/hwr2/ when present and complete
(Baseline-AR-NoiseInjection-* plus the no-noise Baseline-AR-blconv_b run,
5-fold mean CER); otherwise the pre-aggregated numbers in HWRFormer.json,
so the paper figure can be regenerated without re-running training.

Run from anywhere:
    python plot_noise_injection_modes_bars.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType, no Type3 (Springer)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[2]))
RESULTS_ROOT = REPO / "results" / "hwr2"
OUT_PDF = REPO / "figures" / "noise_injection_modes_bars.pdf"

DATASETS: list[tuple[str, str]] = [
    ("onhw_wi_word_rh",  "OnHW-WI"),
    ("onhw_wd_word_rh",  "OnHW-WD"),
    ("wi_word_hw6_meta", "Priv-words"),
    ("wi_sent_hw6_meta", "Priv-sent"),
]

# (key, display label, colour)
MODES: list[tuple[str, str, str]] = [
    ("uniform",      "uniform",        "#1f77b4"),
    ("bigramright",  "bigram-right",   "#2ca02c"),
    ("bigramleft",   "bigram-left",    "#ff7f0e"),
    ("selfconf",     "self-confusion", "#9467bd"),
    ("adjacentswap", "adjacent-swap",  "#d62728"),
]

# Fixed left-to-right series: no-noise baseline first, then the five modes.
SERIES: list[tuple[str, str, str]] = [
    ("__baseline__", "baseline", "#7f7f7f"),
    *MODES,
]


def mode_dir(mode_key: str, dataset_key: str) -> Path:
    return (
        RESULTS_ROOT
        / f"Baseline-AR-NoiseInjection-{mode_key}"
        / f"ar_transformer__{dataset_key}"
    )


def baseline_dir(dataset_key: str) -> Path:
    return RESULTS_ROOT / "Baseline-AR-blconv_b" / f"ar_transformer__{dataset_key}"


def read_5fold_cer(model_dir: Path) -> float | None:
    cers: list[float] = []
    for k in range(5):
        files = sorted(glob.glob(str(model_dir / f"{k}/train_*.json")))
        if not files:
            return None
        try:
            with open(files[-1]) as f:
                d = json.load(f)
            cers.append(float(d["best"]["character_error_rate"][1]) * 100.0)
        except (KeyError, ValueError, json.JSONDecodeError):
            return None
    return float(np.mean(cers))


def load_data() -> dict[str, dict[str, float | None]]:
    """5-fold mean CER per dataset and mode: from results/ if complete,
    otherwise from the pre-aggregated figure2 block in HWRFormer.json."""
    data: dict[str, dict[str, float | None]] = {}
    if RESULTS_ROOT.exists():
        complete = True
        for ds_key, _ in DATASETS:
            row: dict[str, float | None] = {}
            row["__baseline__"] = read_5fold_cer(baseline_dir(ds_key))
            for mode_key, _, _ in MODES:
                row[mode_key] = read_5fold_cer(mode_dir(mode_key, ds_key))
            data[ds_key] = row
            if any(v is None for v in row.values()):
                complete = False
        if complete:
            return data
        print(f"results under {RESULTS_ROOT} incomplete; falling back to HWRFormer.json")
    json_path = REPO / "HWRFormer.json"
    block = json.load(open(json_path))["figure2_noise_modes"]
    print(f"using pre-aggregated numbers from {json_path}")
    for ds_key, _ in DATASETS:
        entry = block[ds_key]
        row = {"__baseline__": float(entry["baseline"])}
        for mode_key, _, _ in MODES:
            row[mode_key] = float(entry[mode_key])
        data[ds_key] = row
    return data


def main() -> None:
    data = load_data()

    fig, ax = plt.subplots(figsize=(11.2, 3.5))
    group_w = 0.90
    bar_w = group_w / len(SERIES)

    all_cers: list[float] = []
    for row in data.values():
        all_cers.extend(v for v in row.values() if v is not None)
    ymax = max(all_cers) * 1.15 if all_cers else 1.0

    x = np.arange(len(DATASETS))
    for ds_idx, (ds_key, _ds_label) in enumerate(DATASETS):
        row = data[ds_key]
        for j, (key, _label, color) in enumerate(SERIES):
            cer = row.get(key)
            if cer is None:
                continue
            pos = x[ds_idx] - group_w / 2 + (j + 0.5) * bar_w
            ax.bar(
                pos,
                cer,
                bar_w * 0.92,
                color=color,
                edgecolor="black",
                linewidth=0.45,
            )

    handles = [Patch(facecolor=c, edgecolor="black", label=lab) for _, lab, c in SERIES]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=10.5,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in DATASETS], fontsize=11.5)
    ax.set_ylabel("CER (%)", fontsize=12.0)
    ax.set_ylim(0, ymax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    print(f"saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
