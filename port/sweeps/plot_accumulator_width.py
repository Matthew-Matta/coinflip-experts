"""Chart for the accumulator-width sweep (EXPLORATORY): route flips and routediff detection vs accumulator mantissa bits.

Top: tokens routed differently from fp64 (both layers pooled) for the accumulators that can flip routes.
Bottom: per accumulator and width, whether calibrated routediff flags it on both layers (filled), one layer (half),
or neither (empty); a cross marks a width where it flagged but localised to a different op.

  python -m port.sweeps.plot_accumulator_width data/results/accumulator-width/accumulator_width.parquet \
      --out findings/accumulator-width/flips_vs_bits.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

FIRST_OP = {"oproj": "o_proj_out", "rms": "router_input", "router": "router_logits", "expert": "expert_gate",
            "combine": "output"}
LABELS = {"oproj": "o_proj", "rms": "RMSNorm variance", "router": "router", "expert": "expert matmuls",
          "combine": "combine"}
SERIES = {"oproj": "#2a78d6", "router": "#eb6834"}  # validated pair (dataviz reference palette, light)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    df = pd.read_parquet(a.parquet)
    df = df[(df.split == "all") & df.variant.str.startswith("e8m")].copy()
    df["m"] = df.variant.str[3:].astype(int)
    tokens = int(df.groupby("layer").tokens.first().sum())

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.2, 5.4), dpi=160, sharex=True,
                                   gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.18})
    flips = df.groupby(["knob", "m"]).flips_vs_oracle.sum()
    for knob, color in SERIES.items():
        s = flips[knob].sort_index()
        pct = 100 * s / tokens
        top.plot(s.index, pct.where(s > 0), color=color, lw=2, marker="o", ms=4.5,
                 markeredgecolor=SURFACE, markeredgewidth=1, label=LABELS[knob] + " accumulator")
    zero_from = {k: int(flips[k][flips[k].index > flips[k][flips[k] > 0].index.max()].index.min()) for k in SERIES}
    top.text(0.99, 0.50, "0 flips on either layer from " + ", ".join(f"{zero_from[k]} bits ({LABELS[k]})" for k in SERIES),
             transform=top.transAxes, ha="right", fontsize=7, color=MUTED)
    top.set_yscale("log")
    top.set_ylim(0.02, 30)
    top.set_ylabel(f"tokens routed differently\nfrom fp64 (% of {tokens})")
    top.grid(color=GRID, lw=0.6, which="major")
    top.legend(frameon=False, fontsize=7.5, loc="upper right")
    top.text(0.99, 0.62, "RMSNorm, expert and combine accumulators:\n0 flips at every width (not plotted)",
             transform=top.transAxes, ha="right", fontsize=7, color=MUTED)
    top.set_title("OLMoE-1B-7B layers 0 + 7, one accumulator narrowed at a time (exploratory)", fontsize=8.5,
                  color=INK, loc="left")
    for side in ("top", "right"):
        top.spines[side].set_visible(False)
        bot.spines[side].set_visible(False)

    knobs = list(LABELS)
    for i, knob in enumerate(knobs):
        g = df[df.knob == knob]
        for m, gm in g.groupby("m"):
            n_flag = int(gm.calibrated_flagged.sum())
            wrong = bool(((gm.calibrated_localized_op.notna()) & (gm.calibrated_localized_op != FIRST_OP[knob])).any()
                         | (gm.calibrated_flagged & gm.calibrated_patch_checked & gm.calibrated_localized_op.isna()).any())
            y = len(knobs) - 1 - i
            if n_flag == 2:
                bot.plot(m, y, "o", ms=6, color=INK)
            elif n_flag == 1:
                bot.plot(m, y, "o", ms=6, color=INK, fillstyle="left", markerfacecoloralt=SURFACE)
            else:
                bot.plot(m, y, "o", ms=6, color=MUTED, fillstyle="none", markeredgewidth=0.8)
            if wrong:
                bot.plot(m, y + 0.32, "x", ms=4, color=MUTED, markeredgewidth=1)
    bot.set_yticks(range(len(knobs)))
    bot.set_yticklabels([LABELS[k] for k in reversed(knobs)])
    bot.set_ylim(-0.6, len(knobs) - 0.3)
    bot.set_xticks(range(7, 24, 2))
    bot.set_xlabel("accumulator mantissa bits (8-bit exponent; 7 = bf16, 23 = fp32)")
    bot.set_title("routediff, calibrated (frozen real-v2 tolerances)\n● flags on both layers   ◐ one layer   ○ neither"
                  "   × localised to a downstream op", fontsize=7, color=MUTED, loc="left")
    bot.grid(axis="x", color=GRID, lw=0.6)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.94, bottom=0.09)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(a.out.with_suffix(".svg"))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
