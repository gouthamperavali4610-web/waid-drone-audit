"""
6_plot_curves.py
-----------------
Turns experiments.csv into the figures for your paper.

Makes three files in results/:
    envelope_altitude.png    accuracy vs how small the animal is
    envelope_occlusion.png   accuracy vs how covered the animal is
    envelope_combined.png    both side by side (use this one in the paper)

It also finds your BREAKDOWN THRESHOLDS -- the point where performance falls
below a usable level. Those numbers are the direct answer to your research
question, so they go in your abstract.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # draw to files instead of opening a window
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).parent
RESULTS_CSV = PROJECT_DIR / "results" / "experiments.csv"
OUT_DIR = PROJECT_DIR / "results"

# "Usable" cutoff. Below this, detection is too unreliable to trust in the
# field. 0.5 is a defensible choice -- state it explicitly in your methods
# and say it's a threshold you chose, not a law of nature.
USABLE_THRESHOLD = 0.50


def find_threshold(df, x_column, metric="mAP50"):
    """
    Find where performance crosses below the usable line.

    We walk through the results in order and find the last point that was still
    good and the first point that wasn't -- the breakdown happens between them.
    """
    ordered = df.sort_values(x_column)
    above = ordered[ordered[metric] >= USABLE_THRESHOLD]
    below = ordered[ordered[metric] < USABLE_THRESHOLD]

    if below.empty:
        return None, "never dropped below threshold in the range tested"
    if above.empty:
        return None, "already below threshold at the easiest setting"

    last_good = above.iloc[-1]
    first_bad = below.iloc[0]
    return (last_good[x_column], first_bad[x_column]), (
        f"between {last_good['level_label']} and {first_bad['level_label']}"
    )


def plot_one(ax, df, x_column, x_label, title, invert=False):
    """Draw one panel: the metrics plotted against the difficulty axis."""
    ordered = df.sort_values(x_column)

    for metric, style in [("mAP50", "o-"), ("recall", "s--"), ("precision", "^:")]:
        ax.plot(ordered[x_column], ordered[metric], style, label=metric, linewidth=2)

    # The usable-performance line
    ax.axhline(USABLE_THRESHOLD, color="red", linestyle="-.", alpha=0.6,
               label=f"usable threshold ({USABLE_THRESHOLD})")

    ax.set_xlabel(x_label)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if invert:
        # For the altitude plot: bigger animals on the left, smaller on the
        # right, so the graph reads left-to-right as "flying higher".
        ax.invert_xaxis()


def main():
    if not RESULTS_CSV.exists():
        raise SystemExit(f"No results at {RESULTS_CSV}\nRun: python 5_experiments.py")

    df = pd.read_csv(RESULTS_CSV)
    altitude = df[df["experiment"] == "altitude"]
    occlusion = df[df["experiment"] == "occlusion"]

    # --- Individual figures ------------------------------------------------
    if not altitude.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_one(ax, altitude, "median_animal_width_px",
                 "median animal width in pixels  (smaller = flown higher)",
                 "Altitude envelope", invert=True)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "envelope_altitude.png", dpi=200)
        plt.close(fig)
        print(f"[done] {OUT_DIR / 'envelope_altitude.png'}")

    if not occlusion.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_one(ax, occlusion, "level",
                 "fraction of animal covered", "Occlusion envelope")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "envelope_occlusion.png", dpi=200)
        plt.close(fig)
        print(f"[done] {OUT_DIR / 'envelope_occlusion.png'}")

    # --- Combined figure for the paper -------------------------------------
    if not altitude.empty and not occlusion.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        plot_one(axes[0], altitude, "median_animal_width_px",
                 "median animal width (px)", "A. Altitude envelope", invert=True)
        plot_one(axes[1], occlusion, "level",
                 "fraction of animal covered", "B. Occlusion envelope")
        fig.suptitle("WAID-trained detector: operating envelope", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "envelope_combined.png", dpi=200)
        plt.close(fig)
        print(f"[done] {OUT_DIR / 'envelope_combined.png'}")

    # --- The numbers that answer your research question --------------------
    print("\n" + "=" * 62)
    print("BREAKDOWN THRESHOLDS  (put these in your abstract)")
    print("=" * 62)

    if not altitude.empty:
        _, description = find_threshold(altitude, "median_animal_width_px")
        best = altitude["mAP50"].max()
        worst = altitude["mAP50"].min()
        print(f"\nALTITUDE:")
        print(f"  mAP50 range across the sweep : {worst:.3f} to {best:.3f}")
        print(f"  Falls below {USABLE_THRESHOLD}          : {description}")

    if not occlusion.empty:
        _, description = find_threshold(occlusion, "level")
        best = occlusion["mAP50"].max()
        worst = occlusion["mAP50"].min()
        print(f"\nOCCLUSION:")
        print(f"  mAP50 range across the sweep : {worst:.3f} to {best:.3f}")
        print(f"  Falls below {USABLE_THRESHOLD}          : {description}")

    # --- How to write this up ---------------------------------------------
    print("\n" + "-" * 62)
    print("WRITING IT UP:")
    print("  1. These curves are the core result. Lead with the combined figure.")
    print("  2. Occlusion is the under-studied axis -- if your occlusion curve")
    print("     drops faster than the altitude one, say so plainly. That is a")
    print("     genuine contribution, not a footnote.")
    print("  3. Then compare your REAL drone footage (script 4) against these")
    print("     curves. Does real footage behave like 'high altitude', like")
    print("     'occluded', or worse than both? That answer is your sim-to-real")
    print("     gap, and it's the part nobody has published for WAID.")
    print("  4. Say clearly that these are PROXIES for altitude and occlusion,")
    print("     not real flights. Naming your method's limits makes the work")
    print("     more credible, not less.")


if __name__ == "__main__":
    main()
