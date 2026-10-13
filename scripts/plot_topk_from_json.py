#!/usr/bin/env python3
"""Draw two Top-K panels from a small JSON file, without model inference."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ("true_target_at_k", "query_matching_share_at_k")
COLORS = ("#3975AD", "#CB792F", "#19816A", "#B44D68", "#767676")
MARKERS = ("o", "s", "^", "D", "v")

# -----------------------------
# Figure appearance parameters
# -----------------------------
TITLE_FONTSIZE = 12
PANEL_LABEL_FONTSIZE = 12   # (a), (b) 的字体大小
PANEL_LABEL_Y = -0.24       # (a), (b) 的垂直位置；越小越往下


def validate(data):
    ks = data.get("k", [])

    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("k must be a nonempty list of positive integers")

    if ks != sorted(set(ks)):
        raise ValueError("k must be strictly increasing without duplicates")

    series = data.get("series", [])

    if not series:
        raise ValueError("series must contain at least one method")

    names = set()

    for item in series:
        name = item.get("name")

        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Each series requires a nonempty unique name")

        names.add(name)

        for field in METRICS:
            values = item.get(field, [])

            if len(values) != len(ks):
                raise ValueError(
                    f"{name}: {field} must have one value per K"
                )

            if any(
                type(v) not in (int, float)
                or not math.isfinite(v)
                or not 0 <= v <= 100
                for v in values
            ):
                raise ValueError(
                    f"{name}: {field} must contain finite percentages in [0, 100]"
                )


def plot(data, output):
    validate(data)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Create two subplots
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.2),
    )

    # Titles no longer contain (a) and (b)
    titles = (
        "True Target@K",
        "Query-Matching Share@K",
    )

    # Panel labels shown below each subplot
    panel_labels = (
        "(a)",
        "(b)",
    )

    ylabels = (
        "Queries with target (%)",
        "Matching proposals (%)",
    )

    # -----------------------------
    # Plot curves
    # -----------------------------
    for index, series in enumerate(data["series"]):
        for ax, field in zip(axes, METRICS):
            ax.plot(
                data["k"],
                series[field],
                label=series["name"],
                color=COLORS[index % len(COLORS)],
                marker=MARKERS[index % len(MARKERS)],
                linewidth=2,
                markersize=5,
            )

    # -----------------------------
    # Configure subplots
    # -----------------------------
    for ax, title, panel_label, ylabel in zip(
        axes,
        titles,
        panel_labels,
        ylabels,
    ):
        # Title above subplot
        ax.set_title(
            title,
            fontsize=TITLE_FONTSIZE,
            pad=10,
        )

        ax.set_xlabel("Number of Top-K Objects")
        ax.set_ylabel(ylabel)

        ax.set_xticks(data["k"])

        ax.spines[["top", "right"]].set_visible(False)

        ax.grid(
            axis="y",
            color="#E4E7EB",
            linewidth=0.7,
        )

        ax.set_axisbelow(True)

        # -----------------------------
        # Add (a) / (b) below subplot
        # -----------------------------
        ax.text(
            0.5,
            PANEL_LABEL_Y,
            panel_label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=PANEL_LABEL_FONTSIZE,
        )

    # Y-axis ranges
    axes[0].set_ylim(15, 85)
    axes[1].set_ylim(40, 85)

    # -----------------------------
    # Shared legend
    # -----------------------------
    handles, labels = axes[0].get_legend_handles_labels()

    legend_columns = min(3, len(labels))
    legend_rows = math.ceil(len(labels) / legend_columns)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=legend_columns,
        frameon=False,
    )

    # -----------------------------
    # Optional overall title
    # -----------------------------
    if data.get("title"):
        fig.suptitle(
            data["title"],
            fontsize=11,
            y=0.995,
        )

    # -----------------------------
    # Optional note
    # -----------------------------
    if data.get("note"):
        fig.text(
            0.5,
            0.025,
            data["note"],
            ha="center",
            fontsize=9,
            color="#5C6269",
        )

    # More bottom space is needed for (a) and (b)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.78 - 0.06 * (legend_rows - 1),
        bottom=0.25,
        wspace=0.30,
    )

    # -----------------------------
    # Save figures
    # -----------------------------
    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for suffix in ("png", "pdf"):
        path = output / f"stage1_topk_preservation.{suffix}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        print(path.resolve())

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSON with K and measured metric percentages",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for PNG and PDF figures",
    )

    args = parser.parse_args()

    data = json.loads(
        args.input.read_text(
            encoding="utf-8"
        )
    )

    plot(
        data,
        args.output,
    )


if __name__ == "__main__":
    main()


#     conda activate m3drefclip
# cd /path/to/ambirefer3d_artifact

# python scripts/plot_topk_from_json.py \
#   --input output/Multi3DRefer/topk_paper_preview_diagnostic_v1/topk_plot_input.json \
#   --output output/Multi3DRefer/topk_paper_preview_diagnostic_v1/plot_only