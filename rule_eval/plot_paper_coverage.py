"""Recreate the paper's final coverage-threshold and LLM-DQR comparison plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_line,
    element_rect,
    element_text,
    facet_grid,
    geom_line,
    geom_point,
    geom_vline,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_linetype_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)

DATASETS = ["Beers", "Flights", "Hospital", "PCI", "Tax"]
MODELS = ["Gemma", "GLM", "Qwen"]
PROMPT_LABELS = {
    "8": "Dirty & Clean",
    "10": "Dirty & Type",
    "11": "Dirty & Column",
    "9": "Just Dirty",
}
METRICS = ["total_coverage_agg_perc", "clean_val_agg_perc"]


def load_plot_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the exact long-form data used for the submitted paper figure."""
    data = pd.read_csv(path)
    required = {
        "promptId",
        "dataset",
        "Modelname_short",
        "variable",
        "value",
        "plot_x",
        "plot_role",
        "baseline_label",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    data["dataset"] = pd.Categorical(data["dataset"], DATASETS, ordered=True)
    data["Modelname_short"] = pd.Categorical(
        data["Modelname_short"], MODELS, ordered=True
    )
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["plot_x"] = pd.to_numeric(data["plot_x"], errors="raise")
    if data[["dataset", "Modelname_short", "variable", "value"]].isna().any().any():
        raise ValueError(f"{path} contains unknown category labels or missing values")

    lines = data[data["plot_role"] == "LeDQeR threshold sweep"].copy()
    baseline = data[data["plot_role"].str.startswith("Fixed LLM-DQR", na=False)].copy()
    lines["promptId"] = lines["promptId"].astype(int).astype(str)
    baseline["baseline_label"] = "LLM-DQR baseline"

    expected_lines = {
        (dataset, model, prompt, metric)
        for dataset in DATASETS
        for model in MODELS
        for prompt in PROMPT_LABELS
        for metric in METRICS
    }
    observed_lines = set(
        lines[["dataset", "Modelname_short", "promptId", "variable"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_lines != expected_lines:
        raise ValueError(
            "Coverage curves do not cover every dataset/model/prompt/metric "
            f"combination: missing={sorted(expected_lines - observed_lines)}"
        )

    expected_baseline = {
        (dataset, model, metric)
        for dataset in DATASETS
        for model in MODELS
        for metric in METRICS
    }
    observed_baseline = set(
        baseline[["dataset", "Modelname_short", "variable"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_baseline != expected_baseline:
        raise ValueError(
            "LLM-DQR points do not cover every dataset/model/metric combination: "
            f"missing={sorted(expected_baseline - observed_baseline)}"
        )
    return lines, baseline


def build_plot(lines: pd.DataFrame, baseline: pd.DataFrame):
    return (
        ggplot(lines, aes(x="plot_x", y="value", linetype="promptId", color="variable"))
        + geom_line(size=1.15)
        + geom_vline(xintercept=1.08, linetype="dotted", color="#888888", size=0.5)
        + geom_point(
            data=baseline,
            mapping=aes(
                x="plot_x", y="value", color="variable", shape="baseline_label"
            ),
            inherit_aes=False,
            size=3.0,
        )
        + facet_grid(cols="dataset", rows="Modelname_short")
        + theme_bw()
        + scale_x_continuous(
            limits=(0, 1.35),
            breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            labels=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "LLM-\nDQR"],
            expand=(0, 0),
        )
        + scale_y_continuous(
            limits=(0, 1.05),
            breaks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
            expand=(0.02, 0),
        )
        + scale_linetype_manual(
            values={"8": "solid", "10": "dashdot", "11": "dotted", "9": "dashed"},
            labels=PROMPT_LABELS,
        )
        + scale_color_manual(
            values={
                "total_coverage_agg_perc": "#d7191c",
                "clean_val_agg_perc": "#2c7bb6",
            },
            labels={
                "total_coverage_agg_perc": "Dirty cells",
                "clean_val_agg_perc": "Clean cells",
            },
        )
        + scale_shape_manual(values={"LLM-DQR baseline": "D"})
        + guides(
            linetype=guide_legend(
                title="LeDQeR prompt", override_aes={"color": "#d7191c"}
            ),
            color=guide_legend(title="Cells"),
            shape=guide_legend(title="Baseline"),
        )
        + labs(
            x="LeDQeR minimum precision threshold. LLM-DQR is a separate fixed baseline.",
            y="Fraction of cells flagged",
        )
        + theme(
            legend_position="top",
            axis_text_x=element_text(size=12),
            axis_text_y=element_text(size=12),
            axis_title=element_text(size=15),
            legend_text=element_text(size=13),
            legend_title=element_text(size=14),
            strip_text=element_text(size=13),
            axis_line_x=element_line(color="black", size=1.4),
            strip_background=element_rect(fill="#d3d3d3", color="#888888"),
            figure_size=(16, 8.5),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/plot_inputs/figure7_coverage_dqr_point_reference.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/figure7_coverage_dqr_point_reference.png"),
    )
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    lines, baseline = load_plot_data(arguments.input)
    build_plot(lines, baseline).save(arguments.output, dpi=300, verbose=False)
    print(f"Wrote {arguments.output}")


if __name__ == "__main__":
    main()
