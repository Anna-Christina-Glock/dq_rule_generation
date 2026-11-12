"""Recreate the paper's Step 2 correctness and Step 3 coverage-recall plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_rect,
    element_text,
    facet_grid,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_manual,
    scale_shape_manual,
    scale_x_continuous,
    scale_y_discrete,
    theme,
    theme_bw,
)

PROMPTS = ["Dirty & Clean", "Dirty & Type", "Dirty & Column", "Just Dirty"]
DATASETS = ["Beers", "Flights", "Hospital", "PCI", "Tax"]
ERROR_ORDER = ["con", "dmv", "dov", "ebv", "emv", "ics", "ifo", "mfv", "spm"]
MODEL_ORDER = ["Gemma", "GLM", "Qwen"]
MODEL_COLORS = {"Gemma": "#1b9e77", "GLM": "#d95f02", "Qwen": "#7570b3"}
MODEL_MARKERS = {"Gemma": "s", "GLM": "^", "Qwen": "x"}


def load_plot_data(path: Path) -> pd.DataFrame:
    """Load and validate one committed paper plot-data table."""
    data = pd.read_csv(path)
    required = {"prompt_label", "Modelname_short", "datasetName", "error_code", "value"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    data["value"] = pd.to_numeric(data["value"], errors="raise")
    data["prompt_label"] = pd.Categorical(
        data["prompt_label"], categories=PROMPTS, ordered=True
    )
    class_counts = data.groupby("datasetName", observed=True)["error_code"].nunique()
    dataset_order = sorted(
        DATASETS,
        key=lambda dataset: (
            -int(class_counts.get(dataset, 0)),
            DATASETS.index(dataset),
        ),
    )
    data["datasetName"] = pd.Categorical(
        data["datasetName"], categories=dataset_order, ordered=True
    )
    data["error_code"] = pd.Categorical(
        data["error_code"], categories=ERROR_ORDER, ordered=True
    )
    data["Modelname_short"] = pd.Categorical(
        data["Modelname_short"], categories=MODEL_ORDER, ordered=True
    )
    if data[list(required)].isna().any().any():
        raise ValueError(f"{path} contains unknown category labels or missing values")
    return data


def build_plot(data: pd.DataFrame, x_label: str):
    dataset_order = list(data["datasetName"].cat.categories)
    class_counts = data.groupby("datasetName", observed=True)["error_code"].nunique()
    row_ratios = [max(float(class_counts.get(dataset, 0)), 1.5) for dataset in dataset_order]
    return (
        ggplot(
            data,
            aes(
                x="value",
                y="error_code",
                color="Modelname_short",
                shape="Modelname_short",
            ),
        )
        + geom_point(size=4.0, stroke=1.4, fill="white")
        + facet_grid(
            rows="datasetName",
            cols="prompt_label",
            scales="free_y",
            space={"x": [1] * len(PROMPTS), "y": row_ratios},
        )
        + scale_x_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1])
        + scale_y_discrete(drop=True)
        + scale_color_manual(values=MODEL_COLORS, limits=MODEL_ORDER)
        + scale_shape_manual(values=MODEL_MARKERS, limits=MODEL_ORDER)
        + labs(x=x_label, y="Error Type", color="Model", shape="Model")
        + guides(color=guide_legend(nrow=1), shape=guide_legend(nrow=1))
        + theme_bw()
        + theme(
            figure_size=(16, 7.6),
            panel_spacing_y=0.003,
            legend_position="top",
            axis_text_x=element_text(size=12),
            axis_text_y=element_text(size=12),
            axis_title=element_text(size=15),
            legend_text=element_text(size=13),
            legend_title=element_text(size=14),
            strip_text=element_text(size=13),
            strip_text_y=element_text(size=13, rotation=0),
            strip_background=element_rect(fill="#d3d3d3", color="#888888"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/plot_inputs"),
        help="Directory containing the committed Step 2/3 plot-data CSVs",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/plots"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    plots = (
        ("step2_f1_paper_style", "F1"),
        ("step3_recall_paper_style", "Recall"),
    )
    for stem, label in plots:
        data = load_plot_data(arguments.input_dir / f"{stem}.csv")
        output = arguments.output_dir / f"{stem}.png"
        build_plot(data, label).save(output, dpi=200, verbose=False)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
