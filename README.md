# LeDQeR: LLM-Based Data-Quality Rule Generation

This repository contains the code used to generate and evaluate the Great Expectations (GX) rules reported in the LeDQeR paper. LeDQeR generates candidate rules from dirty tuples and processes them through four stages:

1. syntactic filtering removes responses that cannot be parsed or executed;
2. correctness filtering retains rules that detect the reported error in the prompting tuple;
3. coverage filtering can remove rules whose recall on additional matching dirty tuples is below a configured threshold; and
4. greedy redundancy reduction selects a compact rule set at a configurable precision threshold.

## Repository layout

```text
dq_rule_generation/
├── data/in/                         labeled benchmark data
├── data/plot_inputs/                committed data behind the paper plots
├── data/results_gen/                generated LLM responses (ignored by Git)
├── data/results_eval/               evaluation outputs (ignored by Git)
├── rule_gen/
│   ├── README.md                   generation parameter guide
│   ├── config/
│   │   ├── connection.json          local LLM endpoint and model identifiers
│   │   └── parameter.json           datasets, prompt variants, and concurrency
│   ├── main_server_paralell.py      parallel rule-generation entry point
│   ├── ruleGenPrompt.py             paper prompt variants
│   ├── adaptiveConcurrency.py       TTFT-based concurrency controller
│   ├── llmUtils.py                  model-client dispatch
│   ├── openAIModels.py              OpenAI-compatible/vLLM client
│   └── ollamaModels.py              Ollama client
├── rule_eval/
│   ├── README.md                   evaluation parameter guide
│   ├── parameter.json               evaluation configuration
│   ├── main.py                      evaluation pipeline entry point
│   ├── evaluate_rules.py            parsing and source/coverage evaluation
│   ├── parallel_eval.py             windowed held-out-data evaluation
│   ├── prepare_selection_data.py    conversion for redundancy reduction
│   ├── paired_coverage.py            paired A/B coverage analysis
│   ├── rule_selection.py            greedy redundancy reduction
│   ├── plot_paper_step23.py         paper Step 2/3 figures
│   └── plot_paper_coverage.py       paper coverage figure
├── r_plots/                         additional plotting scripts
└── requirements.txt
```

The filename `main_server_paralell.py` is retained because it is the script used for the reported experiments.

## Requirements

- Python 3.12 or newer
- a local Ollama server or an OpenAI-compatible endpoint such as vLLM
- the packages in `requirements.txt`

Install the Python dependencies from the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The repository includes the labeled Beers, Flights, Hospital, and Tax datasets used by the scripts. Their original clean and dirty versions come from the [Raha datasets](https://github.com/BigDaMa/raha/tree/master/datasets). The synthetic PCI dataset is generated with the [PCI LLM Toolkit](https://github.com/Anna-Christina-Glock/pci-llm-toolkit/tree/main/data-generator) and is not duplicated here.

The rule-generation configuration includes PCI as the optional `pci` dataset key. To run it, generate or obtain the PCI dirty and clean CSVs and place them at the paths named by `Files.pci` in `rule_gen/config/parameter.json`. Then add `"pci"` to `Public.File.Input`. It is not selected by default because those synthetic files are not included in the repository.

Every dirty input CSV must contain:

- a unique `pk` column;
- an `isDirty` indicator;
- `errCol`, containing the affected column or columns; and
- `errRule`, containing the error-type label used for evaluation.

## 1. Generate candidate rules

### Configure the model endpoint

Edit `rule_gen/config/connection.json`. The committed file contains safe local placeholders for the three models used in the paper:

- `cyankiwi/GLM-4.7-Flash-AWQ-4bit`;
- `RedHatAI/gemma-4-31B-it-FP8-Dynamic`; and
- `Qwen/Qwen3.8-27B-FP8`.

Set `base_url` to the OpenAI-compatible endpoint serving each model and set `api_key` if the endpoint requires one. Do not commit credentials.

### Configure the experiment

`rule_gen/config/parameter.json` contains only parameters used by the current runner. Important keys are:

| Key | Meaning |
|---|---|
| `modelConfigs` | entries from `connection.json` to run |
| `promptIds` | input variants: `9` Just Dirty, `11` Dirty & Column, `10` Dirty & Type, `8` Dirty & Clean |
| `expectationLiIds` | `0` supplies expectation names; `2` additionally supplies GX parameter information |
| `maxRows` | optional global row limit; `null` processes the complete configured input |
| `adaptive` | TTFT-based concurrency limits |
| `Public.File.Input` | dataset keys selected from `Files` |
| `Files` | dirty/clean filenames and prompt metadata |

Run generation from the repository root:

```bash
python rule_gen/main_server_paralell.py
```

Each output CSV in `data/results_gen/` contains the source tuple, prompt, model answer, model and prompt identifiers, errors, TTFT, total request time, output-token count when reported by the server, and observed concurrency.

Edit `rule_eval/parameter.json`. The checked-in values are placeholders and must be replaced with your file names and schema width before running. The flat configuration uses `dataFiles`, `dataFilesBig`, and `allColumnCount`. For datasets with different schemas or multiple test files, use `dataSets` instead. Each entry has `name`, `generationResult`, `evaluationData` (CSV paths relative to `dataIn`), and `allColumnCount`.

For the paper's PCI evaluation, the held-out file stem is
`trainGenData_adressPerson_dirtyPol_whole_1000000_25_1` and its source schema
has 14 columns. For example, if the generated response CSVs are under
`data/results_gen/` and the held-out PCI CSV is under `data/in/`, set
`dataIn` to `data` and configure:

```json
{
  "dataSets": [
    {
      "name": "pci",
      "generationResult": "results_gen/<generated-PCI-response-file-stem>",
      "evaluationData": ["in/trainGenData_adressPerson_dirtyPol_whole_1000000_25_1"],
      "allColumnCount": 14
    }
  ]
}
```

Use the generated response filename stem corresponding to the model, prompt
variant, and expectation-list setting being evaluated.

Run from the repository root:

```bash
python -m rule_eval.main --config rule_eval/parameter.json
```

The modular evaluator extracts model JSON in `evaluate_rules.py`, constructs
GX suites, records unparseable or non-executable rules, computes source-tuple
precision/recall, and evaluates recall on additional matching dirty tuples.
`parallel_eval.py` applies the selected suites to held-out data in row windows,
including typed numeric/date handling for GX `between` expectations. Outputs
are written to the configured `dataOut` and `suiteOut` directories.

## 3. Greedy redundancy reduction

`rule_eval/prepare_selection_data.py` converts evaluator output into the rule/coverage tables consumed by `rule_eval/rule_selection.py`. The primary functions are:

```python
from rule_eval.rule_selection import (
    get_coverage_for_precision_levels,
    greedy_rule_selection_pairs,
)

selected_rules, coverage_by_rule, covered_cells = greedy_rule_selection_pairs(
    df_pairs,
    df_rules,
    precision_required=1.0,
)
```

`precision_required` controls the per-rule precision threshold. The selection is greedy: each iteration chooses the remaining rule that covers the most previously uncovered dirty cells.

## Plotting

`data/plot_inputs/` contains the exact long-form data used to render the three
paper figures. Recreate the Step 2 correctness and Step 3 coverage-recall
figures with:

```bash
python -m rule_eval.plot_paper_step23 \
  --input-dir data/plot_inputs \
  --output-dir data/plots
```

Recreate the final coverage-threshold/LLM-DQR figure with:

```bash
python -m rule_eval.plot_paper_coverage \
  --input data/plot_inputs/figure7_coverage_dqr_point_reference.csv \
  --output data/plots/figure7_coverage_dqr_point_reference.png
```

The scripts under `r_plots/` provide additional R-based analyses.

## Reproducibility notes

- Generation uses one LLM request per dirty tuple and prompt configuration.
- The filtering stages do not invoke the LLM again; they parse and execute GX rules and perform deterministic selection.
- `rule_gen/config/connection.json` contains no private endpoint or credential values.
- Generated responses, evaluation outputs, logs, caches, and local environments are excluded by `.gitignore`.
