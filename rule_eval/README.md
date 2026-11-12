# Rule-evaluation runner

`main.py` evaluates generated GX rules on their source tuples and, optionally, on held-out data. It writes evaluation summaries, GX suites, and inputs for greedy redundancy reduction.

## Parameters to configure

Edit `parameter.json` before running. Its checked-in file names are placeholders.

### Required inputs and outputs

| Key | What to set |
|---|---|
| `dataIn` | Root directory for the generation-result CSVs and held-out CSVs. All configured paths are resolved relative to this directory. |
| `dataFiles` | In flat mode, generation-result filename stems, without `.csv`. |
| `dataFilesBig` | In flat mode, held-out filename stems, without `.csv`. If its length equals `dataFiles`, the files are paired by position. Otherwise each generation result is evaluated against every listed held-out file. |
| `allColumnCount` | Number of data columns at the start of each generation-result CSV, before evaluator metadata columns. Must be a positive integer. |
| `dataOut` | Directory for result CSV/JSON/pickle files and selection tables. |
| `suiteOut` | Directory for serialized GX suites. |

For datasets with different schema widths or multiple held-out datasets, use `dataSets` instead of `dataFiles`, `dataFilesBig`, and the top-level `allColumnCount`. Each entry has `name`, `generationResult`, `evaluationData` (one or more paths relative to `dataIn`), and `allColumnCount`.

```json
{
  "dataIn": "data",
  "dataSets": [
    {
      "name": "pci",
      "generationResult": "results_gen/<generated-PCI-response-file-stem>",
      "evaluationData": [
        "in/trainGenData_adressPerson_dirtyPol_whole_1000000_25_1"
      ],
      "allColumnCount": 14
    }
  ],
  "suiteOut": "data/results_eval/suite_out",
  "dataOut": "data/results_eval",
  "useOnlyRelevantRules": true,
  "onlyRunBig": false,
  "isRunBigTest": true,
  "windowSize": 100000,
  "suiteWindowSize": 1,
  "maxWorkers": 1,
  "regexPrecheckRows": 200,
  "regexPrecheckSeconds": 0.01
}
```

Replace the response placeholder with the actual generation-result stem for the model, prompt variant, and GX-information setting. PCI data are not included in this repository. Generate or obtain the held-out file before running this configuration.

### Pipeline options

| Key | What to set |
|---|---|
| `useOnlyRelevantRules` | `true` selects the relevant suite for held-out evaluation; `false` selects the complete parsed suite. |
| `onlyRunBig` | `true` skips source-tuple evaluation and loads the previously saved suite from `suiteOut`. |
| `isRunBigTest` | `true` applies the chosen suite to the configured held-out data after source-tuple evaluation. |
| `windowSize` | Number of held-out rows per validation window. |
| `suiteWindowSize` | Number of expectations per GX suite partition. `1` isolates rule failures and is the default. |
| `maxWorkers` | Kept for configuration compatibility. GX dataframe validation is serial; values other than `1` are clamped to `1`. |
| `regexPrecheckRows` | Number of values sampled when prechecking regex rules. |
| `regexPrecheckSeconds` | Time budget for a regex precheck; rules exceeding it are skipped for held-out validation. |

`allColumnCount: null` and the angle-bracket file names in the checked-in `parameter.json` are placeholders. Replace them before running. `onlyRunBig` requires the matching suite JSON to already exist in `suiteOut`.

## Optimized Docker evaluator

`rule_eval_all_optimized_docker.py` is a standalone legacy runner, separate
from the configurable `main.py` pipeline. With no paired dataset variables
set, it evaluates the selected dataset, processes saved GX validation results,
and writes its legacy metric, SQLite, and coverage outputs.

Configure it with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `PATTERN_NAME` | Dataset processed by the legacy mode (`hospital`, `flights`, `beers`, `tax`, or `pci`) | `pci` |
| `DATA_IN` | Directory containing the dataset CSV | `/data` |
| `DATA_OUT` | Directory for evaluation outputs | `/config` |
| `RULE_RESULTS_PATH` | Directory containing saved GX validation-result files | `/config` |
| `USE_GLOB` | Use generated filename glob patterns when `true` | `true` |
| `PATTERN_LI` | Optional list of filename patterns; used when glob matching is disabled | — |

When `USE_GLOB=true`, the runner generates patterns for the selected dataset;
this takes precedence over `PATTERN_LI`. In legacy mode with glob matching
disabled, a supplied `PATTERN_LI` overrides the built-in dataset patterns.
Paired mode requires `PATTERN_LI` when glob matching is disabled.

`tax` forces glob matching on. In paired mode, `tax_gemma4` and `tax_glm` do
as well.

Run the script in the Docker environment with its data and output mounts
configured for the selected paths:

```bash
python rule_eval/rule_eval_all_optimized_docker.py
```

For the script's paired A/B mode, set both `REDUCTION_DATA_FILE` and
`COVERAGE_DATA_FILE` to dataset filenames or paths. Rule selection uses the
reduction dataset (A); coverage is measured on the coverage dataset (B).
The mode uses the same `PATTERN_NAME`, `DATA_IN`, `DATA_OUT`, and
`RULE_RESULTS_PATH` settings. If only one of the two dataset variables is set,
the script exits with an error.

The separate `paired_coverage.py` CLI below implements this analysis outside
the Docker runner. Prefer that CLI for a direct command-line run.

## Paired coverage analysis

`paired_coverage.py` implements the A/B analysis from
`rule_eval_all_optimized_docker.py` as a separate module and CLI.
It selects rules by greedy dirty-cell coverage on labeled dataset A, using
each rule's precision on A, then measures the selected rules' dirty-cell
coverage and clean prediction count on dataset B. It writes one threshold
table per available setting. Result files must be the saved GX validation
outputs for both data files and must use the paper's
`BigName_<dataset-stem>_ValiBigOut_..._SizeN.txt` naming convention. The
`clean_val` column counts distinct clean row/column-set pairs, following the
source analysis.

Run from the repository root:

```bash
python -m rule_eval.paired_coverage \
  --pattern-name pci \
  --data-in data/in \
  --results-dir data/results_eval \
  --data-out data/results_eval/paired_coverage \
  --reduction-data-file trainGenData_adressPerson_dirtyPol_whole_6000_100_1 \
  --coverage-data-file trainGenData_adressPerson_dirtyPol_whole_1000000_25_1
```

`--pattern-name` selects the dataset-specific result-file prefix (`tax_gemma4`
and `tax_glm` are also supported). A and B CSV
files must contain `isDirty` and `errCol`, and validation-result files must be
present for matching rule definitions in both datasets. By default the runner
checks setting suffixes `9_0`, `10_0`, `8_0`, `10_2`, `8_2`, `9_2`, `11_0`, and
`11_2`; pass `--settings` followed by one or more suffixes to limit the run.
Each output CSV contains results for minimum precision thresholds from 0.00
through 1.00 in increments of 0.05. This is separate from the standard
`parameter.json` workflow.

## Run

From the repository root:

```bash
python -m rule_eval.main --config rule_eval/parameter.json
```

Paths in the configuration are relative to the process working directory. Outputs include the per-response evaluation table, serialized `suite_all` and `suite_relevant`, selection pair/rule CSVs, and—when held-out evaluation is enabled—per-run JSON and summary CSV files.

See the [repository README](../README.md) for plotting commands and the [redundancy-selection module](prepare_selection_data.py) for the next pipeline stage.
