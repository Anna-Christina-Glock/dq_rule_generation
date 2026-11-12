# Rule-generation runner

`main_server_paralell.py` is the rule-generation entry point used for the LeDQeR experiments. It sends one prompt per dirty tuple to an OpenAI-compatible/vLLM or Ollama endpoint and writes each prompt, answer, error, and available timing measurements to CSV.

See the repository-level [`README.md`](../README.md) for installation, datasets, evaluation, and redundancy reduction.

## Run


## Parameters to set before a run

Edit `config/connection.json`:

| Key | Change it to |
|---|---|
| `base_url` | The reachable API endpoint serving the model |
| `api_key` | The endpoint's key, if required; otherwise retain the local placeholder |
| `modelname` | The exact model identifier served by the endpoint |
| `llm-package` | `openai` for an OpenAI-compatible endpoint or `ollama` for Ollama |
| `reasoning` | Optional provider-specific reasoning settings |

Edit `config/parameter.json`:

| Key | What to configure |
|---|---|
| `location` | Selects the matching path/file block, currently `Public` |
| `modelConfigs` | Selects keys defined in `connection.json` |
| `promptIds` | Selects paper variants: `9` Just Dirty, `11` Dirty & Column, `10` Dirty & Type, `8` Dirty & Clean |
| `expectationLiIds` | Selects GX prompt detail: `0` names only, `2` names plus parameter information |
| `maxRows` | Optional positive limit for a smoke run; use `null` for all rows |
| `Public.Paths.Data` | Directory containing the configured dirty and clean CSVs |
| `Public.Paths.Result` | Directory for generated response CSVs |
| `Public.File.Input` | Dataset keys to run; each must exist in `Files` |
| `Files.<dataset>.name` | Dirty input CSV filename |
| `Files.<dataset>.cleanFile` | Matching clean CSV filename |
| `Files.<dataset>.OutputSuffix` | Dataset suffix in output filenames |
| `Files.<dataset>.n_cols` | Number of tuple columns shown in prompts |
| `Files.<dataset>.id_column` | Shared key used to align dirty and clean tuples |
| `Files.<dataset>.encoding` | CSV encoding expected by both files |
| `Files.<dataset>.delimiter` | CSV field delimiter |
| `Files.<dataset>.maxRows` | Optional per-dataset row limit; overrides top-level `maxRows` |
| `Files.<dataset>.err_dect_info` | Optional domain context, when needed |

The CSV filenames and endpoint settings must match the local environment. The
checked-in `glm`, `gemma`, and `qwen` connection entries are local examples.

### Optional PCI input

The `Files.pci` entry is configured but not selected by default because the
synthetic PCI files are not included in the repository. Generate or obtain
these files first, place them under `Public.Paths.Data`, then add `"pci"` to
`Public.File.Input`:

- dirty: `trainGenData_adressPerson_dirtyPol_whole_6000_100_1.csv`
- clean: `trainGenData_adressPerson_6000_new_vorwahlFix.csv`

Do not enable PCI until both files exist at those paths.
After updating the settings, run from the repository root:

```bash
python rule_gen/main_server_paralell.py
```

Configuration files are resolved relative to this directory. Data and result
paths in `parameter.json` are resolved relative to the process working
directory.


## Prompt variants

| ID | Paper label | Input |
|---:|---|---|
| 9 | Just Dirty | dirty tuple |
| 11 | Dirty & Column | dirty tuple and affected column |
| 10 | Dirty & Type | dirty tuple, affected column, and error type |
| 8 | Dirty & Clean | dirty tuple and clean counterpart |

`expectationLiIds` selects the GX information supplied with the prompt. ID `0` supplies expectation names; ID `2` additionally supplies GX parameter information.

## Adaptive concurrency

The optional `adaptive` block controls a shared request gate for each model configuration:

| Key | Meaning |
|---|---|
| `enabled` | enable or disable adaptive request admission |
| `max_threads` | maximum number of admitted concurrent requests |
| `min_threads` | minimum concurrency |
| `ttft_window` | number of recent TTFT measurements used by the controller |
| `ttft_decrease_if_above_s` | median TTFT that triggers a decrease |
| `decrease_factor` | multiplicative decrease factor |
| `cooldown_s` | minimum time between limit changes |
| `ttft_increase_if_below_s` | all values in a full window must be below this value before an increase |
| `increase_step` | additive concurrency increase |

Running requests are not interrupted. The gate only controls admission of new requests.

## Output

The result filename contains the model configuration, reasoning tag, prompt ID, expectation-information ID, timestamp, and dataset suffix. Each CSV includes:

| Column | Meaning |
|---|---|
| source columns | original dirty tuple and labels |
| `prompt` | complete prompt sent to the model |
| `answer` | visible model response |
| `error` | request exception, if any |
| `modelname` / `modelConfig` | served model and connection entry |
| `promptId` / `expectationLiId` | experimental input configuration |
| `reasoning` | configured reasoning tag |
| `ttft_s` | time to first streamed chunk |
| `total_time_s` | complete request duration |
| `out_tokens` | completion tokens reported by the server |
| `out_tokens_per_s` | output-token rate after the first chunk |
| `in_flight` | admitted requests when this request started |

Ollama requests use the non-streaming path, so TTFT and output-token measurements may be unavailable.
