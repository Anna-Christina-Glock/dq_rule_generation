# dq_rule_generation - Data Quality Rule Generation

A system for automatically generating data quality rules using Large Language Models (LLMs) with Great Expectations integration.

## Overview

This project implements an LLM-based pipeline for generating data quality validation rules. The system:

- Takes clean and polluted datasets as input
- Uses LLMs (Ollama or OpenAI) to analyze differences and generate appropriate validation rules
- Produces Great Expectations expectations that can be added to validation suites
- Includes evaluation and plotting tools to analyze rule quality

## Features

- **LLM-based Rule Generation**: Automatically generate Great Expectations rules using LLMs
- **Parallel Processing**: Process multiple data rows concurrently using threading
- **Multiple Prompt Templates**: Different prompt strategies for rule generation (IDs: 0, 1, 2)
- **Rule Evaluation**: Evaluate generated rules with precision/recall metrics
- **Rule Selection**: Greedy algorithms for selecting optimal rule subsets
- **Visualization**: R and Python-based plotting for evaluation results

## Project Structure

```
dq_rule_generation/
├── rule_gen/                 # Rule generation module
│   ├── __init__.py
│   ├── llm_utils.py         # LLM client utilities (Ollama/OpenAI)
│   ├── main.py              # Main entry point
│   ├── main_server_parallel.py  # Parallel processing implementation
│   └── rule_gen_prompt.py   # Prompt templates
├── rule_eval/               # Rule evaluation module
│   ├── __init__.py
│   ├── evaluate_rules.py    # Great Expectations validation
│   ├── plot_results.py      # Result visualization
│   └── rule_selection.py    # Greedy rule selection algorithms
├── r_plots/                 # R plotting scripts
│   ├── rule_execution_status.R
│   └── r_metric_error_class.R
├── data/                    # Input/output data
├── configs/                 # Configuration files
│   ├── parameter.json       # Processing parameters
│   └── connection.json      # LLM connection settings
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Git

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd dq_rule_generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure LLM connection (see Configuration section below)

## Configuration

### Connection Configuration (`configs/connection.json`)

Create a `configs/connection.json` file with your LLM provider settings:

```json
{
  "default": {
    "llm-package": "ollama",
    "base_url": "http://localhost:11434",
    "modelname": "gemma-4-31B-it-FP8-Dynamic"
  },
  "openai": {
    "llm-package": "openai",
    "api_key": "your-api-key",
    "base_url": "https://api.openai.com/v1",
    "modelname": "gpt-4"
  }
}
```

Available LLM packages:
- `ollama`: For local Ollama instances
- `openai`: For OpenAI API

### Parameter Configuration (`configs/parameter.json`)

Create a `configs/parameter.json` file with your processing parameters:

```json
{
  "location": "PCI",
  "promptIds": [8, 9, 10],
  "expectationListIds": [0, 2],
  "threadCount": 52,
  "Files": {
    "pci_data": {
      "name": "dirty_data.csv",
      "cleanFile": "clean_data.csv",
      "encoding": "utf-8",
      "delimiter": ",",
      "n_cols": 14,
      "id_column": "pk",
      "err_dect_info": null
    }
  },
  "PCI": {
    "Paths": {
      "Data": "data/pci",
      "Result": "results/"
    },
    "File": {
      "Input": ["pci_data"]
    }
  }
}
```

## Usage

### Running Rule Generation

```bash
# Using default config paths
python -m rule_gen.main

# With custom config paths
python -m rule_gen.main --config-path path/to/config.json --connection-config-path path/to/connection.json
```

### Running Evaluation

```bash
# Generate plots and statistics
python -m rule_eval.plot_results --input-dir results/ --output-dir plots/ --dataset PCI
```

### Rule Selection

```python
from rule_eval.rule_selection import greedy_rule_selection

# Use greedy algorithms to select optimal rules
selected_rules, coverage_by_rule, covered_pks, covered_cols = greedy_rule_selection(
    df_pairs, 
    df_rules, 
    precision_required=0.8,
    row_weight=1.0,
    col_weight=0.5
)
```

## LLM Integration

### Ollama (Local)

If you do not have Ollame installed you can download it from here (https://ollama.com/download)
For local Ollama integration, ensure your Ollama server is running:

```bash
# Start Ollama server (if not already running)
ollama serve

# Pull a model
ollama run gemma4
```

### OpenAI (Cloud)

For OpenAI API, set your API key in the connection configuration:

```json
{
  "openai": {
    "llm-package": "openai",
    "api_key": "sk-...",
    "modelname": "gpt-4"
  }
}
```

## Prompt Templates

The system supports three prompt template IDs:

| ID | Description |
|----|-------------|
| 0 | Clean + Dirty data with error detection info |
| 1 | Dirty data only with error detection info |
| 2 | Dirty data with explicit error column information |

## Evaluation Metrics

The system calculates:

- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: 2 × (Precision × Recall) / (Precision + Recall)

Metrics are calculated at multiple levels:
- Per row level
- Per rule/column level
- Per column level

## Datasets

The unlabeled datasets beers, flights, hospital and tax stem from https://github.com/BigDaMa/raha/tree/master/datasets.
The PCI dataset was generated by https://github.com/Anna-Christina-Glock/pci-llm-toolkit/tree/main/data-generator.

## Dependencies

See `requirements.txt` for the full list of dependencies.

Key dependencies:
- `great-expectations`: Data validation framework
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `ollama`: Local LLM client
- `openai`: OpenAI API client

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- Built using [Great Expectations](https://greatexpectations.io/)
- LLM integration via Ollama and OpenAI