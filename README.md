# beyond-algorithms-llms-CoEx

Exploring how LLMs can be used in mechanistic interpretability through
**compositional explanations** of neurons, building on
[Compositional Explanations of Neurons (Mu & Andreas, 2020)](https://github.com/jayelm/compexp).

## Layout

```
src/llms_for_CE/
  analyze.py            # adapted from compexp/nli/code/analyze.py
  formula.py            # logical-formula representation for explanations
  experiments/
    full_input.py       # one prompt per neuron, with every sentence
    batched_input.py    # activations sent to the LLM in batches
    pairs_100.py        # currently identical to batched_input.py
scripts/
  data_prep/            # one-off data conversion (clean_acts, json_to_csv_acts)
  measure_size/         # model / KV-cache / prompt size estimates
data/                   # local only (gitignored)
results/                # local only (gitignored); experiments write output.csv here
compexp/                # external repo, cloned separately (gitignored)
```

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
git clone https://github.com/jayelm/compexp.git   # needed by analyze.py
```

`analyze.py` imports `settings`, `util`, `vis` and `data` from
`compexp/nli/code`, so that directory has to be on `PYTHONPATH` when you run it.

## Configuration

Paths come from environment variables. Each one has a default, so nothing
needs setting for local runs.

| Variable       | Default         | Purpose                                     |
|----------------|-----------------|---------------------------------------------|
| `DATA_DIR`     | `<repo>/data`    | Input data                                  |
| `RESULTS_DIR`  | `<repo>/results` | Experiment outputs                          |
| `HF_HOME`      | `~/.cache/huggingface` | Hugging Face model weights            |
| `UV_CACHE_DIR` | `~/.cache/uv`   | uv package cache                            |

On the Kubernetes pod, point the caches at the persistent volume so they
survive restarts (add these lines to `~/.bashrc`):

```bash
export UV_CACHE_DIR=/workspace/.uv-cache
export HF_HOME=/workspace/.cache/huggingface
```

In a container, mount volumes for data, results and weights, then set the same
variables to point at them. `.dockerignore` keeps them out of the image.

## Data

`data/` isn't in the repo. The experiments expect these files:

- `data/full/sentence_pairs_mask.csv`
- `data/full/acts_only.csv` (created from `preds_acts.csv` by `scripts/data_prep/clean_acts.py`)
- `data/full/top_n_concepts_from_compexp.csv`

## Running

```bash
uv run python src/llms_for_CE/experiments/batched_input.py
```

Output goes to `results/output.csv`.
