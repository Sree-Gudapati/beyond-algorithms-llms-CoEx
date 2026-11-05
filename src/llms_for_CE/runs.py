"""Create a new results directory for each experiment run and log it in runs.csv.
   LLM Generated. Only for formatting results nicely"""
import csv
import json
from datetime import datetime
from pathlib import Path


def start_run(results_dir, script_path, model, name_params=(), **parameters):
    """Create results_dir/<timestamp>_<script>_<model>[_<param>-<value>...]/results.csv
    and add a line to results_dir/runs.csv recording when it ran, which script,
    model and parameters.

    name_params: names of entries in `parameters` to also include in the
    directory name, e.g. name_params=["n_pairs"] -> ..._n_pairs-100

    Returns (out_file, csv_writer).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    script = Path(script_path).name
    now = datetime.now()

    run_name = f"{now:%Y-%m-%d_%H%M%S}_{Path(script).stem}_{model.split('/')[-1]}"
    for key in name_params:
        run_name += f"_{key}-{parameters[key]}"

    # exist_ok=False errors instead of reusing an existing run directory
    run_dir = results_dir / run_name
    run_dir.mkdir(exist_ok=False)
    out_path = run_dir / "results.csv"

    # mode "x" errors instead of overwriting an existing file
    out_file = open(out_path, mode="x", newline="", encoding="utf-8")
    csv_writer = csv.writer(out_file)
    csv_writer.writerow(["neuron", "formula"])

    # log the run
    log = results_dir / "runs.csv"
    is_new = not log.exists()
    with open(log, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["results_file", "started", "script", "model", "parameters"])
        w.writerow([out_path.relative_to(results_dir).as_posix(), f"{now:%Y-%m-%d %H:%M:%S}",
                    script, model, json.dumps(parameters)])

    return out_file, csv_writer
