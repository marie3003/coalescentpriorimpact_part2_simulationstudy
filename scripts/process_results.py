"""
Build per-scenario DataFrames from successful_mcmc_runs.csv.

For each (sampling, population_model, mutation_signal) combination, saves a
pickle containing one row per replicate tree that completed successfully for
BOTH constcoal and skyline, with all file paths and population model parameters.

Also saves a summary CSV with successful tree counts per condition.
Per-scenario output is one TSV per (sampling, population_model, mutation_signal).

Usage
-----
Run from the BESP_paper-analyses project root:
    python scripts/process_results.py --beast_dir results/run1/beast_inference
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def assign_model_params(pop_model):
    params = {
        "expgrowthfast": {"present_pop_size": 2000, "growth_rate": 0.02, "bottleneck_size": None, "bottleneck_start": None, "bottleneck_end": None},
        "expgrowthslow": {"present_pop_size": 2000, "growth_rate": 0.01, "bottleneck_size": None, "bottleneck_start": None, "bottleneck_end": None},
        "uniform":       {"present_pop_size": 1000, "growth_rate": None, "bottleneck_size": None, "bottleneck_start": None, "bottleneck_end": None},
        "bottleneck":    {"present_pop_size": 1000, "growth_rate": None, "bottleneck_size": 100, "bottleneck_start": 20,   "bottleneck_end": 23},
    }
    return params.get(pop_model, {"present_pop_size": None, "growth_rate": None, "bottleneck_size": None, "bottleneck_start": None, "bottleneck_end": None})


def parse_path(trees_path):
    """Extract metadata and derive all related file paths from a combined.trees path."""
    p = Path(trees_path)
    stem = p.stem.replace(".combined", "")

    m = re.match(
        r"(constcoal|skyline)_(independenthomochronous|linearconstant)_([a-z0-9]+)_([a-z]+mutsig)\.T(\d+)",
        stem
    )
    if not m:
        return None

    inference_model, sampling, pop_model, mutsig, tree_index = m.groups()
    run_folder = next((part for part in p.parts if part.startswith("run")), "run1")
    sim_tree   = Path("results") / run_folder / "simulated_data" / sampling / pop_model / f"{pop_model}.trees"

    return {
        "inference_model":  inference_model,
        "sampling":         sampling,
        "population_model": pop_model,
        "mutation_signal":  mutsig.replace("mutsig", ""),
        "tree_index":       int(tree_index),
        "log_path":         str(p.parent / f"{stem}.combined.log"),
        "trees_path":       trees_path,
        "sim_tree_path":    str(sim_tree),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beast_dir", required=True, help="Path to beast_inference directory (contains successful_mcmc_runs.csv)")
    args = parser.parse_args()

    beast_dir = Path(args.beast_dir)
    eval_dir  = beast_dir.parent / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(beast_dir / "successful_mcmc_runs.csv")

    # Parse every path into structured metadata
    records = []
    for trees_path in df["trees_path"]:
        row = parse_path(trees_path)
        if row:
            records.append(row)

    all_df = pd.DataFrame(records)

    # Split into constcoal and skyline, then inner-join on shared keys
    shared = ["sampling", "population_model", "mutation_signal", "tree_index", "sim_tree_path"]
    constcoal = (all_df[all_df["inference_model"] == "constcoal"]
                 .rename(columns={"log_path": "log_path_constcoal", "trees_path": "trees_path_constcoal"})
                 .drop(columns="inference_model"))
    skyline   = (all_df[all_df["inference_model"] == "skyline"]
                 .rename(columns={"log_path": "log_path_skyline", "trees_path": "trees_path_skyline"})
                 .drop(columns="inference_model"))

    merged = constcoal.merge(skyline, on=shared, how="inner")

    # Add population model parameters
    params = pd.DataFrame(
        merged["population_model"].apply(assign_model_params).tolist(),
        index=merged.index
    )
    merged = pd.concat([merged, params], axis=1)

    # Save one TSV per (sampling, population_model, mutation_signal)
    for (sampling, pop_model, mutsig), group in merged.groupby(["sampling", "population_model", "mutation_signal"]):
        folder = eval_dir / sampling / pop_model
        folder.mkdir(parents=True, exist_ok=True)
        group.reset_index(drop=True).to_csv(folder / f"{mutsig}.tsv", sep="\t", index=False)

    # Summary: successful tree counts per condition
    summary = (merged.groupby(["sampling", "population_model", "mutation_signal"])
               .size()
               .reset_index(name="n_successful"))
    summary.to_csv(eval_dir / "successful_counts.tsv", sep="\t", index=False)

    print(f"Saved {len(merged.groupby(['sampling','population_model','mutation_signal']))} scenario files to {eval_dir}")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()