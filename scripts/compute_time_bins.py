"""
compute_time_bins.py

Re-reads subsampled BEAST posterior trees/logs for a scenario and computes
per-time-bin summaries (median + 95% HPD) over ALL individual observations
(1000 posterior samples × up to 100 replicates × all nodes whose true height
falls in the bin).

Bin edges: [0, bw, 2*bw, ..., cutoff, inf]
The last bin collects everything beyond --cutoff into one interval.

All --configs are processed in a single pass over the data.
Output: one TSV per config, named {scenario}_time_bins_w{bw}_c{cutoff}.tsv

Usage
-----
    python scripts/compute_time_bins.py \
        --tsv results/run1/evaluation/linearconstant/uniform/med.tsv \
        --out_dir results/run1/evaluation/linearconstant/uniform/ \
        --configs "10,400" "20,400" "10,200"
"""

import argparse
import re
import numpy as np
import pandas as pd
from array import array
from pathlib import Path
from collections import defaultdict
from Bio import Phylo
from io import StringIO


# =============================================================================
# Statistics
# =============================================================================

def hpd(vals, credible_mass=0.95):
    vals = np.sort(vals)
    n = len(vals)
    width = int(np.ceil(credible_mass * n))
    spans = vals[width - 1:] - vals[:n - width + 1]
    i = np.argmin(spans)
    return float(vals[i]), float(vals[i + width - 1])


def summarize(values):
    vals = np.frombuffer(values, dtype=np.float32) if isinstance(values, array) else np.array(values, dtype=np.float32)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = hpd(vals)
    return float(np.median(vals)), lo, hi


# =============================================================================
# Tree helpers
# =============================================================================

def traverse_tree(tree):
    depths      = tree.depths()
    root_height = max(d for node, d in depths.items() if node.is_terminal())
    node_dict   = {}
    for node_idx, clade in enumerate(tree.find_clades(order="preorder")):
        is_int  = not clade.is_terminal()
        node_id = f"internal_{node_idx}" if is_int else f"tip_{node_idx}"
        if is_int:
            clade.name = node_id
        h  = root_height - depths[clade]
        bl = clade.branch_length or 0.0
        node_dict[node_idx] = (h, bl, is_int, node_id)
    return node_dict


def stream_beast_trees(path):
    """Yield one parsed tree at a time — O(1) memory."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.lower().startswith("tree "):
                continue
            newick = re.sub(r"\[&[^\]]*\]", "", line)
            eq = newick.find("=")
            if eq == -1:
                continue
            newick = newick[eq + 1:].strip().rstrip(";") + ";"
            yield Phylo.read(StringIO(newick), "newick")


def load_log(path):
    df        = pd.read_csv(path, sep="\t", comment="#")
    state_col = next((c for c in ("State", "state", "Sample", "sample") if c in df.columns), None)
    if state_col is None:
        raise KeyError(f"No state/sample column found in {path}. Columns: {list(df.columns)}")
    return df[pd.to_numeric(df[state_col], errors="coerce").notna()].reset_index(drop=True)


# =============================================================================
# Population helpers
# =============================================================================

# Constant ancestral population size before exponential growth starts
# (second populationModel of the CompoundPopulationModel in the ReMASTER templates)
ANCESTRAL_POP_SIZE = 10.0


def get_true_traj(pop_model, row):
    if pop_model in ("expgrowthfast", "expgrowthslow"):
        N0, r = float(row["present_pop_size"]), float(row["growth_rate"])
        return lambda t: max(N0 * np.exp(-r * t), ANCESTRAL_POP_SIZE)
    elif pop_model.startswith("bottleneck"):
        N0, Nb = float(row["present_pop_size"]), float(row["bottleneck_size"])
        bs, be = float(row["bottleneck_start"]),  float(row["bottleneck_end"])
        return lambda t: Nb if bs < t <= be else N0
    else:
        return lambda t: float(row["present_pop_size"])


def skyline_boundaries_from_preorder(node_dict, num_groups):
    times       = sorted(h for h, _, is_int, _ in node_dict.values() if is_int)
    base, r     = divmod(len(times), num_groups)
    sizes       = [base + (1 if i < r else 0) for i in range(num_groups)]
    bounds, idx = [], 0
    for i, s in enumerate(sizes):
        idx += s
        bounds.append(times[idx - 1] if i < num_groups - 1 else max(times))
    return bounds


def skyline_pop_sizes(log_row, num_groups):
    sizes = []
    for i in range(1, num_groups + 1):
        for col in (f"skyline.popSize{i}", f"skyline.popSize.{i}"):
            if col in log_row.index:
                sizes.append(float(log_row[col])); break
    return sizes


def skyline_Ne_at(t, bounds, pop_sizes):
    # bounds[i] is the end time of interval i, so interval i covers (bounds[i-1], bounds[i]];
    # times beyond the root fall into the last interval
    idx = int(np.searchsorted(bounds, t, side="left"))
    return pop_sizes[min(idx, len(pop_sizes) - 1)]


# =============================================================================
# Error computation
# =============================================================================

METRICS = [
    "pop_diff",    "pop_rel_error",    "pop_abs_rel_error",
    "rate_diff",   "rate_rel_error",   "rate_abs_rel_error",
    "height_diff", "height_rel_error", "height_abs_rel_error",
    "bl_diff",     "bl_rel_error",     "bl_abs_rel_error",
]
# raw values accumulated per-sample (estimated) or once per node (true)
EST_METRICS  = ["height_est", "bl_est", "Ne_est", "rate_est"]
TRUE_METRICS = ["height_sim", "bl_sim", "Ne_true", "rate_true"]


def compute_errors(h_sim, bl_sim, h_est, bl_est, is_int, Ne_true, Ne_est):
    def sd(a, b): return a / b if b and b != 0 and not np.isnan(b) else np.nan

    pop_d  = Ne_est - Ne_true
    rt     = 1.0 / Ne_true if Ne_true else np.nan
    re_    = 1.0 / Ne_est  if Ne_est  else np.nan
    rate_d = re_ - rt if not np.isnan(rt) else np.nan
    h_d    = h_est - h_sim if is_int else np.nan
    bl_d   = bl_est - bl_sim

    return {
        "pop_diff":             pop_d,
        "pop_rel_error":        sd(pop_d, Ne_true),
        "pop_abs_rel_error":    abs(sd(pop_d, Ne_true)),
        "rate_diff":            rate_d,
        "rate_rel_error":       sd(rate_d, rt),
        "rate_abs_rel_error":   abs(sd(rate_d, rt)),
        "height_diff":          h_d,
        "height_rel_error":     sd(h_d, h_sim) if is_int else np.nan,
        "height_abs_rel_error": abs(sd(h_d, h_sim)) if is_int else np.nan,
        "bl_diff":              bl_d,
        "bl_rel_error":         sd(bl_d, bl_sim),
        "bl_abs_rel_error":     abs(sd(bl_d, bl_sim)),
    }


# =============================================================================
# Binning
# =============================================================================

def make_bins(bin_width, cutoff):
    edges = list(np.arange(0, cutoff, bin_width)) + [cutoff, np.inf]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def make_edges(bin_width, cutoff):
    """Flat left-edge array for use with np.searchsorted."""
    return np.array(list(np.arange(0, cutoff, bin_width)) + [cutoff])


def assign_bin(h, edges, n_bins):
    return min(int(np.searchsorted(edges, h, side="right")) - 1, n_bins - 1)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv",        required=True)
    parser.add_argument("--out_dir",    required=True)
    parser.add_argument("--config",     required=True,
                        help="Single bin_width,cutoff pair e.g. '10,400'")
    parser.add_argument("--num_groups", type=int, default=10)
    parser.add_argument("--max_reps",   type=int, default=None)
    parser.add_argument("--log_suffix",   default="subsampled",
                        help="Suffix used in place of 'combined' for log files, e.g. 'combined' to use full chain")
    parser.add_argument("--trees_suffix", default="subsampled",
                        help="Suffix used in place of 'combined' for trees files, e.g. 'combined' to use full chain")
    args = parser.parse_args()

    bw, co = args.config.split(",")
    configs = [(float(bw), float(co))]

    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_df = pd.read_csv(args.tsv, sep="\t")

    if scenario_df.empty:
        print(f"No successful runs in {args.tsv}, skipping.")
        return

    if args.max_reps is not None:
        scenario_df = scenario_df.head(args.max_reps)

    row0      = scenario_df.iloc[0]
    pop_model = row0["population_model"]
    scenario  = f"{row0['sampling']}_{pop_model}_{row0['mutation_signal']}mutsig"

    # Build bin lists, edge arrays and accumulators for all configs up front
    all_bins  = {(bw, co): make_bins(bw, co)  for bw, co in configs}
    all_edges = {(bw, co): make_edges(bw, co) for bw, co in configs}
    def farr(): return array('f')

    acc = {
        (bw, co): {
            model: {b: defaultdict(farr) for b in range(len(bins))}
            for model in ("constcoal", "skyline")
        }
        for (bw, co), bins in all_bins.items()
    }
    # true values are model-independent — accumulated once per node per replicate
    true_acc = {
        (bw, co): {b: defaultdict(farr) for b in range(len(bins))}
        for (bw, co), bins in all_bins.items()
    }
    # CI coverage: fraction of (replicate × node) pairs where true value is inside posterior HPD
    ci_acc = {
        (bw, co): {
            model: {b: {"height_ci": farr(), "bl_ci": farr()} for b in range(len(bins))}
            for model in ("constcoal", "skyline")
        }
        for (bw, co), bins in all_bins.items()
    }
    n_nodes_per_bin  = {(bw, co): defaultdict(int) for bw, co in configs}
    n_trees_per_bin  = {(bw, co): defaultdict(set) for bw, co in configs}

    # Parallel accumulators for nodes with bl_sim > 1 only
    acc_long = {
        (bw, co): {
            model: {b: defaultdict(farr) for b in range(len(bins))}
            for model in ("constcoal", "skyline")
        }
        for (bw, co), bins in all_bins.items()
    }
    n_nodes_per_bin_long = {(bw, co): defaultdict(int) for bw, co in configs}
    n_trees_per_bin_long = {(bw, co): defaultdict(set) for bw, co in configs}

    PER_TREE_METRICS = [
        "bl_diff", "bl_rel_error", "bl_abs_rel_error", "bl_est", "bl_sim",
        "pop_diff", "pop_rel_error", "pop_abs_rel_error", "Ne_est", "Ne_true",
        "rate_diff", "rate_rel_error", "rate_abs_rel_error", "rate_est", "rate_true",
    ]
    per_tree_rows      = []
    per_tree_rows_long = []

    for _, rep_row in scenario_df.iterrows():
        tree_index = int(rep_row["tree_index"])

        # check files exist before accumulating anything, so skipped replicates
        # don't contribute true values or node/tree counts
        paths = {}
        skip = False
        for model in ("constcoal", "skyline"):
            tp = rep_row[f"trees_path_{model}"].replace(".combined.trees", f".{args.trees_suffix}.trees")
            lp = rep_row[f"log_path_{model}"].replace(".combined.log",     f".{args.log_suffix}.log")
            if not Path(tp).exists() or Path(tp).stat().st_size == 0:
                skip = True; break
            paths[model] = (tp, lp)
        if skip:
            continue

        with open(rep_row["sim_tree_path"]) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        sim_dict  = traverse_tree(Phylo.read(StringIO(lines[tree_index]), "newick"))
        true_traj = get_true_traj(pop_model, rep_row)

        # per-tree accumulator: key -> model -> bin_idx -> met -> list
        rep_acc = {
            key: {
                model: {b: defaultdict(farr) for b in range(len(all_bins[key]))}
                for model in ("constcoal", "skyline")
            }
            for key in configs
        }
        rep_acc_long = {
            key: {
                model: {b: defaultdict(farr) for b in range(len(all_bins[key]))}
                for model in ("constcoal", "skyline")
            }
            for key in configs
        }

        # precompute bin assignment per node for each config (h_sim is fixed)
        bin_assignments = {
            key: {
                node_idx: assign_bin(h_sim, all_edges[key], len(all_bins[key]))
                for node_idx, (h_sim, _, _, _) in sim_dict.items()
            }
            for key in configs
        }
        for key, node_bins in bin_assignments.items():
            for node_idx, bin_idx in node_bins.items():
                n_nodes_per_bin[key][bin_idx] += 1
                n_trees_per_bin[key][bin_idx].add(tree_index)
                _, bl_sim_node, _, _ = sim_dict[node_idx]
                if bl_sim_node > 1:
                    n_nodes_per_bin_long[key][bin_idx] += 1
                    n_trees_per_bin_long[key][bin_idx].add(tree_index)
                h_sim, bl_sim, is_int, _ = sim_dict[node_idx]
                Ne_true = true_traj(h_sim)
                if is_int:
                    true_acc[key][bin_idx]["height_sim"].append(h_sim)
                true_acc[key][bin_idx]["bl_sim"].append(bl_sim)
                true_acc[key][bin_idx]["Ne_true"].append(Ne_true)
                true_acc[key][bin_idx]["rate_true"].append(1.0 / Ne_true if Ne_true else np.nan)

        for model in ("constcoal", "skyline"):
            tp, lp = paths[model]
            log_df = load_log(lp)

            # per-node accumulators for CI coverage (reset each replicate × model)
            node_h_est  = {i: array('f') for i in sim_dict}
            node_bl_est = {i: array('f') for i in sim_dict}

            for s_idx, post_tree in enumerate(stream_beast_trees(tp)):
                if s_idx >= len(log_df):
                    break
                log_row   = log_df.iloc[s_idx]
                post_dict = traverse_tree(post_tree)

                if model == "skyline":
                    sky_b  = skyline_boundaries_from_preorder(post_dict, args.num_groups)
                    sky_Ne = skyline_pop_sizes(log_row, args.num_groups)
                else:
                    Ne_c = float(log_row["constant.popSize"])

                for node_idx, (h_est, bl_est, _, _) in post_dict.items():
                    h_sim, bl_sim, is_int, node_id = sim_dict[node_idx]
                    is_root = node_id == "internal_0"
                    if bl_est < 0:
                        print(f"WARNING: negative bl_est={bl_est:.6f} node={node_id} rep={tree_index} model={model}")
                    Ne_est = (Ne_c if model == "constcoal"
                              else skyline_Ne_at(h_sim, sky_b, sky_Ne))
                    obs = compute_errors(h_sim, bl_sim, h_est, bl_est, is_int,
                                         true_traj(h_sim), Ne_est)

                    for key in configs:
                        bin_idx = bin_assignments[key][node_idx]
                        for met, val in obs.items():
                            if is_root and met.startswith("bl"):
                                continue
                            if not np.isnan(val) and not np.isinf(val):
                                acc[key][model][bin_idx][met].append(val)
                                if met in PER_TREE_METRICS:
                                    rep_acc[key][model][bin_idx][met].append(val)
                        if is_int:
                            acc[key][model][bin_idx]["height_est"].append(h_est)
                        if not is_root:
                            acc[key][model][bin_idx]["bl_est"].append(bl_est)
                            rep_acc[key][model][bin_idx]["bl_est"].append(bl_est)
                            rep_acc[key][model][bin_idx]["bl_sim"].append(bl_sim)
                        acc[key][model][bin_idx]["Ne_est"].append(Ne_est)
                        acc[key][model][bin_idx]["rate_est"].append(1.0 / Ne_est if Ne_est else np.nan)
                        rep_acc[key][model][bin_idx]["Ne_est"].append(Ne_est)
                        rep_acc[key][model][bin_idx]["Ne_true"].append(true_traj(h_sim))
                        rep_acc[key][model][bin_idx]["rate_est"].append(1.0 / Ne_est if Ne_est else np.nan)
                        rep_acc[key][model][bin_idx]["rate_true"].append(1.0 / true_traj(h_sim) if true_traj(h_sim) else np.nan)

                        if bl_sim > 1:
                            for met, val in obs.items():
                                if is_root and met.startswith("bl"):
                                    continue
                                if not np.isnan(val) and not np.isinf(val):
                                    acc_long[key][model][bin_idx][met].append(val)
                                    if met in PER_TREE_METRICS:
                                        rep_acc_long[key][model][bin_idx][met].append(val)
                            if not is_root:
                                acc_long[key][model][bin_idx]["bl_est"].append(bl_est)
                                rep_acc_long[key][model][bin_idx]["bl_est"].append(bl_est)
                                rep_acc_long[key][model][bin_idx]["bl_sim"].append(bl_sim)
                            acc_long[key][model][bin_idx]["Ne_est"].append(Ne_est)
                            acc_long[key][model][bin_idx]["rate_est"].append(1.0 / Ne_est if Ne_est else np.nan)
                            rep_acc_long[key][model][bin_idx]["Ne_est"].append(Ne_est)
                            rep_acc_long[key][model][bin_idx]["Ne_true"].append(true_traj(h_sim))
                            rep_acc_long[key][model][bin_idx]["rate_est"].append(1.0 / Ne_est if Ne_est else np.nan)
                            rep_acc_long[key][model][bin_idx]["rate_true"].append(1.0 / true_traj(h_sim) if true_traj(h_sim) else np.nan)

                    if is_int:
                        node_h_est[node_idx].append(h_est)
                    if not is_root:
                        node_bl_est[node_idx].append(bl_est)

            # compute CI coverage per node after all posterior samples are accumulated
            for node_idx, (h_sim, bl_sim, is_int, node_id_ci) in sim_dict.items():
                h_vals  = np.frombuffer(node_h_est[node_idx],  dtype=np.float32)
                bl_vals = np.frombuffer(node_bl_est[node_idx], dtype=np.float32)
                is_root = node_id_ci == "internal_0"
                for key in configs:
                    bin_idx = bin_assignments[key][node_idx]
                    if is_int and len(h_vals) >= 2:
                        h_lo, h_hi = hpd(h_vals)
                        ci_acc[key][model][bin_idx]["height_ci"].append(float(h_lo <= h_sim <= h_hi))
                    if not is_root and len(bl_vals) >= 2:
                        b_lo, b_hi = hpd(bl_vals)
                        ci_acc[key][model][bin_idx]["bl_ci"].append(float(b_lo <= bl_sim <= b_hi))

        # Summarize per-tree per-bin
        for (bw, co), bins in all_bins.items():
            key = (bw, co)
            for model in ("constcoal", "skyline"):
                for bin_idx, (b_lo, b_hi) in enumerate(bins):
                    row = {
                        "scenario":   scenario,
                        "tree_index": tree_index,
                        "model":      model,
                        "bin_lower":  b_lo,
                        "bin_upper":  b_hi if not np.isinf(b_hi) else np.nan,
                        "bin_width":  bw,
                        "cutoff":     co,
                    }
                    for met in PER_TREE_METRICS:
                        med, lo, hi = summarize(rep_acc[key][model][bin_idx].get(met, []))
                        row[f"{met}_median"]    = med
                        row[f"{met}_hpd_lower"] = lo
                        row[f"{met}_hpd_upper"] = hi
                    per_tree_rows.append(row)
        del rep_acc

        for (bw, co), bins in all_bins.items():
            key = (bw, co)
            for model in ("constcoal", "skyline"):
                for bin_idx, (b_lo, b_hi) in enumerate(bins):
                    row = {
                        "scenario":   scenario,
                        "tree_index": tree_index,
                        "model":      model,
                        "bin_lower":  b_lo,
                        "bin_upper":  b_hi if not np.isinf(b_hi) else np.nan,
                        "bin_width":  bw,
                        "cutoff":     co,
                    }
                    for met in PER_TREE_METRICS:
                        med, lo, hi = summarize(rep_acc_long[key][model][bin_idx].get(met, []))
                        row[f"{met}_median"]    = med
                        row[f"{met}_hpd_lower"] = lo
                        row[f"{met}_hpd_upper"] = hi
                    per_tree_rows_long.append(row)
        del rep_acc_long

        print(f"  Processed replicate T{tree_index}", flush=True)

    # Write one output file per config
    for (bw, co), bins in all_bins.items():
        output_rows      = []
        output_rows_long = []
        for model in ("constcoal", "skyline"):
            for bin_idx, (b_lo, b_hi) in enumerate(bins):
                t_acc = true_acc[(bw, co)][bin_idx]
                base = {
                    "scenario":   scenario,
                    "model":      model,
                    "bin_lower":  b_lo,
                    "bin_upper":  b_hi if not np.isinf(b_hi) else np.nan,
                    "bin_center": float(np.median(t_acc["height_sim"])) if t_acc["height_sim"] else (b_lo + b_hi) / 2 if not np.isinf(b_hi) else np.nan,
                    "n_nodes":    n_nodes_per_bin[(bw, co)].get(bin_idx, 0),
                    "n_trees":    len(n_trees_per_bin[(bw, co)].get(bin_idx, set())),
                }
                row = dict(base)
                # true values (model-independent)
                for met in TRUE_METRICS:
                    med, lo, hi = summarize(t_acc.get(met, []))
                    row[f"{met}_median"]    = med
                    row[f"{met}_hpd_lower"] = lo
                    row[f"{met}_hpd_upper"] = hi
                # estimated values and errors (per model)
                bin_acc = acc[(bw, co)][model][bin_idx]
                for met in EST_METRICS + METRICS:
                    med, lo, hi = summarize(bin_acc.get(met, []))
                    row[f"{met}_median"]    = med
                    row[f"{met}_hpd_lower"] = lo
                    row[f"{met}_hpd_upper"] = hi
                # CI coverage: fraction of nodes whose true value is inside posterior HPD
                ci_b = ci_acc[(bw, co)][model][bin_idx]
                h_ci = np.frombuffer(ci_b["height_ci"], dtype=np.float32)
                bl_ci = np.frombuffer(ci_b["bl_ci"],    dtype=np.float32)
                row["height_ci_coverage"] = float(np.mean(h_ci))  if len(h_ci)  > 0 else np.nan
                row["bl_ci_coverage"]     = float(np.mean(bl_ci)) if len(bl_ci) > 0 else np.nan
                output_rows.append(row)

                # long-bl variant
                row_long = {
                    "scenario":   scenario,
                    "model":      model,
                    "bin_lower":  b_lo,
                    "bin_upper":  b_hi if not np.isinf(b_hi) else np.nan,
                    "bin_center": base["bin_center"],
                    "n_nodes":    n_nodes_per_bin_long[(bw, co)].get(bin_idx, 0),
                    "n_trees":    len(n_trees_per_bin_long[(bw, co)].get(bin_idx, set())),
                }
                bin_acc_long = acc_long[(bw, co)][model][bin_idx]
                for met in EST_METRICS + METRICS:
                    med, lo, hi = summarize(bin_acc_long.get(met, []))
                    row_long[f"{met}_median"]    = med
                    row_long[f"{met}_hpd_lower"] = lo
                    row_long[f"{met}_hpd_upper"] = hi
                output_rows_long.append(row_long)

        suffix   = f"w{int(bw)}_c{int(co)}"
        out_path = out_dir / f"{scenario}_time_bins_{suffix}.tsv"
        pd.DataFrame(output_rows).to_csv(out_path, sep="\t", index=False)
        print(f"  Written: {out_path} ({len(bins)} bins × 2 models)")

        out_path_long = out_dir / f"{scenario}_time_bins_{suffix}_long_bl.tsv"
        pd.DataFrame(output_rows_long).to_csv(out_path_long, sep="\t", index=False)
        print(f"  Written: {out_path_long}")

    if per_tree_rows:
        per_tree_df = pd.DataFrame(per_tree_rows)
        for (bw, co) in configs:
            suffix   = f"w{int(bw)}_c{int(co)}"
            sub      = per_tree_df[(per_tree_df["bin_width"] == bw) & (per_tree_df["cutoff"] == co)]
            out_path = out_dir / f"{scenario}_time_bins_{suffix}_per_tree.tsv"
            sub.drop(columns=["bin_width", "cutoff"]).to_csv(out_path, sep="\t", index=False)
            print(f"  Written: {out_path}")

    if per_tree_rows_long:
        per_tree_df_long = pd.DataFrame(per_tree_rows_long)
        for (bw, co) in configs:
            suffix   = f"w{int(bw)}_c{int(co)}"
            sub      = per_tree_df_long[(per_tree_df_long["bin_width"] == bw) & (per_tree_df_long["cutoff"] == co)]
            out_path = out_dir / f"{scenario}_time_bins_{suffix}_long_bl_per_tree.tsv"
            sub.drop(columns=["bin_width", "cutoff"]).to_csv(out_path, sep="\t", index=False)
            print(f"  Written: {out_path}")

    print(f"Done: {scenario}")


if __name__ == "__main__":
    main()
