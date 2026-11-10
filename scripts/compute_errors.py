"""
compute_errors.py

For a given scenario TSV (from process_results), computes per-node error
summaries (median + 95% CI across posterior samples) for each replicate.

Topology is fixed: all posterior trees share the same topology as the
simulated tree, so nodes are matched by preorder position throughout.

Each tree is traversed exactly once. The simulated tree produces a node
dictionary reused for all posterior samples.

Output: one TSV per scenario, one row per (replicate, node).

Usage
-----
    python scripts/compute_errors.py \
        --tsv results/run1/evaluation/independenthomochronous/expgrowthfast/low.tsv \
        --out_dir results/run1/evaluation/independenthomochronous/expgrowthfast/
"""

import argparse
import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path
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
    vals = np.array([v for v in values if not np.isnan(v)], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    lo, hi = hpd(vals)
    return float(np.median(vals)), lo, hi


# =============================================================================
# Single tree traversal
# =============================================================================

def traverse_tree(tree):
    """
    Single traversal. Assigns node names (internal_i / tip_i) in place,
    builds node dict.

    Returns:
      node_dict : node_idx -> (height, bl, is_internal, node_id)
    """
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


# =============================================================================
# Population trajectory
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


# =============================================================================
# Skyline helpers
# =============================================================================

# each bound is the end time of an interval
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
# Per-node error computation
# =============================================================================

METRICS = [
    "pop_diff",    "pop_rel_error",    "pop_abs_rel_error",
    "rate_diff",   "rate_rel_error",   "rate_abs_rel_error",
    "height_diff", "height_rel_error", "height_abs_rel_error",
    "bl_diff",     "bl_rel_error",     "bl_abs_rel_error",
]


def compute_errors(h_sim, bl_sim, h_est, bl_est, is_int, Ne_true, Ne_est):
    def sd(a, b): return a / b if b and b != 0 and not np.isnan(b) else np.nan

    pop_d  = Ne_est - Ne_true
    rt     = 1.0 / Ne_true if Ne_true else np.nan
    re     = 1.0 / Ne_est  if Ne_est  else np.nan
    rate_d = re - rt if not np.isnan(rt) else np.nan
    h_d    = h_est - h_sim if is_int else np.nan
    bl_d   = bl_est - bl_sim

    return {
        "pop_diff":             pop_d,          "pop_rel_error":        sd(pop_d, Ne_true),
        "pop_abs_rel_error":    abs(sd(pop_d, Ne_true)),
        "rate_diff":            rate_d,         "rate_rel_error":       sd(rate_d, rt),
        "rate_abs_rel_error":   abs(sd(rate_d, rt)),
        "height_diff":          h_d,            "height_rel_error":     sd(h_d, h_sim)  if is_int else np.nan,
        "height_abs_rel_error": abs(sd(h_d, h_sim))    if is_int else np.nan,
        "bl_diff":              bl_d,           "bl_rel_error":         sd(bl_d, bl_sim),
        "bl_abs_rel_error":     abs(sd(bl_d, bl_sim)),
    }


# =============================================================================
# Main
# =============================================================================

def load_beast_trees(path):
    """Parse BEAST nexus trees, stripping metadata annotations BioPython can't handle."""
    trees = []
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
            trees.append(Phylo.read(StringIO(newick), "newick"))
    return trees


def load_log(path):
    df        = pd.read_csv(path, sep="\t", comment="#")
    state_col = next((c for c in ("State", "state", "Sample", "sample") if c in df.columns), None)
    if state_col is None:
        raise KeyError(f"No state/sample column found in {path}. Columns: {list(df.columns)}")
    return df[pd.to_numeric(df[state_col], errors="coerce").notna()].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv",         required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--num_groups",  type=int, default=10)
    parser.add_argument("--max_reps",    type=int, default=None)
    parser.add_argument("--traj_points", type=int, default=200,
                        help="Number of time grid points for population trajectory output")
    parser.add_argument("--log_suffix",   default="subsampled",
                        help="Suffix used in place of 'combined' for log files, e.g. 'subsampled_v2'")
    parser.add_argument("--trees_suffix", default="subsampled",
                        help="Suffix used in place of 'combined' for trees files, e.g. 'subsampled_v2'")
    args = parser.parse_args()

    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_df = pd.read_csv(args.tsv, sep="\t")

    if scenario_df.empty:
        print(f"No successful runs in {args.tsv}, skipping.")
        return

    row0        = scenario_df.iloc[0]
    pop_model   = row0["population_model"]
    scenario    = f"{row0['sampling']}_{pop_model}_{row0['mutation_signal']}mutsig"
    output_rows  = []
    pop_sum_rows = []

    # Global accumulators across all replicates for trajectory output
    global_sky_b  = []
    global_sky_Ne = []
    global_Ne_c   = []

    if args.max_reps is not None:
        scenario_df = scenario_df.head(args.max_reps)

    for _, rep_row in scenario_df.iterrows():
        tree_index = int(rep_row["tree_index"])

        # Simulated tree
        with open(rep_row["sim_tree_path"]) as fh:
            lines = [l.strip() for l in fh if l.strip()]
        sim_dict  = traverse_tree(Phylo.read(StringIO(lines[tree_index]), "newick"))
        true_traj = get_true_traj(pop_model, rep_row)

        # Load subsampled files — skip replicate if either model's files are missing/empty
        model_data = {}
        skip = False
        for model in ("constcoal", "skyline"):
            tp = rep_row[f"trees_path_{model}"].replace(".combined.trees", f".{args.trees_suffix}.trees")
            lp = rep_row[f"log_path_{model}"].replace(".combined.log",     f".{args.log_suffix}.log")
            if not Path(tp).exists() or Path(tp).stat().st_size == 0:
                skip = True; break
            log   = load_log(lp)
            trees = load_beast_trees(tp)
            model_data[model] = {"log": log, "trees": trees, "n": min(len(trees), len(log))}
        if skip:
            continue

        # Accumulators: model -> node_idx -> metric -> [values]
        EXTRA = ["height_est", "bl_est"]
        acc = {m: {n: {met: [] for met in METRICS + EXTRA} for n in range(len(sim_dict))}
               for m in ("constcoal", "skyline")}

        # Total branch length per posterior sample
        tbl_acc = {"constcoal": [], "skyline": []}


        # Pop-size accumulators for summary output
        all_sky_b  = []
        all_sky_Ne = []
        all_Ne_c   = []

        for model in ("constcoal", "skyline"):
            for s_idx in range(model_data[model]["n"]):
                post_dict = traverse_tree(model_data[model]["trees"][s_idx])
                log_row   = model_data[model]["log"].iloc[s_idx]

                if model == "skyline":
                    sky_b  = skyline_boundaries_from_preorder(post_dict, args.num_groups)
                    sky_Ne = skyline_pop_sizes(log_row, args.num_groups)
                    all_sky_b.append(sky_b)
                    all_sky_Ne.append(sky_Ne)
                else:
                    Ne_c   = float(log_row["constant.popSize"])
                    sky_b  = sky_Ne = None
                    all_Ne_c.append(Ne_c)

                tbl_acc[model].append(
                    sum(bl for h, bl, _, nid in post_dict.values() if nid != "internal_0")
                )

                for node_idx, (h_est, bl_est, _, _) in post_dict.items():
                    h_sim, bl_sim, is_int, node_id = sim_dict[node_idx]
                    is_root = node_id == "internal_0"
                    if bl_est < 0:
                        print(f"WARNING: negative bl_est={bl_est:.6f} node={node_id} rep={tree_index} model={model}")
                    Ne_est = (float(log_row["constant.popSize"]) if model == "constcoal"
                              else skyline_Ne_at(h_sim, sky_b, sky_Ne))
                    obs = compute_errors(
                        h_sim, bl_sim, h_est, bl_est, is_int,
                        true_traj(h_sim), Ne_est,
                    )
                    for met in METRICS:
                        val = np.nan if is_root and met.startswith("bl") else obs[met]
                        acc[model][node_idx][met].append(val)
                    acc[model][node_idx]["height_est"].append(h_est if is_int else np.nan)
                    acc[model][node_idx]["bl_est"].append(np.nan if is_root else bl_est)


        # Feed into global accumulators
        global_sky_b.extend(all_sky_b)
        global_sky_Ne.extend(all_sky_Ne)
        global_Ne_c.extend(all_Ne_c)

        # Build pop-size summary over all posterior samples
        sky_b_arr  = np.array(all_sky_b)
        sky_Ne_arr = np.array(all_sky_Ne)
        Ne_c_arr   = np.array(all_Ne_c)

        sky_lowers = []
        sky_uppers = []
        for i in range(sky_Ne_arr.shape[1]):
            lo, hi = hpd(sky_Ne_arr[:, i])
            sky_lowers.append(lo)
            sky_uppers.append(hi)

        c_lo, c_hi = hpd(Ne_c_arr)

        # subsample to 100 for storage of raw samples only
        rng       = np.random.default_rng(seed=tree_index)
        n_keep    = min(100, len(sky_Ne_arr))
        idx_sky   = rng.choice(len(sky_Ne_arr), size=n_keep, replace=False)
        idx_c     = rng.choice(len(Ne_c_arr),   size=n_keep, replace=False)

        # Simulated total branch length (fixed, not from posterior)
        tbl_sim = sum(bl for _, bl, _, nid in sim_dict.values() if nid != "internal_0")

        tbl_row = {}
        for model in ("constcoal", "skyline"):
            vals = np.array(tbl_acc[model])
            lo, hi = hpd(vals)
            tbl_row[f"{model}_total_bl_median"]    = float(np.median(vals))
            tbl_row[f"{model}_total_bl_hpd_lower"] = float(lo)
            tbl_row[f"{model}_total_bl_hpd_upper"] = float(hi)

            diff     = vals - tbl_sim
            rel      = diff / tbl_sim
            abs_rel  = np.abs(rel)
            for name, arr in (("total_bl_diff",         diff),
                               ("total_bl_rel_error",    rel),
                               ("total_bl_abs_rel_error",abs_rel)):
                lo, hi = hpd(arr)
                tbl_row[f"{model}_{name}_median"]    = float(np.median(arr))
                tbl_row[f"{model}_{name}_hpd_lower"] = float(lo)
                tbl_row[f"{model}_{name}_hpd_upper"] = float(hi)

            for name, exclude_root, only_internal in (
                ("bl_rel_error",        True,  False),
                ("bl_abs_rel_error",    True,  False),
                ("height_rel_error",    False, True),
                ("height_abs_rel_error",False, True),
            ):
                pool = []
                for node_idx, (h_sim, bl_sim, is_int, node_id) in sim_dict.items():
                    is_root = node_id == "internal_0"
                    if exclude_root and is_root:
                        continue
                    if only_internal and not is_int:
                        continue
                    vals = [v for v in acc[model][node_idx][name]
                            if not np.isnan(v) and not np.isinf(v)]
                    pool.extend(vals)
                arr = np.array(pool)
                lo, hi = hpd(arr) if len(arr) >= 2 else (np.nan, np.nan)
                tbl_row[f"{model}_{name}_median"]    = float(np.median(arr)) if len(arr) else np.nan
                tbl_row[f"{model}_{name}_hpd_lower"] = float(lo)
                tbl_row[f"{model}_{name}_hpd_upper"] = float(hi)

        pop_sum_rows.append({
            "scenario":               scenario,
            "sampling":               row0["sampling"],
            "population_model":       pop_model,
            "mutation_signal":        row0["mutation_signal"],
            "tree_index":             tree_index,
            "skyline_times":          list(np.median(sky_b_arr, axis=0)),
            "skyline_medians":        list(np.median(sky_Ne_arr, axis=0)),
            "skyline_lowers":         sky_lowers,
            "skyline_uppers":         sky_uppers,
            "skyline_samples":        pd.DataFrame(sky_Ne_arr[idx_sky], columns=range(args.num_groups)),
            "skyline_times_samples":  pd.DataFrame(sky_b_arr[idx_sky],  columns=range(args.num_groups)),
            "coalescent_median":      float(np.median(Ne_c_arr)),
            "coalescent_lower":       float(c_lo),
            "coalescent_upper":       float(c_hi),
            "coalescent_samples":     pd.Series(Ne_c_arr[idx_c]),
            **tbl_row,
        })

        for node_idx, (h_sim, bl_sim, is_int, node_id) in sim_dict.items():
            row = {
                "scenario": scenario, "tree_index": tree_index,
                "node_idx": node_idx, "node_id":    node_id,
                "internal": is_int,   "height_sim": h_sim, "bl_sim": bl_sim,
            }
            for model in ("constcoal", "skyline"):
                for met in METRICS:
                    med, lo, hi = summarize(acc[model][node_idx][met])
                    row[f"{model}_{met}_median"]    = med
                    row[f"{model}_{met}_hpd_lower"] = lo
                    row[f"{model}_{met}_hpd_upper"] = hi
                for est in EXTRA:
                    med, lo, hi = summarize(acc[model][node_idx][est])
                    row[f"{model}_{est}_median"]    = med
                    row[f"{model}_{est}_hpd_lower"] = lo
                    row[f"{model}_{est}_hpd_upper"] = hi
                h_lo = row[f"{model}_height_est_hpd_lower"]
                h_hi = row[f"{model}_height_est_hpd_upper"]
                b_lo = row[f"{model}_bl_est_hpd_lower"]
                b_hi = row[f"{model}_bl_est_hpd_upper"]
                row[f"{model}_height_inside_ci"] = bool(h_lo <= h_sim <= h_hi) if is_int else np.nan
                row[f"{model}_bl_inside_ci"]     = np.nan if node_id == "internal_0" else bool(b_lo <= bl_sim <= b_hi)
            output_rows.append(row)

        print(f"  Processed replicate T{tree_index} ({len(sim_dict)} nodes)")

    pd.DataFrame(output_rows).to_csv(out_dir / f"{scenario}_node_errors.tsv", sep="\t", index=False)

    pop_sum_df = pd.DataFrame(pop_sum_rows)
    with open(out_dir / f"{scenario}_pop_summary.pkl", "wb") as fh:
        pickle.dump(pop_sum_df, fh)

    # Population trajectory: median + HPD at a dense time grid across all replicates
    if global_sky_Ne:
        g_sky_b_arr  = np.array(global_sky_b)   # (N_samples_total, num_groups)
        g_sky_Ne_arr = np.array(global_sky_Ne)   # (N_samples_total, num_groups)
        g_Ne_c_arr   = np.array(global_Ne_c)     # (N_samples_total,)

        time_grid = np.linspace(0, 500, args.traj_points)

        # Evaluate skyline Ne at each time point for every sample
        sky_at_t = np.array([
            [skyline_Ne_at(t, g_sky_b_arr[s], g_sky_Ne_arr[s]) for t in time_grid]
            for s in range(len(g_sky_Ne_arr))
        ])  # (N_samples_total, traj_points)

        c_med_traj = float(np.median(g_Ne_c_arr))
        c_lo_traj, c_hi_traj = hpd(g_Ne_c_arr)

        traj_rows = []
        for k, t in enumerate(time_grid):
            sky_vals = sky_at_t[:, k]
            sky_lo, sky_hi = hpd(sky_vals)
            traj_rows.append({
                "time":                    t,
                "skyline_median":          float(np.median(sky_vals)),
                "skyline_hpd_lower":       float(sky_lo),
                "skyline_hpd_upper":       float(sky_hi),
                "constcoal_median":        c_med_traj,
                "constcoal_hpd_lower":     c_lo_traj,
                "constcoal_hpd_upper":     c_hi_traj,
            })

        pd.DataFrame(traj_rows).to_csv(
            out_dir / f"{scenario}_pop_trajectory.tsv", sep="\t", index=False
        )

    print(f"Done: {scenario} — {len(output_rows)} node-error rows, {len(pop_sum_rows)} pop-summary rows")


if __name__ == "__main__":
    main()
