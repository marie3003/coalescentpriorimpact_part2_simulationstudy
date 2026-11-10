from Bio import Nexus, Phylo, SeqIO
from collections import defaultdict
from io import StringIO

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib import gridspec

import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pandas as pd
import numpy as np

import re
from pathlib import Path

import itertools
from itertools import combinations

from scipy.stats import gaussian_kde

# Embed fonts as editable text (not outlines) in vector exports
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})

_VECTOR_FORMATS = {".pdf", ".svg", ".eps", ".ps"}

# Constant ancestral population size before exponential growth starts
# (second populationModel of the CompoundPopulationModel in the ReMASTER templates)
ANCESTRAL_POP_SIZE = 10.0


def save_vector_figure(fig, save_path):
    """
    Save fig as a vector graphic; non-vector extensions (e.g. .png) are replaced by .pdf.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    save_path : str or Path
        Output path. Supported vector formats are .pdf, .svg, .eps and .ps;
        any other suffix is replaced by .pdf. Saved with a transparent
        background and a tight bounding box.
    """
    save_path = Path(save_path)
    if save_path.suffix.lower() not in _VECTOR_FORMATS:
        save_path = save_path.with_suffix(".pdf")
        print(f"Saving as vector graphic: {save_path}")
    fig.savefig(save_path, bbox_inches="tight", transparent=True)


def hpd(vals, credible_mass=0.95):
    """
    Highest posterior density interval of a sample.

    Finds the narrowest interval that contains ceil(credible_mass * n) of
    the n sample values.

    Parameters
    ----------
    vals : array-like
        Sample values (need not be sorted; must not contain NaN).
    credible_mass : float
        Fraction of the sample the interval must contain (default 0.95).

    Returns
    -------
    (float, float)
        Lower and upper bound of the HPD interval.
    """
    vals = np.sort(vals)
    n = len(vals)
    width = int(np.ceil(credible_mass * n))
    spans = vals[width - 1:] - vals[:n - width + 1]
    i = np.argmin(spans)
    return float(vals[i]), float(vals[i + width - 1])


def plot_ci_coverage_heatmap(root_height_df_wide, sampling, title=None,
                             save_path=None, figsize=None,
                             cbar_label="Fraction of replicates"):
    """
    Root height CI coverage heatmap, comparing constcoal and skyline.

    For each combination of mutation signal × population model, compute the fraction
    of replicates falling into each of 4 categories:
      - both inside CI
      - only constcoal inside CI
      - only skyline inside CI
      - neither inside CI
    and show it as a 2×2 heatmap. Rows = mutation signal, columns = population model.

    Parameters
    ----------
    root_height_df_wide : DataFrame
        One row per replicate, e.g. the root rows of load_node_errors()
        (node_id == "internal_0"). Needs sampling, mutation_signal,
        population_model and the boolean CI columns
        constcoal_height_inside_ci / skyline_height_inside_ci
        (or height_inside_ci_constcoal / height_inside_ci_skyline).
    sampling : str
        Sampling scheme to plot, e.g. "linearconstant".
    title : str, optional
        Suptitle; defaults to "Root height CI coverage — {sampling}".
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    figsize : tuple, optional
        Override the default figure size.
    cbar_label : str
        Colorbar label.
    """
    mutsig_order    = ["low", "med", "high"]
    mutsig_labels   = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}
    pop_model_order = ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]
    title_map       = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                       "uniform": "Uniform", "bottleneck": "Bottleneck"}

    _cmap_colors = ["#c5dfc0", "#7db87a", "#4e9a8a", "#397398", "#00181E"]
    cmap   = plt.matplotlib.colors.LinearSegmentedColormap.from_list("green_cmap", _cmap_colors)
    cmap_r = cmap.reversed()

    df = root_height_df_wide[root_height_df_wide["sampling"] == sampling].copy()
    const_col = "constcoal_height_inside_ci" if "constcoal_height_inside_ci" in df.columns else "height_inside_ci_constcoal"
    sky_col   = "skyline_height_inside_ci"   if "skyline_height_inside_ci"   in df.columns else "height_inside_ci_skyline"
    ci_const = df[const_col].astype(bool)
    ci_sky   = df[sky_col].astype(bool)
    df["both"]           = ci_const & ci_sky
    df["only constcoal"] = ci_const & ~ci_sky
    df["only skyline"]   = ~ci_const & ci_sky
    df["neither"]        = ~ci_const & ~ci_sky

    nrows, ncols = len(mutsig_order), len(pop_model_order)
    default_figsize = (2.8 * ncols + 1.4, 2.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or default_figsize, squeeze=False)

    for i, mutsig in enumerate(mutsig_order):
        for j, pop_model in enumerate(pop_model_order):
            ax = axes[i, j]
            subset = df[(df["mutation_signal"] == mutsig) & (df["population_model"] == pop_model)]
            n = len(subset)

            matrix = pd.DataFrame(
                [[subset["only skyline"].sum(), subset["both"].sum()],
                 [subset["neither"].sum(),      subset["only constcoal"].sum()]],
                index=["Skyline ✓", "Skyline ✗"],
                columns=["Constcoal ✗", "Constcoal ✓"],
                dtype=float
            ) / max(n, 1)

            annot = (matrix * 100).map(lambda v: f"{v:.0f}%")
            sns.heatmap(matrix, ax=ax, vmin=0, vmax=1, annot=annot, fmt="",
                        annot_kws={"fontsize": 11, "fontweight": "bold", "color": "white"},
                        cmap=cmap_r, linewidths=0.5, cbar=False)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(title_map.get(pop_model, pop_model), fontsize=12)
            if j == 0:
                ax.set_ylabel(mutsig_labels[mutsig], fontsize=12)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")

    suptitle = title if title is not None else f"Root height CI coverage — {sampling}"
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    sm      = plt.cm.ScalarMappable(cmap=cmap_r, norm=plt.Normalize(vmin=0, vmax=100))
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_label(cbar_label, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    plt.show()


def plot_ci_coverage_heatmap_nodes(tree_metrics_combined, sampling, mode="height",
                                    title=None, figsize=None, save_path=None,
                                    cbar_label="Fraction of nodes"):
    """
    CI coverage contingency heatmap from full tree_metrics_combined (wide format).

    Same layout as plot_ci_coverage_heatmap, but counts all nodes of all
    replicates instead of only the root.

    mode="height": node height CI coverage, internal nodes only (tips excluded via NaN)
    mode="bl":     branch length CI coverage, non-root nodes only (root excluded via NaN)

    Parameters
    ----------
    tree_metrics_combined : DataFrame
        One row per (replicate, node), e.g. the output of load_node_errors().
        Needs sampling, mutation_signal, population_model and
        {constcoal,skyline}_height_inside_ci or {constcoal,skyline}_bl_inside_ci.
    sampling : str
        Sampling scheme to plot.
    mode : {"height", "bl"}
        Which CI coverage to show (see above).
    title : str, optional
        Suptitle; defaults to a description of mode plus the sampling scheme.
    figsize : tuple, optional
        Override the default figure size.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    cbar_label : str
        Colorbar label.

    Raises
    ------
    ValueError
        If mode is not "height" or "bl".
    """
    if mode == "height":
        const_col = "constcoal_height_inside_ci"
        sky_col   = "skyline_height_inside_ci"
        default_title = "Node height CI coverage (internal nodes)"
    elif mode == "bl":
        const_col = "constcoal_bl_inside_ci"
        sky_col   = "skyline_bl_inside_ci"
        default_title = "Branch length CI coverage (non-root nodes)"
    else:
        raise ValueError("mode must be 'height' or 'bl'")

    mutsig_order    = ["low", "med", "high"]
    mutsig_labels   = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}
    pop_model_order = ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]
    title_map       = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                       "uniform": "Uniform", "bottleneck": "Bottleneck"}

    _cmap_colors = ["#c5dfc0", "#7db87a", "#4e9a8a", "#397398", "#00181E"]
    cmap_r = plt.matplotlib.colors.LinearSegmentedColormap.from_list("green_cmap", _cmap_colors).reversed()

    df = tree_metrics_combined[tree_metrics_combined["sampling"] == sampling].copy()
    df = df.dropna(subset=[const_col, sky_col])
    ci_const = df[const_col].astype(bool)
    ci_sky   = df[sky_col].astype(bool)
    df["both"]           = ci_const & ci_sky
    df["only constcoal"] = ci_const & ~ci_sky
    df["only skyline"]   = ~ci_const & ci_sky
    df["neither"]        = ~ci_const & ~ci_sky

    nrows, ncols = len(mutsig_order), len(pop_model_order)
    default_figsize = (2.8 * ncols + 1.4, 2.6 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or default_figsize, squeeze=False)

    for i, mutsig in enumerate(mutsig_order):
        for j, pop_model in enumerate(pop_model_order):
            ax = axes[i, j]
            subset = df[(df["mutation_signal"] == mutsig) & (df["population_model"] == pop_model)]
            n = max(len(subset), 1)

            matrix = pd.DataFrame(
                [[subset["only skyline"].sum(), subset["both"].sum()],
                 [subset["neither"].sum(),      subset["only constcoal"].sum()]],
                index=["Skyline ✓", "Skyline ✗"],
                columns=["Constcoal ✗", "Constcoal ✓"],
                dtype=float
            ) / n

            annot = (matrix * 100).map(lambda v: f"{v:.0f}%")
            sns.heatmap(matrix, ax=ax, vmin=0, vmax=1, annot=annot, fmt="",
                        annot_kws={"fontsize": 11, "fontweight": "bold", "color": "white"},
                        cmap=cmap_r, linewidths=0.5, cbar=False)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(title_map.get(pop_model, pop_model), fontsize=12)
            if j == 0:
                ax.set_ylabel(mutsig_labels[mutsig], fontsize=12)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelsize=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=90, va="center")

    suptitle = title if title is not None else f"{default_title} — {sampling}"
    fig.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    sm      = plt.cm.ScalarMappable(cmap=cmap_r, norm=plt.Normalize(vmin=0, vmax=100))
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.65])
    cbar    = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_label(cbar_label, fontsize=13)
    cbar.ax.tick_params(labelsize=12)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    plt.show()


### PLOTTING

def plot_population_summary_ax(ax, skyline_all_times, skyline_all_medians, constant_all_estimates,
                               true_traj,
                               color_exp="#a6444f", color_sky="#80557e", color_const="#397398",
                               time_horizon=0, y_range = None, mode = 'both'):
    """
    Draw the true population size and the across-replicate median and 95%
    interval of the estimated population size onto ax.

    Fallback used by plot_population_summary and
    plot_combined_population_and_error when no precomputed trajectory_df is
    given. Each replicate's skyline step function is evaluated on a shared
    time grid; the median and 2.5/97.5 percentiles across replicates are
    drawn. For constcoal, the median and 2.5/97.5 percentiles of the
    per-replicate estimates are drawn as a horizontal line and band.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    skyline_all_times : list of list of float
        Per replicate, the end times of the skyline intervals (time before
        present, increasing; last entry = root height).
    skyline_all_medians : list of list of float
        Per replicate, the median population size of each skyline interval.
    constant_all_estimates : list of float
        Per replicate, the median constant population size estimate.
    true_traj : callable
        t -> true population size, e.g. from get_true_traj().
    color_exp, color_sky, color_const : str
        Colours of the true, skyline and constcoal curves.
    time_horizon : float
        Plot up to this time before present; 0 means up to the oldest root.
    y_range : tuple, optional
        y-axis limits.
    mode : {"both", "skyline", "constcoal"}
        Which estimates to draw.
    """
    # Compute root height
    root_height = max([max(times) for times in skyline_all_times])
    t_max = root_height if time_horizon == 0 else min(time_horizon, root_height)
    t_vals = np.linspace(0, t_max, 1000)
    N_true = np.array([true_traj(t) for t in t_vals])

    # Interpolate all skyline estimates onto a common grid
    interpolated = []
    for times, medians in zip(skyline_all_times, skyline_all_medians):
        step_times = [0.0] + times[:-1]
        step_vals = np.zeros_like(t_vals)
        for start, end, val in zip(step_times, times, medians):
            mask = (t_vals >= start) & (t_vals < end)
            step_vals[mask] = val
        step_vals[t_vals >= times[-1]] = medians[-1]
        interpolated.append(step_vals)

    interpolated = np.array(interpolated)
    skyline_median = np.median(interpolated, axis=0)
    skyline_lower = np.percentile(interpolated, 2.5, axis=0)
    skyline_upper = np.percentile(interpolated, 97.5, axis=0)

    # Plot
    ax.plot(t_vals, N_true, color=color_exp, linewidth=2, label="True Population Size", zorder = 6)

    const_median = np.median(constant_all_estimates)
    const_lower = np.percentile(constant_all_estimates, 2.5)
    const_upper = np.percentile(constant_all_estimates, 97.5)

    if mode == 'both':
        ax.hlines(const_median, 0, t_max, color=color_const, linestyle=':', label="Constant Estimate", linewidth = 2, zorder = 5)
        ax.fill_between(t_vals, const_lower, const_upper, color=color_const, alpha=0.3, zorder = 4)

        ax.plot(t_vals, skyline_median, color=color_sky, linestyle='--', label="Skyline Estimate", linewidth = 2, zorder = 5)
        ax.fill_between(t_vals, skyline_lower, skyline_upper, color=color_sky, alpha=0.3, zorder = 4)
    elif mode == 'skyline':
        ax.plot(t_vals, skyline_median, color=color_sky, linestyle='--', label="Skyline Estimate", linewidth = 2, zorder = 5)
        ax.fill_between(t_vals, skyline_lower, skyline_upper, color=color_sky, alpha=0.3, zorder = 4)
    elif mode == 'constcoal':
        ax.hlines(const_median, 0, t_max, color=color_const, linestyle=':', label="Constant Estimate", linewidth = 2, zorder = 5)
        ax.fill_between(t_vals, const_lower, const_upper, color=color_const, alpha=0.3, zorder = 4)
    
    if y_range:
        ax.set_ylim(y_range)
    
    ax.set_xlabel("Time before present")
    ax.set_ylabel("Population Size")
    ax.invert_xaxis()

def _add_skyline_changepoint_strip(ax, subset, x_range=None, color_sky="#80557e",
                                    dot_size=25, grey_alpha=0.3, y_frac=0.9):
    """
    Plot skyline interval change points near the top of ax.

    Each replicate's change points are drawn as small grey dots on one
    horizontal strip; the median change point per interval across
    replicates is drawn as a larger dot in color_sky.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on; its current y-limits determine the strip height.
    subset : DataFrame
        Replicates of one scenario, with a skyline_times column (list of
        interval end times per replicate; the last entry, the root height,
        is dropped).
    x_range : tuple, optional
        Unused; kept for call compatibility.
    color_sky : str
        Colour of the median change points.
    dot_size : float
        Marker size of the per-replicate dots (median dots are 1.5× larger).
    grey_alpha : float
        Transparency of the per-replicate dots.
    y_frac : float
        Vertical position of the strip as a fraction of the y-range.
    """
    all_times_arr = np.array(list(subset["skyline_times"]))[:, :-1]  # drop last (root height)

    ylims = ax.get_ylim()
    y_pos = ylims[0] + (ylims[1] - ylims[0]) * y_frac

    for row in all_times_arr:
        ax.scatter(row, np.full(len(row), y_pos), color="darkgrey", s=dot_size, alpha=grey_alpha, zorder=4, linewidths=0, clip_on=True)

    medians = np.median(all_times_arr, axis=0)
    ax.scatter(medians, np.full(len(medians), y_pos), color=color_sky, s=dot_size * 1.5, zorder=5, linewidths=0, clip_on=True)


def plot_population_summary(path_info_df, sampling, x_range=None, title=None, y_range=None,
                            add_samples=False, mode='both', pop_models=None, mut_signals=None,
                            show_changepoints=True, changepoint_dot_size=25, changepoint_grey_alpha=0.3,
                            x_label=None, y_label=None,
                            trajectory_df=None,
                            show_pop_model_labels=True, show_mutsig_labels=True, show_legend=True,
                            figsize=None,
                            label_fontsize=15, tick_fontsize=13, legend_fontsize=16,
                            save_path=None):
    """
    Summary plot per condition (mutation signal × population model) for a given sampling type,
    showing median and 95% CI for skyline and constant-coalescent estimates.

    Rows = mutation signal, columns = population model. Each panel shows the
    true population size trajectory and the estimated trajectories, plus
    (optionally) the skyline change points of every replicate.

    Parameters
    ----------
    path_info_df : DataFrame
        One row per replicate, e.g. the output of load_pop_summaries(). Needs
        sampling, population_model, mutation_signal, the true population
        parameters (see get_true_traj) and, unless trajectory_df is given,
        skyline_times, skyline_medians and coalescent_median.
    sampling : str
        Sampling scheme to plot.
    x_range : tuple, optional
        (min, max) time before present; the max also limits the plotted time.
    title : str, optional
        Suptitle; defaults to the sampling scheme.
    y_range : tuple, optional
        y-axis limits.
    add_samples : bool
        If True, also draw individual posterior trajectories in light grey.
        Needs skyline_samples, coalescent_samples (and optionally
        skyline_times_samples) in path_info_df.
    mode : {"both", "skyline", "constcoal"}
        Which estimates to draw.
    pop_models, mut_signals : str or list of str, optional
        Population models (columns) and mutation signals (rows) to show.
        Default: all population models in the data, and low/med/high.
    show_changepoints : bool
        Draw the skyline change point strip (see _add_skyline_changepoint_strip).
    changepoint_dot_size, changepoint_grey_alpha : float
        Styling of the change point strip.
    x_label, y_label : str, optional
        Axis labels.
    trajectory_df : DataFrame, optional
        Precomputed trajectories from load_pop_trajectories(). If given, its
        median and HPD columns (skyline_*/constcoal_* median, hpd_lower,
        hpd_upper) are plotted; otherwise they are computed from
        path_info_df via plot_population_summary_ax.
    show_pop_model_labels, show_mutsig_labels, show_legend : bool
        Toggle column titles, row labels and the figure legend.
    figsize : tuple, optional
        Override the default figure size.
    label_fontsize, tick_fontsize, legend_fontsize : float
        Font sizes.
    save_path : str or Path, optional
        If given, save the figure (see save_vector_figure) instead of
        showing it.
    """
    if pop_models is None:
        pop_models = sorted(path_info_df["population_model"].unique())
    if mut_signals is None:
        mut_signals = ["low", "med", "high"]
    if isinstance(pop_models, str):
        pop_models = [pop_models]
    if isinstance(mut_signals, str):
        mut_signals = [mut_signals]
    ncols, nrows = len(pop_models), len(mut_signals)

    title_map   = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                   "uniform": "Uniform", "bottleneck": "Bottleneck"}
    mutsig_labels = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}

    df = path_info_df[path_info_df["sampling"] == sampling]
    time_horizon = x_range[1] if x_range else 0

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (5 * ncols, 3.5 * nrows), squeeze=False)

    for i, mut_sig in enumerate(mut_signals):
        for j, pop_model in enumerate(pop_models):
            ax = axes[i][j]
            subset = df[
                (df["mutation_signal"] == mut_sig) &
                (df["population_model"] == pop_model)
            ]

            if subset.empty:
                continue

            true_traj = get_true_traj(pop_model, subset.iloc[0])
            skyline_all_times = subset["skyline_times"].tolist()
            skyline_all_medians = subset["skyline_medians"].tolist()
            constant_all_estimates = subset["coalescent_median"].tolist()

            if add_samples:
                skyline_sample_dfs   = subset["skyline_samples"]
                skyline_times_sample_dfs = subset["skyline_times_samples"] if "skyline_times_samples" in subset.columns else None
                constant_samples = subset["coalescent_samples"]

                for k, samples_df in enumerate(skyline_sample_dfs):
                    times_df = skyline_times_sample_dfs.iloc[k] if skyline_times_sample_dfs is not None else None
                    for row_index, (l, row) in enumerate(samples_df.iterrows()):
                        sky_times = list(times_df.iloc[row_index]) if times_df is not None else skyline_all_times[k]
                        plot_population_trajectories_ax(
                            ax,
                            skyline_times=sky_times,
                            skyline_medians=row,
                            constant_pop_estimate=constant_samples.iloc[k].iloc[row_index],
                            true_traj=true_traj,
                            alpha=0.01,
                            first_plot=False,
                            time_horizon=time_horizon,
                            color_const='lightgray',
                            color_sky='lightgray',
                            mode=mode,
                        )

            if trajectory_df is not None:
                traj_sub = trajectory_df[
                    (trajectory_df["sampling"] == sampling) &
                    (trajectory_df["mutation_signal"] == mut_sig) &
                    (trajectory_df["population_model"] == pop_model)
                ]
                if not traj_sub.empty:
                    t_vals = traj_sub["time"].values
                    color_exp  = "#a6444f"
                    color_sky  = "#80557e"
                    color_const = "#397398"
                    t_max = x_range[1] if x_range else t_vals.max()
                    mask = t_vals <= t_max
                    t_plot = t_vals[mask]
                    t_smooth = np.linspace(0, t_max, 1000)
                    N_true = np.array([true_traj(t) for t in t_smooth])
                    ax.plot(t_smooth, N_true, color=color_exp, linewidth=2, label="True Population Size", zorder=6)
                    if mode in ('both', 'skyline'):
                        ax.plot(t_plot, traj_sub["skyline_median"].values[mask],
                                color=color_sky, linestyle='--', label="Skyline Estimate", linewidth=2, zorder=5)
                        ax.fill_between(t_plot,
                                        traj_sub["skyline_hpd_lower"].values[mask],
                                        traj_sub["skyline_hpd_upper"].values[mask],
                                        color=color_sky, alpha=0.3, zorder=4)
                    if mode in ('both', 'constcoal'):
                        ax.plot(t_plot, traj_sub["constcoal_median"].values[mask],
                                color=color_const, linestyle=':', label="Constant Estimate", linewidth=2, zorder=5)
                        ax.fill_between(t_plot,
                                        traj_sub["constcoal_hpd_lower"].values[mask],
                                        traj_sub["constcoal_hpd_upper"].values[mask],
                                        color=color_const, alpha=0.3, zorder=4)
                    if y_range:
                        ax.set_ylim(y_range)
                    ax.invert_xaxis()
            else:
                plot_population_summary_ax(
                    ax,
                    skyline_all_times=skyline_all_times,
                    skyline_all_medians=skyline_all_medians,
                    constant_all_estimates=constant_all_estimates,
                    true_traj=true_traj,
                    time_horizon=time_horizon,
                    y_range=y_range,
                    mode=mode,
                )

            if x_range:
                ax.set_xlim(x_range[1], x_range[0])  # inverted x-axis

            if show_changepoints:
                _add_skyline_changepoint_strip(ax, subset, x_range=x_range,
                                               dot_size=changepoint_dot_size,
                                               grey_alpha=changepoint_grey_alpha)

            ax.tick_params(labelsize=tick_fontsize)
            ax.set_xlabel(x_label or "Time before present", fontsize=label_fontsize)
            ax.set_ylabel(y_label or "Population Size", fontsize=label_fontsize)
            if i == 0 and show_pop_model_labels:
                ax.set_title(title_map.get(pop_model, pop_model), fontsize=15)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if add_samples:
        if mode == 'skyline':
            handles.append(Line2D([0], [0], color='lightgrey', linestyle='--'))
            labels.append('Sample Skyline Trajectories')
        elif mode == 'constcoal':
            handles.append(Line2D([0], [0], color='lightgrey', linestyle=':'))
            labels.append('Sample Constant Trajectories')
    if show_changepoints:
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgrey',
                               markersize=8, label='Skyline change points (per tree)'))
        labels.append('Skyline change points (per tree)')
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#80557e',
                               markersize=10, label='Median change point'))
        labels.append('Median change point')
    if show_legend:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.99),
                   ncol=len(handles), fontsize=legend_fontsize, frameon=False,
                   handletextpad=0.4, columnspacing=1.2)

    suptitle = title if title is not None else sampling
    plt.suptitle(suptitle, fontsize=17)
    top = 0.96 if (suptitle or show_legend) else 1.0
    plt.tight_layout(rect=[0, 0, 1, top])

    # Mutsig row labels left of the y-axis, matching plot_height_vs_popsize_error style
    if show_mutsig_labels:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for i, mut_sig in enumerate(mut_signals):
            ax = axes[i][0]
            ylabel_bb = ax.yaxis.label.get_window_extent(renderer)
            x_fig = inv.transform((ylabel_bb.x0, ylabel_bb.y0 + ylabel_bb.height / 2))[0]
            pos = ax.get_position()
            y_fig = (pos.y0 + pos.y1) / 2
            fig.text(x_fig - 0.01, y_fig, mutsig_labels.get(mut_sig, mut_sig),
                     fontsize=15, ha="center", va="center", rotation=90,
                     transform=fig.transFigure)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    else:
        plt.show()


def plot_population_trajectories_ax(ax, skyline_times, skyline_medians, constant_pop_estimate,
                                     true_traj,
                                     color_exp="#a6444f", color_sky="#80557e", color_const="#397398",
                                     alpha=0.4, label_prefix="", first_plot=True, time_horizon=0, y_range = (0, 6000), mode = 'both'):
    """
    Draw one population size trajectory (one replicate or one posterior
    sample) onto ax: the skyline as a step function and the constant
    estimate as a horizontal line.

    Used by plot_population_summary(add_samples=True) to draw many
    posterior samples in the background.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    skyline_times : list of float
        End times of the skyline intervals (increasing; last entry = root height).
    skyline_medians : list of float
        Population size of each skyline interval.
    constant_pop_estimate : float
        Constant population size estimate.
    true_traj : callable
        t -> true population size, e.g. from get_true_traj().
    color_exp, color_sky, color_const : str
        Colours of the true, skyline and constcoal curves.
    alpha : float
        Transparency of the estimated curves.
    label_prefix : str
        Unused; kept for call compatibility.
    first_plot : bool
        If True, also draw the true trajectory, add legend labels and set
        axis labels, limits and the inverted x-axis. If False, only the
        estimates are drawn (in light grey for mode="skyline"/"constcoal").
    time_horizon : float
        Plot up to this time before present; 0 means up to the root height.
    y_range : tuple, optional
        y-axis limits (only applied if first_plot).
    mode : {"both", "skyline", "constcoal"}
        Which estimates to draw when first_plot is False.
    """

    root_height = skyline_times[-1]
    t_max = root_height if time_horizon == 0 else min(time_horizon, root_height)
    t_vals = np.linspace(0, t_max, 1000)
    N_true = np.array([true_traj(t) for t in t_vals])

    # Stepwise skyline
    skyline_start_times = [0.0] + skyline_times[:-1]
    step_times, step_values = [], []
    for start, end, value in zip(skyline_start_times, skyline_times, skyline_medians):
        if start > t_max:
            break
        end = min(end, t_max)
        step_times.extend([start, end])
        step_values.extend([value, value])
    
    if first_plot:
        ax.plot(t_vals, N_true, color=color_exp, alpha=1, linewidth= 2, label="True Population Size", zorder = 4)
        ax.hlines(constant_pop_estimate, 0, t_max, color=color_const, linestyle=':', 
              alpha=alpha, label = "Constant Estimate")
        ax.plot(step_times, step_values, drawstyle='steps-post', linestyle='--', color=color_sky, 
            alpha=alpha, label="Skyline Estimate")
    else:
        if mode == 'both':
            ax.hlines(constant_pop_estimate, 0, t_max, color=color_const, linestyle=':', 
                alpha=alpha)
            ax.plot(step_times, step_values, drawstyle='steps-post', linestyle='--', color=color_sky, 
                alpha=alpha)
        elif mode == 'skyline':
            ax.plot(step_times, step_values, drawstyle='steps-post', linestyle='--', color='lightgrey', 
                alpha=alpha)
        elif mode == 'constcoal':
            ax.hlines(constant_pop_estimate, 0, t_max, color='lightgrey', linestyle=':', 
                alpha=alpha)

    if first_plot:
        if y_range: 
            ax.set_ylim(y_range)
        ax.set_xlabel("Time before present")
        ax.set_ylabel("Population Size")
        ax.invert_xaxis()


def plot_error_by_mutsig(root_height_df_wide, sampling,
                         y_col="height_rel_error_median",
                         y_label=None, y_range=None,
                         pop_models=None, figsize=None,
                         title=None, save_path=None,
                         violin=False, legend_inside=False,
                         show_pop_model_labels=True,
                         label_fontsize=14, tick_fontsize=14, legend_fontsize=14,
                         show_diff=False, diff_mode="relative", diff_y_range=None,
                         print_results=False):
    """
    Four subplots side by side (one per population model). For each mutation
    signal (low, med, high) two jittered scatter columns are shown — constcoal
    (blue) and skyline (purple) — with individual replicate values in grey and
    the per-group median overlaid as a larger coloured dot.

    y_col is the base column name without the model prefix, e.g.
    'height_rel_error_median'. The function looks up
    '{model}_{y_col}' in root_height_df_wide.

    Vertical bars show the 95% HPD across replicates. Optionally a third
    column per mutation signal shows the per-replicate difference
    constcoal − skyline (green).

    Parameters
    ----------
    root_height_df_wide : DataFrame
        One row per replicate (e.g. root rows of load_node_errors(), or the
        per-replicate population summary). Needs sampling, population_model,
        mutation_signal and the columns constcoal_{y_col} / skyline_{y_col}.
    sampling : str
        Sampling scheme to plot.
    y_col : str
        Base column name (see above).
    y_label : str, optional
        y-axis label; defaults to y_col.
    y_range : tuple, optional
        y-axis limits.
    pop_models : list of str, optional
        Population models (subplots) to show, in order.
    figsize : tuple, optional
        Override the default figure size.
    title : str, optional
        Suptitle; defaults to the sampling scheme.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    violin : bool
        Add a KDE "violin" within the HPD range behind each column.
    legend_inside : bool
        Put the legend in each subplot instead of above the figure.
    show_pop_model_labels : bool
        Show subplot titles.
    label_fontsize, tick_fontsize, legend_fontsize : float
        Font sizes.
    show_diff : bool
        Add the constcoal − skyline difference column.
    diff_mode : {"relative", "absolute"}
        "relative" plots (constcoal − skyline) / skyline on a secondary
        y-axis; "absolute" plots constcoal − skyline on the main axis.
    diff_y_range : tuple, optional
        Limits of the secondary y-axis (diff_mode="relative" only).
    print_results : bool
        Print n, median and 95% HPD per group to stdout.
    """
    colors      = {"constcoal": "#397398", "skyline": "#80557e"}
    diff_color  = "#338778"
    labels      = {"constcoal": "Const. coal.", "skyline": "Skyline"}
    mutsig_order    = ["low", "med", "high"]
    mutsig_labels   = {"low": "Low", "med": "Med.", "high": "High"}
    title_map       = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                       "uniform": "Uniform", "bottleneck": "Bottleneck"}
    pop_model_order = pop_models or ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]

    df = root_height_df_wide[root_height_df_wide["sampling"] == sampling].copy()

    # x positions: each mutsig gets a slot of width 1; constcoal left, skyline right
    offsets = {"constcoal": -0.12, "skyline": 0.12}
    diff_offset = 0.36
    x_base  = {ms: i * 1.5 for i, ms in enumerate(mutsig_order)}

    ncols = len(pop_model_order)
    fig, axes = plt.subplots(1, ncols, figsize=figsize or (4.5 * ncols, 4.5))
    if ncols == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, pop_model in zip(axes, pop_model_order):
        for model in ("constcoal", "skyline"):
            col = f"{model}_{y_col}"
            if col not in df.columns:
                continue
            color = colors[model]
            for ms in mutsig_order:
                sub = df[(df["population_model"] == pop_model) & (df["mutation_signal"] == ms)].dropna(subset=[col])
                x_center = x_base[ms] + offsets[model]
                jitter = rng.uniform(-0.06, 0.06, size=len(sub))
                xs = x_center + jitter
                ax.scatter(xs, sub[col].values, color="#aaaaaa", s=15,
                           alpha=0.8, zorder=2, linewidths=0)
                vals = sub[col].dropna().values
                if len(vals) >= 2:
                    hpd_lo, hpd_hi = hpd(np.sort(vals))
                    ax.plot([x_center, x_center], [hpd_lo, hpd_hi],
                            color=color, linewidth=1.5, zorder=3, solid_capstyle="round")
                    cap = 0.05
                    ax.plot([x_center - cap, x_center + cap], [hpd_lo, hpd_lo],
                            color=color, linewidth=1.5, zorder=3)
                    ax.plot([x_center - cap, x_center + cap], [hpd_hi, hpd_hi],
                            color=color, linewidth=1.5, zorder=3)
                    if violin:
                        kde = gaussian_kde(vals, bw_method=0.4)
                        ys = np.linspace(hpd_lo, hpd_hi, 200)
                        xs_kde = kde(ys)
                        xs_kde = xs_kde / xs_kde.max() * 0.12
                        ax.fill_betweenx(ys, x_center - xs_kde, x_center + xs_kde,
                                         color=color, alpha=0.35, zorder=3, linewidth=0)
                ax.scatter(x_center, sub[col].median(), color=color, s=75,
                           zorder=4, edgecolors="white", linewidths=0.5)

                if print_results:
                    med_of_medians = float(np.median(vals)) if len(vals) else np.nan
                    if len(vals) >= 2:
                        print(f"{pop_model:15s} {labels[model]:12s} {ms:4s} mutsig  "
                              f"n={len(vals):3d}  median={med_of_medians:.4g}  "
                              f"95% HPD=({hpd_lo:.4g}, {hpd_hi:.4g})")
                    else:
                        print(f"{pop_model:15s} {labels[model]:12s} {ms:4s} mutsig  "
                              f"n={len(vals):3d}  median={med_of_medians:.4g}  95% HPD=n/a (need >=2 trees)")

        # third dot: constcoal - skyline difference per tree
        col_cc = f"constcoal_{y_col}"
        col_sk = f"skyline_{y_col}"
        use_twin = show_diff and diff_mode == "relative"
        ax_diff = ax.twinx() if use_twin else None
        if ax_diff is not None:
            ax_diff.tick_params(axis="y", labelsize=tick_fontsize, colors=diff_color)
            ax_diff.spines["right"].set_color(diff_color)
            if diff_y_range:
                ax_diff.set_ylim(diff_y_range)

        if show_diff and col_cc in df.columns and col_sk in df.columns:
            target = ax_diff if use_twin else ax
            for ms in mutsig_order:
                sub = df[(df["population_model"] == pop_model) & (df["mutation_signal"] == ms)].dropna(subset=[col_cc, col_sk])
                diff_vals = ((sub[col_cc] - sub[col_sk]) / sub[col_sk]).values if diff_mode == "relative" else (sub[col_cc] - sub[col_sk]).values
                x_center = x_base[ms] + diff_offset
                jitter = rng.uniform(-0.06, 0.06, size=len(diff_vals))
                target.scatter(x_center + jitter, diff_vals, color="#aaaaaa", s=15,
                               alpha=0.8, zorder=2, linewidths=0)
                if len(diff_vals) >= 2:
                    hpd_lo, hpd_hi = hpd(np.sort(diff_vals))
                    target.plot([x_center, x_center], [hpd_lo, hpd_hi],
                                color=diff_color, linewidth=1.5, zorder=3, solid_capstyle="round")
                    cap = 0.05
                    target.plot([x_center - cap, x_center + cap], [hpd_lo, hpd_lo],
                                color=diff_color, linewidth=1.5, zorder=3)
                    target.plot([x_center - cap, x_center + cap], [hpd_hi, hpd_hi],
                                color=diff_color, linewidth=1.5, zorder=3)
                    if violin:
                        kde = gaussian_kde(diff_vals, bw_method=0.4)
                        ys = np.linspace(hpd_lo, hpd_hi, 200)
                        xs_kde = kde(ys)
                        xs_kde = xs_kde / xs_kde.max() * 0.12
                        target.fill_betweenx(ys, x_center - xs_kde, x_center + xs_kde,
                                             color=diff_color, alpha=0.35, zorder=3, linewidth=0)
                target.scatter(x_center, np.median(diff_vals), color=diff_color, s=75,
                               zorder=4, edgecolors="white", linewidths=0.5)

                if print_results:
                    diff_kind = "rel. diff." if diff_mode == "relative" else "abs. diff."
                    med_diff = float(np.median(diff_vals)) if len(diff_vals) else np.nan
                    if len(diff_vals) >= 2:
                        print(f"{pop_model:15s} {'Const-Sky ' + diff_kind:22s} {ms:4s} mutsig  "
                              f"n={len(diff_vals):3d}  median={med_diff:.4g}  "
                              f"95% HPD=({hpd_lo:.4g}, {hpd_hi:.4g})")
                    else:
                        print(f"{pop_model:15s} {'Const-Sky ' + diff_kind:22s} {ms:4s} mutsig  "
                              f"n={len(diff_vals):3d}  median={med_diff:.4g}  95% HPD=n/a (need >=2 trees)")

        tick_positions = [x_base[ms] + (diff_offset / 2 if show_diff else 0) for ms in mutsig_order]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([mutsig_labels[ms] for ms in mutsig_order], fontsize=tick_fontsize)
        ax.set_xlabel("Mutation signal", fontsize=label_fontsize)
        if show_pop_model_labels:
            ax.set_title(title_map.get(pop_model, pop_model), fontsize=label_fontsize + 1)
        ax.xaxis.grid(False)
        if y_range:
            ax.set_ylim(y_range)
        ax.yaxis.grid(True, color='lightgray', linewidth=0.8, alpha=0.8, zorder=5)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.tick_params(labelsize=tick_fontsize, labelleft=True)
        ax.set_ylabel(y_label or y_col, fontsize=label_fontsize + 1, labelpad=2)
        if use_twin:
            ax_diff.set_ylabel("Rel. diff.", fontsize=label_fontsize + 1, labelpad=4, color=diff_color)
    from matplotlib.lines import Line2D as _L2D
    handles = [_L2D([], [], marker="o", linestyle="none", markerfacecolor=c,
                    markeredgecolor="none", markersize=9, label=labels[m])
               for m, c in colors.items()]
    if show_diff and not (show_diff and diff_mode == "relative"):
        handles.append(_L2D([], [], marker="o", linestyle="none", markerfacecolor=diff_color,
                            markeredgecolor="none", markersize=9, label="Diff."))
    _leg_kwargs = dict(handles=handles, frameon=False, fontsize=legend_fontsize,
                       handlelength=0.0, handletextpad=0.6, columnspacing=0.8)
    suptitle = title if title is not None else sampling
    plt.suptitle(suptitle, fontsize=17)
    top = 0.93 if suptitle else 1.0
    plt.tight_layout(rect=[0, 0, 1, top])
    plt.subplots_adjust(wspace=0.35)

    if legend_inside:
        for ax_ in np.ravel(axes):
            ax_.legend(loc="lower right", ncol=2, bbox_to_anchor=(1.0, -0.038), **_leg_kwargs)
    else:
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01),
                   ncol=3, **_leg_kwargs)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    from IPython.display import display as _display
    _display(fig)
    plt.close(fig)


def plot_height_vs_popsize_error(root_height_df_wide, sampling,
                                  x_col="pop_diff_median", y_col="height_diff_median",
                                  x_range=None, y_range=None, alpha=0.6, s=10,
                                  title=None, x_label=None, y_label=None,
                                  pop_models=None, mutsigs=None,
                                  figsize=None, show_zero_lines=True,
                                  legend_inside=True,
                                  color_by_time=False,
                                  time_col="height_sim",
                                  log_colorbar=False,
                                  cbar_label_constcoal=None,
                                  cbar_label_skyline=None,
                                  save_path=None):
    """
    Scatterplot of y_col vs x_col for constcoal and skyline, one subplot per
    population model (columns) × mutation signal (rows), filtered by sampling type.
    Draws per-replicate HPD intervals as error bars.
    Column names are expected with model prefix: {model}_{col}_median/hpd_lower/hpd_upper.

    Parameters
    ----------
    root_height_df_wide : DataFrame
        One row per replicate, e.g. the root rows of load_node_errors().
        Needs sampling, population_model, mutation_signal and the x/y
        columns with model prefix or suffix ({model}_{col} or {col}_{model}).
    sampling : str
        Sampling scheme to plot.
    x_col, y_col : str
        Median columns without model prefix, e.g. "pop_diff_median". HPD
        columns are found by replacing "_median" with "_hpd_lower/_hpd_upper";
        if they are missing, a plain scatter is drawn.
    x_range, y_range : tuple, optional
        Axis limits.
    alpha : float
        Marker transparency (error bars are drawn more transparent).
    s : float
        Marker size for the plain scatter (no HPD columns).
    title : str, optional
        Suptitle; default to sampling type.
    x_label, y_label : str, optional
        Axis labels; default to the column name.
    pop_models, mutsigs : list of str, optional
        Population models (columns) and mutation signals (rows) to show.
    figsize : tuple, optional
        Override default figure size.
    show_zero_lines : bool
        Draw dashed zero reference lines on each subplot.
    legend_inside : bool
        Put a legend in each subplot instead of one above the figure.
    color_by_time : bool
        Colour points by time_col (one colormap per model, two colorbars).
    time_col : str
        Column used for colouring, e.g. the true root height "height_sim".
    log_colorbar : bool
        Use a logarithmic colour scale.
    cbar_label_constcoal, cbar_label_skyline : str, optional
        Colorbar labels.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    """
    df = root_height_df_wide[root_height_df_wide["sampling"] == sampling].copy()

    mutsig_order       = mutsigs    if mutsigs    is not None else ["low", "med", "high"]
    growth_model_order = pop_models if pop_models is not None else ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]

    mutsig_labels = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}
    title_map     = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                     "uniform": "Uniform", "bottleneck": "Bottleneck"}
    legend_labels = {"constcoal": "Const. coal.", "skyline": "Skyline"}
    colors        = {"constcoal": "#397398", "skyline": "#80557e"}
    markers       = {"constcoal": "o",       "skyline": "^"}
    model_cmaps   = {
        "constcoal": plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                         "constcoal_time", ["#b5d2f2", "#7394c2", "#397398"]),
        "skyline":   plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                         "skyline_time",   ["#a6444f", "#d991b4", "#80557e"]),
    }

    if color_by_time:
        df_all = root_height_df_wide[root_height_df_wide["sampling"] == sampling]
        t_min_pos = max(df_all[time_col].min(), 1)
        t_max     = df_all[time_col].max()
        cnorm = plt.matplotlib.colors.LogNorm(vmin=t_min_pos, vmax=t_max) if log_colorbar \
                else plt.Normalize(vmin=t_min_pos, vmax=t_max)

    nrows, ncols = len(mutsig_order), len(growth_model_order)
    default_figsize = (4 * ncols, 3.2 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize or default_figsize,
                             squeeze=False)

    def _col(model, base):
        if f"{model}_{base}" in df.columns:
            return f"{model}_{base}"
        if f"{base}_{model}" in df.columns:
            return f"{base}_{model}"
        return None

    def _lo(base): return base.replace("_median", "_hpd_lower")
    def _hi(base): return base.replace("_median", "_hpd_upper")

    for i, mutsig in enumerate(mutsig_order):
        for j, pop_model in enumerate(growth_model_order):
            ax = axes[i, j]
            subset = df[(df["mutation_signal"] == mutsig) & (df["population_model"] == pop_model)]

            if subset.empty:
                ax.axis("off")
                continue

            for model, color in colors.items():
                xm_key = _col(model, x_col)
                ym_key = _col(model, y_col)
                if xm_key is None or ym_key is None:
                    continue

                xm = subset[xm_key]
                ym = subset[ym_key]

                xl_key = _col(model, _lo(x_col))
                xh_key = _col(model, _hi(x_col))
                yl_key = _col(model, _lo(y_col))
                yh_key = _col(model, _hi(y_col))

                has_x_ci = xl_key is not None and xh_key is not None
                has_y_ci = yl_key is not None and yh_key is not None

                if color_by_time:
                    t_vals = subset[time_col].clip(lower=t_min_pos)
                    point_colors = [model_cmaps[model](cnorm(t)) for t in t_vals]
                else:
                    point_colors = [color] * len(xm)

                if has_x_ci or has_y_ci:
                    xerr = ([xm - subset[xl_key], subset[xh_key] - xm] if has_x_ci else None)
                    yerr = ([ym - subset[yl_key], subset[yh_key] - ym] if has_y_ci else None)
                    for idx, (xi, yi, c) in enumerate(zip(xm, ym, point_colors)):
                        xe = ([xm.iloc[idx] - subset[xl_key].iloc[idx]],
                              [subset[xh_key].iloc[idx] - xm.iloc[idx]]) if has_x_ci else (None, None)
                        ye = ([ym.iloc[idx] - subset[yl_key].iloc[idx]],
                              [subset[yh_key].iloc[idx] - ym.iloc[idx]]) if has_y_ci else (None, None)
                        eb = ax.errorbar(xi, yi,
                                         xerr=([[xe[0][0]], [xe[1][0]]] if xe[0] is not None else None),
                                         yerr=([[ye[0][0]], [ye[1][0]]] if ye[0] is not None else None),
                                         fmt=markers[model], color=c, ecolor=c,
                                         elinewidth=0.8, capsize=2, markersize=4,
                                         label=legend_labels[model] if idx == 0 else "_nolegend_")
                        eb[0].set_alpha(alpha)
                        for line in eb[1]: line.set_alpha(0.15)
                        for line in eb[2]: line.set_alpha(0.15)
                else:
                    ax.scatter(xm, ym, color=point_colors, marker=markers[model],
                               alpha=alpha, s=s, label=legend_labels[model])

            ax.grid(True, color="lightgray", linewidth=0.4, alpha=0.5, zorder=0)
            if show_zero_lines:
                ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
                ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
            if x_range: ax.set_xlim(x_range)
            if y_range: ax.set_ylim(y_range)
            ax.tick_params(labelsize=13)

            ax.set_xlabel(x_label or x_col, fontsize=14)

            if i == 0:
                ax.set_title(title_map.get(pop_model, pop_model), fontsize=15)
            ax.set_ylabel(y_label or y_col, fontsize=14, labelpad=2)
            if legend_inside:
                leg = ax.legend(loc="upper right", frameon=True, fontsize=11,
                                markerscale=1.2, handletextpad=0.4, borderpad=0.4)
                leg.get_frame().set_linewidth(0.6)
                leg.get_frame().set_edgecolor("gray")

    if not legend_inside:
        handles = []
        for m, c in colors.items():
            marker_line = Line2D([0], [0], marker=markers[m], color=c, markersize=9,
                                 linestyle="none", alpha=1.0)
            caplines    = (Line2D([0], [0], color=c, linewidth=1.0),
                           Line2D([0], [0], color=c, linewidth=1.0))
            barlines    = (LineCollection([[[0, -0.3], [0, 0.3]]], colors=[c], linewidths=[0.8]),
                           LineCollection([[[-0.3, 0], [0.3, 0]]], colors=[c], linewidths=[0.8]))
            handle      = ErrorbarContainer((marker_line, caplines, barlines),
                                            has_xerr=True, has_yerr=True,
                                            label=legend_labels[m])
            handles.append(handle)
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.97),
                   ncol=2, frameon=False, fontsize=15, borderpad=0.8, handletextpad=0.6)

    suptitle = title if title is not None else sampling
    plt.suptitle(suptitle, fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.94 if not legend_inside else 0.97])

    # Place mutsig labels just left of the ylabel, anchored to its rendered position.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    for j, mutsig in enumerate(mutsig_order):
        ax = axes[j][0]
        ylabel_bb = ax.yaxis.label.get_window_extent(renderer)
        x_fig = inv.transform((ylabel_bb.x0, ylabel_bb.y0 + ylabel_bb.height / 2))[0]
        pos = ax.get_position()
        y_fig = (pos.y0 + pos.y1) / 2
        fig.text(x_fig - 0.01, y_fig, mutsig_labels.get(mutsig, mutsig),
                 fontsize=15, ha="center", va="center", rotation=90,
                 transform=fig.transFigure)

    if color_by_time:
        fig.subplots_adjust(right=0.86)
        cbar_const = fig.colorbar(plt.cm.ScalarMappable(cmap=model_cmaps["constcoal"], norm=cnorm),
                                   cax=fig.add_axes([0.88, 0.15, 0.015, 0.7]))
        cbar_const.set_label(cbar_label_constcoal or f"True root height — Const. Coalescent",
                             rotation=270, labelpad=18, fontsize=14)
        cbar_const.ax.tick_params(labelsize=13)
        cbar_sky = fig.colorbar(plt.cm.ScalarMappable(cmap=model_cmaps["skyline"], norm=cnorm),
                                cax=fig.add_axes([0.97, 0.15, 0.015, 0.7]))
        cbar_sky.set_label(cbar_label_skyline or f"True root height — Skyline",
                           rotation=270, labelpad=18, fontsize=14)
        cbar_sky.ax.tick_params(labelsize=13)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    plt.show()


def plot_constcoal_vs_skyline_error(root_height_df_wide, sampling,
                                     error_col,
                                     error_label=None,
                                     x_range=None, y_range=None,
                                     alpha=0.5, s=10,
                                     title=None, x_label=None, y_label=None,
                                     pop_models=None, mutsigs=None,
                                     figsize=None,
                                     color_by_time=False,
                                     time_col="height_sim",
                                     log_colorbar=False,
                                     cbar_label=None,
                                     show_pop_model_labels=True,
                                     show_mutsig_labels=True,
                                     label_fontsize=None, tick_fontsize=None,
                                     legend_fontsize=None, cbar_tick_fontsize=None,
                                     cbar_label_fontsize=None,
                                     save_path=None):
    """
    Scatter plot of constcoal error vs skyline error per replicate.
    Rows = mutation signal, columns = population model.
    Each point is one replicate; error bars show HPD intervals.
    The x=y diagonal is drawn as a black dashed line.

    Parameters
    ----------
    root_height_df_wide : DataFrame
        One row per replicate, e.g. the root rows of load_node_errors().
        Needs sampling, population_model, mutation_signal and
        constcoal_{error_col} / skyline_{error_col} (+ HPD columns).
    sampling : str
        Sampling scheme to plot.
    error_col : str
        Base column name without model prefix, e.g. 'height_diff_median'.
        HPD columns are inferred by replacing '_median' with '_hpd_lower/upper'.
        "_median" is appended if missing.
    error_label : str, optional
        Legend label; defaults to the median column name.
    x_range, y_range : tuple, optional
        Axis limits.
    alpha : float
        Marker transparency.
    s : float
        Marker size for the plain scatter (no HPD columns).
    title : str, optional
        Suptitle; defaults to the sampling scheme.
    x_label, y_label : str, optional
        Axis labels (x = skyline, y = constcoal).
    pop_models, mutsigs : str or list of str, optional
        Population models (columns) and mutation signals (rows) to show.
    figsize : tuple, optional
        Override the default figure size.
    color_by_time : bool
        Colour points by time_col, with one colorbar per subplot (colour
        scale computed per subplot).
    time_col : str
        Column used for colouring, e.g. the true root height "height_sim".
    log_colorbar : bool
        Use a logarithmic colour scale (with hand-picked ticks per population model).
    cbar_label : str, optional
        Colorbar label; defaults to "True root age".
    show_pop_model_labels, show_mutsig_labels : bool
        Toggle column titles and row labels.
    label_fontsize, tick_fontsize, legend_fontsize, cbar_tick_fontsize, cbar_label_fontsize : float, optional
        Font sizes; defaults depend on color_by_time.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    """
    df = root_height_df_wide[root_height_df_wide["sampling"] == sampling].copy()

    # font-size defaults: the color_by_time=True grids (with colorbars) were
    # tuned larger; color_by_time=False keeps the original, smaller sizes,
    # unless the caller passes an explicit value either way.
    if label_fontsize is None:      label_fontsize      = 17 if color_by_time else 14
    if tick_fontsize is None:       tick_fontsize        = 16 if color_by_time else 13
    if legend_fontsize is None:     legend_fontsize      = 14 if color_by_time else 11
    if cbar_tick_fontsize is None:  cbar_tick_fontsize   = 13 if color_by_time else 10

    mutsig_order       = mutsigs    if mutsigs    is not None else ["low", "med", "high"]
    growth_model_order = pop_models if pop_models is not None else ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]
    if isinstance(mutsig_order, str):       mutsig_order       = [mutsig_order]
    if isinstance(growth_model_order, str): growth_model_order = [growth_model_order]

    mutsig_labels = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}
    title_map     = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                     "uniform": "Uniform", "bottleneck": "Bottleneck"}
    color = "#555555"
    _cmap_colors = ["#c5dfc0", "#7db87a", "#4e9a8a", "#397398", "#00181E"]
    time_cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                    "green_cmap", _cmap_colors)

    # per-subplot norms computed inside the loop when color_by_time=True

    def _key(model, col):
        full = f"{model}_{col}"
        return full if full in df.columns else None

    med_col  = error_col if "_median" in error_col else f"{error_col}_median"
    lo_col   = med_col.replace("_median", "_hpd_lower")
    hi_col   = med_col.replace("_median", "_hpd_upper")

    nrows, ncols = len(mutsig_order), len(growth_model_order)
    # extra per-column width when color_by_time: each subplot carves its own
    # colorbar out of its axes, which needs horizontal room the plain grid doesn't.
    default_figsize = (5.2 * ncols if color_by_time else 4 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize or default_figsize,
                             squeeze=False)

    for i, mutsig in enumerate(mutsig_order):
        for j, pop_model in enumerate(growth_model_order):
            ax = axes[i, j]
            subset = df[(df["mutation_signal"] == mutsig) & (df["population_model"] == pop_model)]

            if subset.empty:
                ax.axis("off")
                continue

            xm_key = _key("skyline",   med_col)
            ym_key = _key("constcoal", med_col)
            xl_key = _key("skyline",   lo_col)
            xh_key = _key("skyline",   hi_col)
            yl_key = _key("constcoal", lo_col)
            yh_key = _key("constcoal", hi_col)

            if xm_key is None or ym_key is None:
                ax.axis("off")
                continue

            xm = subset[xm_key]
            ym = subset[ym_key]
            has_x_ci = xl_key is not None and xh_key is not None
            has_y_ci = yl_key is not None and yh_key is not None

            if color_by_time:
                t_min_pos = max(subset[time_col].min(), 1)
                t_max     = subset[time_col].max()
                cnorm = plt.matplotlib.colors.LogNorm(vmin=t_min_pos, vmax=t_max) if log_colorbar \
                        else plt.Normalize(vmin=t_min_pos, vmax=t_max)
                t_vals = subset[time_col].clip(lower=t_min_pos)
                point_colors = [time_cmap(cnorm(t)) for t in t_vals]
            else:
                cnorm = None
                point_colors = [color] * len(xm)

            if has_x_ci or has_y_ci:
                for idx, (xi, yi, c) in enumerate(zip(xm, ym, point_colors)):
                    xe = ([[xm.iloc[idx] - subset[xl_key].iloc[idx]],
                            [subset[xh_key].iloc[idx] - xm.iloc[idx]]] if has_x_ci else None)
                    ye = ([[ym.iloc[idx] - subset[yl_key].iloc[idx]],
                            [subset[yh_key].iloc[idx] - ym.iloc[idx]]] if has_y_ci else None)
                    eb = ax.errorbar(xi, yi, xerr=xe, yerr=ye,
                                     fmt="o", color=c, ecolor=c,
                                     elinewidth=0.8, capsize=2, markersize=4, alpha=alpha)
                    for line in eb[1]: line.set_alpha(0.15)
                    for line in eb[2]: line.set_alpha(0.15)
            else:
                ax.scatter(xm, ym, color=point_colors, s=s, alpha=alpha)

            # x = y diagonal
            ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1.0, zorder=1)

            ax.set_aspect("equal")
            ax.grid(True, color="lightgray", linewidth=0.4, alpha=0.5, zorder=0)
            if x_range: ax.set_xlim(x_range)
            if y_range: ax.set_ylim(y_range)
            ax.tick_params(labelsize=tick_fontsize)
            ax.set_xlabel(x_label or f"Skyline — {med_col}", fontsize=label_fontsize)
            ax.set_ylabel(y_label or f"Const. Coalescent — {med_col}", fontsize=label_fontsize, labelpad=-5 if color_by_time else 8)
            if i == 0 and show_pop_model_labels:
                ax.set_title(title_map.get(pop_model, pop_model), fontsize=label_fontsize + 1)

            marker_line = Line2D([0], [0], marker="o", color=color, markersize=5,
                                 linestyle="none", alpha=1.0)
            caplines    = (Line2D([0], [0], color=color, linewidth=1.0),
                           Line2D([0], [0], color=color, linewidth=1.0))
            barlines    = (LineCollection([[[0, -0.3], [0, 0.3]]], colors=[color], linewidths=[0.8]),
                           LineCollection([[[-0.3, 0], [0.3, 0]]], colors=[color], linewidths=[0.8]))
            handle = ErrorbarContainer((marker_line, caplines, barlines),
                                       has_xerr=True, has_yerr=True, label=error_label or med_col)
            ax.legend(handles=[handle], fontsize=legend_fontsize, frameon=False,
                      loc="upper left", borderpad=0.3, handletextpad=0.6,
                      handlelength=0.5, bbox_to_anchor=(0.0, 1.0))

            if color_by_time and cnorm is not None:
                # attach via a divider (not ax=ax) so the colorbar stays pinned to
                # ax's actual, aspect="equal"-shrunk box instead of its pre-aspect
                # cell — otherwise it can end up floating below the plot.
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad=0.06)
                cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=time_cmap, norm=cnorm), cax=cax)
                if log_colorbar:
                    _cbar_ticks = {
                        "uniform":       ([1e3, 1e4],   ["$10^3$", "$10^4$"]),
                        "bottleneck":    ([1e3, 5e3],   ["$10^3$", "$5\\times10^3$"]),
                        "expgrowthslow": ([3e2, 4e2],   ["$3\\times10^2$", "$4\\times10^2$"]),
                        "expgrowthfast": ([2e2, 2.5e2], ["$2\\times10^2$", "$2.5\\times10^2$"]),
                    }
                    ticks, labels = _cbar_ticks.get(pop_model, ([t_min_pos, t_max], [f"{t_min_pos:.0f}", f"{t_max:.0f}"]))
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(labels)
                    cbar.ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
                cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
                cbar.set_label(cbar_label or "True root age", rotation=270,
                               labelpad=20, fontsize=cbar_label_fontsize or label_fontsize)

    suptitle = title if title is not None else sampling
    plt.suptitle(suptitle, fontsize=17)
    top = 0.97 if suptitle else 1.0
    # multi-row grids need a reserved left margin for the row (mutsig) labels
    # added below; leave the single-subplot layout (nrows==ncols==1) and the
    # non-colored layout (unchanged from before these spacing tweaks) alone.
    left = 0.03 if (show_mutsig_labels and nrows > 1 and color_by_time) else 0.0
    plt.tight_layout(rect=[left, 0, 1, top])
    if color_by_time and (nrows > 1 or ncols > 1):
        # colorbars are carved out of each subplot's own axes (via the divider
        # above), so columns need more breathing room when they're present.
        fig.subplots_adjust(hspace=0.28, wspace=0.5)

    if show_mutsig_labels:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for i, mutsig in enumerate(mutsig_order):
            ax = axes[i][0]
            ylabel_bb = ax.yaxis.label.get_window_extent(renderer)
            x_fig = inv.transform((ylabel_bb.x0, ylabel_bb.y0 + ylabel_bb.height / 2))[0]
            pos = ax.get_position()
            y_fig = (pos.y0 + pos.y1) / 2
            fig.text(x_fig - (0.008 if color_by_time else 0.006), y_fig, mutsig_labels.get(mutsig, mutsig),
                     fontsize=label_fontsize + 1, ha="center", va="center", rotation=90,
                     transform=fig.transFigure)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    from IPython.display import display as _display
    _display(fig)
    plt.close(fig)


def get_true_traj(pop_model, row):
    """
    Return true population trajectory function for a given pop model.

    Parameters
    ----------
    pop_model : str
        "expgrowthfast"/"expgrowthslow" (exponential decline back in time,
        floored at ANCESTRAL_POP_SIZE), any name starting with "bottleneck"
        (constant size with a bottleneck between bottleneck_start and
        bottleneck_end), or anything else (constant size, "uniform").
    row : pandas.Series or dict
        Population parameters of the scenario: present_pop_size, and
        growth_rate or bottleneck_size/bottleneck_start/bottleneck_end
        depending on the model.

    Returns
    -------
    callable
        t -> true population size at time t before present.
    """
    if pop_model in ("expgrowthfast", "expgrowthslow"):
        return lambda t: max(row["present_pop_size"] * np.exp(-row["growth_rate"] * t), ANCESTRAL_POP_SIZE)
    elif pop_model.startswith("bottleneck"):
        ps, bs, bstart, bend = row["present_pop_size"], row["bottleneck_size"], row["bottleneck_start"], row["bottleneck_end"]
        def bottleneck_traj(t):
            if t <= bstart:
                return ps
            elif t <= bend:
                return bs
            else:
                return ps
        return bottleneck_traj
    else:  # uniform
        return lambda t: row["present_pop_size"]


def plot_binned_error_timecoloring(time_bin_df, sampling,
                                   x_col="pop_rel_error",
                                   y_col="bl_rel_error",
                                   x_range=None, y_range=None,
                                   show_errorbars=True,
                                   min_nodes_per_bin=10,
                                   log_colorbar=False,
                                   time_col="bin_center",
                                   x_label=None, y_label=None,
                                   cbar_label="Time before present",
                                   cbar_label_constcoal=None, cbar_label_skyline=None,
                                   pop_models=None, mutsigs=None,
                                   figsize=None,
                                   per_tree_df=None,
                                   show_pop_model_labels=True,
                                   show_mutsig_labels=True,
                                   label_fontsize=14, tick_fontsize=13,
                                   legend_fontsize=13, cbar_tick_fontsize=13,
                                   elog=False,
                                   title=None, save_path=None):
    """
    Scatter plot of y_col vs x_col per pre-computed time bin, colored by time.
    Rows = mutation signal, columns = population model. Each point is one bin;
    error bars show the pre-computed HPD. Bins with fewer than min_nodes_per_bin
    nodes are excluded. Two colorbars (constcoal blue, skyline red→purple).

    Parameters
    ----------
    time_bin_df : DataFrame
        Output of load_time_bins(): one row per (scenario, model, time bin)
        with model, n_nodes, the time column and {x_col,y_col}_median /
        _hpd_lower / _hpd_upper.
    sampling : str
        Sampling scheme to plot.
    x_col, y_col : str
        Base column names without the "_median" suffix.
    x_range, y_range : tuple, optional
        Axis limits.
    show_errorbars : bool
        Draw HPD error bars (ignored when per_tree_df is given).
    min_nodes_per_bin : int
        Bins with fewer nodes are not plotted.
    log_colorbar : bool
        Use a logarithmic colour scale.
    time_col : str
        Column used for colouring, e.g. "bin_center".
    x_label, y_label : str, optional
        Axis labels; default to x_col / y_col.
    cbar_label : str
        Base colorbar label.
    cbar_label_constcoal, cbar_label_skyline : str, optional
        Override the two colorbar labels.
    pop_models, mutsigs : str or list of str, optional
        Population models (columns) and mutation signals (rows) to show.
    figsize : tuple, optional
        Override the default figure size.
    per_tree_df : DataFrame, optional
        Output of load_time_bins_per_tree(). If given, per-tree bin medians
        are drawn as faint background points and the bin points are drawn
        without error bars.
    show_pop_model_labels, show_mutsig_labels : bool
        Toggle column titles and row labels.
    label_fontsize, tick_fontsize, legend_fontsize, cbar_tick_fontsize : float
        Font sizes.
    elog : bool
        If True, apply elog(r) = log(1 + r) to x_col/y_col and their HPD
        columns (and to per_tree_df's x_col/y_col, if given) before plotting.
        elog is monotonic increasing, so HPD lower <= median <= upper is
        preserved, and elog(0) = 0 so the axhline/axvline reference lines at
        0 remain correct. Only defined for r > -1 — relative errors at or
        below -1 become -inf/NaN and won't plot; check your data range if
        that matters for your error_col. Remember to update x_label/y_label
        yourself (e.g. "elog rel. error") since they aren't auto-renamed.
    title : str, optional
        Suptitle; defaults to the sampling scheme.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    """
    df = time_bin_df[time_bin_df["sampling"] == sampling].copy()
    if per_tree_df is not None:
        per_tree_df = per_tree_df[per_tree_df["sampling"] == sampling].copy()
        if "bin_center" not in per_tree_df.columns:
            per_tree_df["bin_center"] = (per_tree_df["bin_lower"] + per_tree_df["bin_upper"]) / 2

    mutsig_order       = mutsigs    if mutsigs    is not None else ["low", "med", "high"]
    growth_model_order = pop_models if pop_models is not None else ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]
    if isinstance(mutsig_order, str):       mutsig_order       = [mutsig_order]
    if isinstance(growth_model_order, str): growth_model_order = [growth_model_order]

    title_map     = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                     "uniform": "Uniform", "bottleneck": "Bottleneck"}
    mutsig_labels = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}
    model_markers = {"constcoal": "o", "skyline": "^"}
    model_cmaps   = {
        "constcoal": plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                         "constcoal_time", ["#b5d2f2", "#7394c2", "#397398"]),
        "skyline":   plt.matplotlib.colors.LinearSegmentedColormap.from_list(
                         "skyline_time",   ["#f2c4d8", "#d991b4", "#80557e"]),
    }

    x_med = f"{x_col}_median"
    x_lo  = f"{x_col}_hpd_lower"
    x_hi  = f"{x_col}_hpd_upper"
    y_med = f"{y_col}_median"
    y_lo  = f"{y_col}_hpd_lower"
    y_hi  = f"{y_col}_hpd_upper"
    has_err = x_lo in df.columns and y_lo in df.columns

    if elog:
        elog_cols = [c for c in (x_med, x_lo, x_hi, y_med, y_lo, y_hi) if c in df.columns]
        df[elog_cols] = np.log1p(df[elog_cols])
        if per_tree_df is not None:
            pt_cols = [c for c in (x_med, y_med) if c in per_tree_df.columns]
            per_tree_df[pt_cols] = np.log1p(per_tree_df[pt_cols])

    t_min = df[time_col].min()
    t_max = df[time_col].max()
    t_min_pos = max(t_min, 1)
    cnorm = plt.matplotlib.colors.LogNorm(vmin=t_min_pos, vmax=t_max) if log_colorbar \
            else plt.Normalize(vmin=t_min_pos, vmax=t_max)

    nrows, ncols = len(mutsig_order), len(growth_model_order)
    default_figsize = (4.5 * ncols, 3.5 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=figsize or default_figsize,
                             squeeze=False)

    for i, mutsig in enumerate(mutsig_order):
        for j, pop_model in enumerate(growth_model_order):
            ax = axes[i, j]
            df_sub = df[(df["mutation_signal"] == mutsig) & (df["population_model"] == pop_model)]

            for model in ("constcoal", "skyline"):
                df_m = df_sub[df_sub["model"] == model].copy()
                df_m = df_m[df_m["n_nodes"] >= min_nodes_per_bin]
                df_m = df_m.dropna(subset=[x_med, y_med])

                if per_tree_df is not None:
                    pt_sub = per_tree_df[
                        (per_tree_df["mutation_signal"] == mutsig) &
                        (per_tree_df["population_model"] == pop_model) &
                        (per_tree_df["model"] == model)
                    ].dropna(subset=[x_med, y_med])
                    bg_marker  = "o" if model == "constcoal" else "^"
                    bg_colors  = [model_cmaps[model](cnorm(max(t, t_min_pos))) for t in pt_sub[time_col]]
                    ax.scatter(pt_sub[x_med], pt_sub[y_med],
                               color=bg_colors, marker=bg_marker, s=8, alpha=0.15, zorder=1, linewidths=0)

                for _, row in df_m.iterrows():
                    t     = max(row[time_col], t_min_pos)
                    color = model_cmaps[model](cnorm(t))
                    xm, ym = row[x_med], row[y_med]

                    if show_errorbars and has_err and per_tree_df is None:
                        ax.errorbar(xm, ym,
                                    xerr=[[xm - row[x_lo]], [row[x_hi] - xm]],
                                    yerr=[[ym - row[y_lo]], [row[y_hi] - ym]],
                                    fmt=model_markers[model], color=color,
                                    ecolor="lightgray", elinewidth=0.8,
                                    capsize=2, markersize=6, alpha=0.85,
                                    zorder=3)
                    else:
                        ax.scatter(xm, ym, color=color,
                                   marker=model_markers[model], s=45,
                                   alpha=1.0, zorder=3,
                                   edgecolors="black", linewidths=0.5)

            ax.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=1)
            ax.axvline(0, color="black", linestyle="--", linewidth=1.5, zorder=1)
            ax.grid(True, color="lightgray", linewidth=0.4, alpha=0.5, zorder=0)
            if x_range: ax.set_xlim(x_range)
            if y_range: ax.set_ylim(y_range)
            ax.tick_params(labelsize=tick_fontsize)
            ax.set_xlabel(x_label or x_col, fontsize=label_fontsize)
            if i == 0 and show_pop_model_labels:
                ax.set_title(title_map.get(pop_model, pop_model), fontsize=label_fontsize + 1)
            if j == 0:
                ax.set_ylabel(y_label or y_col, fontsize=label_fontsize)
            else:
                ax.set_ylabel("")

    suptitle = title if title is not None else sampling
    plt.suptitle(suptitle, fontsize=17)
    top = 0.97 if suptitle else 1.0
    plt.tight_layout(rect=[0, 0, 0.86, top])

    if show_mutsig_labels:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for i, mutsig in enumerate(mutsig_order):
            ax = axes[i][0]
            ylabel_bb = ax.yaxis.label.get_window_extent(renderer)
            x_fig = inv.transform((ylabel_bb.x0, ylabel_bb.y0 + ylabel_bb.height / 2))[0]
            pos = ax.get_position()
            y_fig = (pos.y0 + pos.y1) / 2
            fig.text(x_fig - 0.01, y_fig, mutsig_labels.get(mutsig, mutsig),
                     fontsize=15, ha="center", va="center", rotation=90,
                     transform=fig.transFigure)

    single = (nrows == 1 and ncols == 1)
    right = 0.76 if single else 0.86
    fig.subplots_adjust(right=right)
    cbar_const = fig.colorbar(plt.cm.ScalarMappable(cmap=model_cmaps["constcoal"], norm=cnorm),
                               cax=fig.add_axes([0.78, 0.15, 0.025, 0.7] if single else [0.88, 0.15, 0.015, 0.7]))
    cbar_sky = fig.colorbar(plt.cm.ScalarMappable(cmap=model_cmaps["skyline"], norm=cnorm),
                             cax=fig.add_axes([0.95, 0.15, 0.025, 0.7] if single else [0.97, 0.15, 0.015, 0.7]))
    if single:
        cbar_const.set_label("Const. coalescent", rotation=270, labelpad=5, fontsize=legend_fontsize)
        cbar_sky.set_label("Skyline", rotation=270, labelpad=5, fontsize=legend_fontsize)
        # shared label above both colorbars
        shared_label = "Time (b.p.)" if cbar_label == "Time before present" else cbar_label
        cbar_sky.ax.figure.text(
            (0.78 + 0.92 + 0.015) / 2 + 0.04, 0.87,
            shared_label, ha="center", va="bottom", fontsize=legend_fontsize,
            transform=fig.transFigure
        )
    else:
        cbar_const.set_label(cbar_label_constcoal or f"{cbar_label} — constcoal", rotation=270, labelpad=18, fontsize=legend_fontsize)
        cbar_sky.set_label(cbar_label_skyline or f"{cbar_label} — skyline", rotation=270, labelpad=18, fontsize=legend_fontsize)
    cbar_const.ax.tick_params(labelsize=cbar_tick_fontsize)
    cbar_sky.ax.tick_params(labelsize=cbar_tick_fontsize)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    from IPython.display import display as _display
    _display(fig)
    plt.close(fig)


def plot_combined_population_and_error(df_population, df_time_bins, sampling,
                                       pop_y_range=(0, 5000),
                                       error_y_range=(-1, 3),
                                       error_col='bl_rel_error_median',
                                       time_col='bin_lower',
                                       mode='both',
                                       pop_models=None, mutsigs=None,
                                       min_nodes_per_bin=10,
                                       per_tree_df=None,
                                       raw_alpha=0.15,
                                       jitter_width=0.05,
                                       marker_edgecolor="none",
                                       errorbar_alpha=0.4,
                                       x_label="Time before present",
                                       pop_y_label="Population size",
                                       error_y_label=None,
                                       title=None,
                                       figsize=None,
                                       save_path=None,
                                       ylabel_x_positions=None,
                                       error_ylabel_y_offset=-0.08,
                                       sampling_end_time=None,
                                       trajectory_df=None,
                                       show_changepoints=True,
                                       changepoint_dot_size=25,
                                       changepoint_grey_alpha=0.3,
                                       changepoint_y_frac=0.7,
                                       label_fontsize=12, tick_fontsize=12,
                                       legend_fontsize=13,
                                       hspace=0.3, wspace=0.4,
                                       sampling_fontsize=None,
                                       xticks=None,
                                       show_legend=True):
    """
    Combined population size trajectory (top) + pre-binned error (bottom) per mutsig × pop model.

    With several population models, each mutation signal gets a pair of rows
    (population on top, error below) and each population model a column.
    With a single population model, the mutation signals become columns.
    For independenthomochronous sampling, the change point strip and the
    sampling arrow are always switched off.

    Parameters
    ----------
    df_population : DataFrame
        One row per replicate, e.g. the output of load_pop_summaries(); used
        for the true trajectory (and for the estimates if trajectory_df is None).
    df_time_bins : DataFrame
        Output of load_time_bins(), already has bin_lower/bin_upper/bin_center,
        model column, and pre-computed error medians/HPDs.
    sampling : str
        Sampling scheme to plot.
    pop_y_range, error_y_range : tuple
        y-axis limits of the population and error panels.
    error_col : str
        Median column to plot (e.g. 'bl_rel_error_median'). HPD columns for
        the error bars are found by replacing '_median' with '_hpd_lower/upper'.
    time_col : str
        Time column of the bins used as x position; its maximum is also the
        time horizon of all panels.
    mode : {"both", "skyline", "constcoal"}
        Which population estimates to draw.
    pop_models, mutsigs : str or list of str, optional
        Population models and mutation signals to show (default: all present).
    min_nodes_per_bin : int
        Bins with fewer nodes are not plotted.
    per_tree_df : DataFrame, optional
        Output of load_time_bins_per_tree(); drawn as faint background points.
    raw_alpha, jitter_width : float
        Transparency and horizontal jitter (fraction of the bin width) of
        the per-tree points.
    marker_edgecolor : str
        Edge colour of the bin markers.
    errorbar_alpha : float
        Transparency of the HPD error bars; 0 hides them.
    x_label, pop_y_label, error_y_label : str
        Axis labels; error_y_label defaults to error_col.
    title : str, optional
        Suptitle; defaults to "Population size & {error_col} ({sampling})".
    figsize : tuple, optional
        Override the default figure size.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    ylabel_x_positions : list of float, optional
        Single-population-model layout only: x positions (figure coordinates)
        of the y-labels per column, instead of placing them automatically.
    error_ylabel_y_offset : float
        Single-population-model layout only: vertical offset of the error
        y-label in figure coordinates.
    sampling_end_time : float, optional
        If given, draw an arrow labelled "sampling" from 0 to this time.
    trajectory_df : DataFrame, optional
        Precomputed trajectories from load_pop_trajectories() (see
        plot_population_summary).
    show_changepoints : bool
        Draw the skyline change point strip.
    changepoint_dot_size, changepoint_grey_alpha, changepoint_y_frac : float
        Styling of the change point strip (see _add_skyline_changepoint_strip).
    label_fontsize, tick_fontsize, legend_fontsize : float
        Font sizes.
    hspace, wspace : float
        Spacing between subplots.
    sampling_fontsize : float, optional
        Font size of the "sampling" label; defaults to label_fontsize - 2.
    xticks : list of float, optional
        Set these x ticks on all panels.
    show_legend : bool
        Show the figure legend.
    """
    df_population = df_population[df_population["sampling"] == sampling].copy()
    df_bins       = df_time_bins[df_time_bins["sampling"] == sampling].copy()

    # independenthomochronous: all tips sampled at t=0, so the sampling-window
    # arrow and per-tree skyline changepoints (both about *when* samples/events
    # occur across time) aren't meaningful — suppress regardless of the params
    # passed in, without removing the params themselves for other sampling types.
    if sampling == "independenthomochronous":
        show_changepoints  = False
        sampling_end_time  = None

    error_lo_col = error_col.replace("_median", "_hpd_lower")
    error_hi_col = error_col.replace("_median", "_hpd_upper")

    time_horizon = df_bins[time_col].max()

    if error_y_label is None:
        error_y_label = error_col

    default_mutsig_order       = ["low", "med", "high"]
    default_growth_model_order = ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]
    mutsig_order       = mutsigs    if mutsigs    is not None else [m for m in default_mutsig_order       if m in df_bins["mutation_signal"].unique()]
    growth_model_order = pop_models if pop_models is not None else [m for m in default_growth_model_order if m in df_bins["population_model"].unique()]
    if isinstance(mutsig_order, str):       mutsig_order       = [mutsig_order]
    if isinstance(growth_model_order, str): growth_model_order = [growth_model_order]

    mutsig_labels = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}
    title_map     = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                     "uniform": "Uniform", "bottleneck": "Bottleneck"}
    model_colors  = {"constcoal": "#397398", "skyline": "#80557e"}
    model_markers = {"constcoal": "o", "skyline": "^"}
    model_offsets = {"constcoal": 0, "skyline": 0}
    _bin_width = df_bins[time_col].diff().abs().median()

    ncols = len(growth_model_order)
    single_pop_model = ncols == 1

    if single_pop_model:
        ncols_fig = len(mutsig_order)
        fig, axes = plt.subplots(2, ncols_fig,
                                 figsize=figsize or (5 * ncols_fig, 5),
                                 gridspec_kw={"hspace": hspace, "wspace": wspace})
        if ncols_fig == 1:
            axes = axes.reshape(2, 1)
    else:
        row_heights = ([1, 1, 0.3] * len(mutsig_order))[:-1]
        fig = plt.figure(figsize=figsize or (8 * ncols, 2.5 * len(row_heights)))
        gs  = gridspec.GridSpec(len(row_heights), ncols,
                                height_ratios=row_heights, hspace=hspace, wspace=wspace)

    ax_pop_first_col = {}
    ax_err_first_col = {}

    for i, mutsig in enumerate(mutsig_order):
        for j, growth_model in enumerate(growth_model_order):
            if single_pop_model:
                ax_pop = axes[0, i]
                ax_err = axes[1, i]
            else:
                ax_pop = fig.add_subplot(gs[i * 3,     j])
                ax_err = fig.add_subplot(gs[i * 3 + 1, j])

            if j == 0:
                ax_pop_first_col[i] = ax_pop
                ax_err_first_col[i] = ax_err

            # ── Population trajectory ──────────────────────────────────────
            subset_pop = df_population[
                (df_population["mutation_signal"]  == mutsig) &
                (df_population["population_model"] == growth_model)]
            if not subset_pop.empty:
                true_traj = get_true_traj(growth_model, subset_pop.iloc[0])
                color_exp   = "#a6444f"
                color_sky   = "#80557e"
                color_const = "#397398"
                if trajectory_df is not None:
                    traj_sub = trajectory_df[
                        (trajectory_df["sampling"]          == sampling) &
                        (trajectory_df["mutation_signal"]   == mutsig) &
                        (trajectory_df["population_model"]  == growth_model)
                    ]
                    if not traj_sub.empty:
                        t_vals = traj_sub["time"].values
                        mask   = t_vals <= time_horizon
                        t_plot = t_vals[mask]
                        t_smooth = np.linspace(0, time_horizon, 1000)
                        N_true = np.array([true_traj(t) for t in t_smooth])
                        ax_pop.plot(t_smooth, N_true, color=color_exp, linewidth=2,
                                    label="True Population Size", zorder=6)
                        if mode in ('both', 'skyline'):
                            ax_pop.plot(t_plot, traj_sub["skyline_median"].values[mask],
                                        color=color_sky, linestyle='--', label="Skyline Estimate",
                                        linewidth=2, zorder=5)
                            ax_pop.fill_between(t_plot,
                                                traj_sub["skyline_hpd_lower"].values[mask],
                                                traj_sub["skyline_hpd_upper"].values[mask],
                                                color=color_sky, alpha=0.3, zorder=4)
                        if mode in ('both', 'constcoal'):
                            ax_pop.plot(t_plot, traj_sub["constcoal_median"].values[mask],
                                        color=color_const, linestyle=':', label="Constant Estimate",
                                        linewidth=2, zorder=5)
                            ax_pop.fill_between(t_plot,
                                                traj_sub["constcoal_hpd_lower"].values[mask],
                                                traj_sub["constcoal_hpd_upper"].values[mask],
                                                color=color_const, alpha=0.3, zorder=4)
                        if pop_y_range:
                            ax_pop.set_ylim(pop_y_range)
                        ax_pop.invert_xaxis()
                else:
                    plot_population_summary_ax(
                        ax=ax_pop,
                        skyline_all_times=subset_pop["skyline_times"].tolist(),
                        skyline_all_medians=subset_pop["skyline_medians"].tolist(),
                        constant_all_estimates=subset_pop["coalescent_median"].tolist(),
                        true_traj=true_traj,
                        time_horizon=time_horizon, y_range=pop_y_range, mode=mode)
                ax_pop.set_xlim(time_horizon, 0)
            ax_pop.tick_params(labelsize=tick_fontsize)
            ax_pop.set_xlabel(x_label, fontsize=label_fontsize)
            show_ylabel = j == 0
            ax_pop.set_ylabel(pop_y_label if show_ylabel else "", fontsize=label_fontsize)

            if show_changepoints and not subset_pop.empty and "skyline_times" in subset_pop.columns:
                _add_skyline_changepoint_strip(ax_pop, subset_pop,
                                               dot_size=changepoint_dot_size,
                                               grey_alpha=changepoint_grey_alpha,
                                               y_frac=changepoint_y_frac)

            if sampling_end_time is not None:
                ylims = ax_pop.get_ylim()
                y_arrow = ylims[1] * 0.85
                ax_pop.annotate(
                    "", xy=(sampling_end_time, y_arrow), xytext=(0, y_arrow),
                    arrowprops=dict(arrowstyle="-|>", color="grey", lw=1.5),
                    annotation_clip=False,
                )
                ax_pop.text(
                    sampling_end_time / 2, y_arrow + (ylims[1] - ylims[0]) * 0.02, "sampling",
                    color="grey", fontsize=sampling_fontsize if sampling_fontsize is not None else label_fontsize - 2, ha="center", va="bottom",
                )

            # ── Pre-binned error ───────────────────────────────────────────
            subset_bins = df_bins[
                (df_bins["mutation_signal"]  == mutsig) &
                (df_bins["population_model"] == growth_model) &
                (df_bins[time_col] <= time_horizon) &
                (df_bins["n_nodes"]   >= min_nodes_per_bin)
            ].copy()

            has_err = error_lo_col in subset_bins.columns and error_hi_col in subset_bins.columns

            # ── Raw individual node dots (background) ──────────────────────
            if per_tree_df is not None:
                pt_sub_all = per_tree_df[
                    (per_tree_df["sampling"]          == sampling) &
                    (per_tree_df["mutation_signal"]   == mutsig) &
                    (per_tree_df["population_model"]  == growth_model)
                ].copy()
                rng = np.random.default_rng(42)
                for model, color in model_colors.items():
                    pt_sub = pt_sub_all[pt_sub_all["model"] == model].dropna(subset=[error_col, time_col])
                    x_jitter = rng.uniform(-_bin_width * jitter_width, _bin_width * jitter_width, size=len(pt_sub))
                    ax_err.scatter(pt_sub[time_col].values + x_jitter, pt_sub[error_col].values,
                                   color=color, marker=model_markers[model],
                                   s=10, alpha=raw_alpha, zorder=1, linewidths=0)

            for model, color in model_colors.items():
                df_m = subset_bins[subset_bins["model"] == model].dropna(subset=[error_col])
                if df_m.empty:
                    continue
                xvals = df_m[time_col] + model_offsets[model]
                yvals = df_m[error_col]
                if has_err and errorbar_alpha > 0:
                    eb = ax_err.errorbar(
                        xvals, yvals,
                        yerr=[yvals - df_m[error_lo_col], df_m[error_hi_col] - yvals],
                        fmt=model_markers[model], color=color,
                        ecolor=color, elinewidth=1.4, capsize=4,
                        markersize=7, markeredgecolor=marker_edgecolor, markeredgewidth=0.8,
                        alpha=1.0, zorder=3)
                    for line in eb[1]: line.set_alpha(errorbar_alpha)
                    for line in eb[2]: line.set_alpha(errorbar_alpha)
                else:
                    ax_err.scatter(xvals, yvals, color=color,
                                   marker=model_markers[model], s=40,
                                   edgecolors=marker_edgecolor, linewidths=0.8,
                                   alpha=1.0, zorder=3)

            ax_err.axhline(0, color="black", linestyle="--", linewidth=1.5, zorder=1)
            ax_err.grid(True, color="lightgray", linewidth=0.4, alpha=0.5, zorder=0)
            ax_err.invert_xaxis()
            padding = (time_horizon - 0) * 0.05
            ax_err.set_xlim(time_horizon + padding, -padding)
            if error_y_range:
                ax_err.set_ylim(error_y_range)
            ax_err.tick_params(labelsize=tick_fontsize)
            ax_err.set_xlabel(x_label, fontsize=label_fontsize)
            ax_err.set_ylabel(error_y_label if show_ylabel else "", fontsize=label_fontsize)

            # ── Column titles (pop model) ──────────────────────────────────
            if i == 0:
                ax_pop.set_title(title_map.get(growth_model, growth_model), fontsize=label_fontsize + 2)

    handles = [Line2D([0], [0], color=c, lw=2, marker=model_markers[m],
                      markersize=7, label={"constcoal": "Const. coal.", "skyline": "Skyline"}[m])
               for m, c in model_colors.items()]

    if single_pop_model:
        for i, mutsig in enumerate(mutsig_order):
            ax_pop_first_col[i].set_title(mutsig_labels.get(mutsig, mutsig), fontsize=label_fontsize)
        if show_legend:
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.99, 1.10),
                       ncol=2, frameon=False, fontsize=legend_fontsize)
        # place ylabels as fig.text at a shared x, each centered on its own axis
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for i in ax_pop_first_col:
            if ylabel_x_positions is not None and i < len(ylabel_x_positions):
                x_label_x = ylabel_x_positions[i]
            else:
                col_axes = [ax_pop_first_col[i], ax_err_first_col[i]]
                x_left = min(
                    inv.transform([[ax.get_tightbbox(renderer).x0, 0]])[0][0]
                    for ax in col_axes
                )
                x_label_x = x_left - 0.01
            for ax, label, y_offset in [(ax_pop_first_col[i], pop_y_label, 0.0), (ax_err_first_col[i], error_y_label, error_ylabel_y_offset)]:
                ax.set_ylabel("")
                pos = ax.get_position()
                y_fig = (pos.y0 + pos.y1) / 2 + y_offset
                fig.text(x_label_x, y_fig, label, fontsize=label_fontsize,
                         ha="center", va="center", rotation=90, transform=fig.transFigure)
    else:
        # Mutsig row labels centred between the two axes on the left
        fig.canvas.draw()
        renderer  = fig.canvas.get_renderer()
        inv       = fig.transFigure.inverted()
        # place ylabels as fig.text at a shared x, centered on each axis
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        # find the leftmost tick label edge across all left-column axes
        all_left = list(ax_pop_first_col.values()) + list(ax_err_first_col.values())
        x_min = min(
            inv.transform(ax.yaxis.get_ticklabels()[-1].get_window_extent(renderer))[0][0]
            if ax.yaxis.get_ticklabels() else 1.0
            for ax in all_left
        )
        x_label_x = x_min - 0.02
        for i in ax_pop_first_col:
            ax_p = ax_pop_first_col[i]
            ax_e = ax_err_first_col[i]
            ax_p.set_ylabel("")
            ax_e.set_ylabel("")
            pos_p = ax_p.get_position()
            pos_e = ax_e.get_position()
            y_pop = (pos_p.y0 + pos_p.y1) / 2
            y_err = (pos_e.y0 + pos_e.y1) / 2
            if i == 0:
                fig.text(x_label_x, y_pop, pop_y_label, fontsize=label_fontsize,
                         ha="center", va="center", rotation=90, transform=fig.transFigure)
                fig.text(x_label_x, y_err, error_y_label, fontsize=label_fontsize,
                         ha="center", va="center", rotation=90, transform=fig.transFigure)
        for i, mutsig in enumerate(mutsig_order):
            ax_p = ax_pop_first_col[i]
            ax_e = ax_err_first_col[i]
            pos_p = ax_p.get_position()
            pos_e = ax_e.get_position()
            y_fig = (pos_p.y0 + pos_e.y1) / 2
            ylabel_bb = ax_p.yaxis.label.get_window_extent(renderer)
            x_fig = inv.transform((ylabel_bb.x0, 0))[0] - 0.01
            fig.text(x_fig, y_fig, mutsig_labels.get(mutsig, mutsig),
                     fontsize=label_fontsize, ha="center", va="center", rotation=90,
                     transform=fig.transFigure)
        if show_legend:
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.97, 0.97),
                       ncol=2, frameon=False, fontsize=legend_fontsize)

    if xticks is not None:
        for ax in fig.get_axes():
            ax.set_xticks(xticks)

    suptitle = title if title is not None else f"Population size & {error_col} ({sampling})"
    plt.suptitle(suptitle, fontsize=label_fontsize + 2)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.93, bottom=0.07)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    plt.show()


# =============================================================================
# Loaders for compute_errors.py outputs
# =============================================================================

def load_pop_trajectories(eval_dir):
    """
    Read all *_pop_trajectory.tsv files from eval_dir and concatenate into
    one DataFrame with added sampling, population_model, mutation_signal columns.

    The files are written by scripts/compute_errors.py and contain, per
    scenario, the median and HPD of the estimated population size across
    all posterior samples and replicates on a shared time grid.

    Parameters
    ----------
    eval_dir : str or Path
        Evaluation directory, searched recursively (e.g. results/run1/evaluation).

    Returns
    -------
    DataFrame
        Columns time, skyline_/constcoal_ median, hpd_lower, hpd_upper, ...
        plus sampling, population_model and mutation_signal (parsed from the
        file name).

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    eval_dir = Path(eval_dir)
    dfs = []
    for tsv in sorted(eval_dir.rglob("*_pop_trajectory.tsv")):
        df = pd.read_csv(tsv, sep="\t")
        # parse scenario from filename: {sampling}_{pop_model}_{mutsig}mutsig_pop_trajectory.tsv
        stem = tsv.stem.replace("_pop_trajectory", "")
        mutsig = stem.split("_")[-1].replace("mutsig", "")
        rest = stem[: stem.rfind(f"_{mutsig}mutsig")]
        sampling = None
        for s in ("independenthomochronous", "linearconstant"):
            if rest.startswith(s):
                sampling = s
                pop_model = rest[len(s) + 1:]
                break
        df["sampling"] = sampling
        df["population_model"] = pop_model
        df["mutation_signal"] = mutsig
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No *_pop_trajectory.tsv files found under {eval_dir}")
    return pd.concat(dfs, ignore_index=True)


def load_pop_summaries(eval_dir):
    """
    Read all *_pop_summary.pkl files from eval_dir, concatenate, and merge
    with the scenario TSVs (high/med/low.tsv) to add population parameters
    (present_pop_size, growth_rate, bottleneck_size, bottleneck_start,
    bottleneck_end) per replicate.

    The pickles are written by scripts/compute_errors.py, the scenario TSVs
    by scripts/process_results.py. Only replicates present in both are kept
    (inner join).

    Parameters
    ----------
    eval_dir : str or Path
        Evaluation directory, searched recursively.

    Returns
    -------
    DataFrame
        One row per replicate with the per-replicate population size
        estimates (e.g. skyline_times, skyline_medians, coalescent_median)
        and the true population parameters.

    Raises
    ------
    FileNotFoundError
        If no *_pop_summary.pkl files are found.
    """
    eval_dir = Path(eval_dir)

    # Load all pop summaries
    dfs = []
    for pkl in sorted(eval_dir.rglob("*_pop_summary.pkl")):
        dfs.append(pd.read_pickle(pkl))
    if not dfs:
        raise FileNotFoundError(f"No *_pop_summary.pkl files found under {eval_dir}")
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop(columns=["scenario"], errors="ignore")

    # Load all scenario TSVs and concatenate
    pop_params_cols = ["sampling", "population_model", "mutation_signal", "tree_index",
                       "present_pop_size", "growth_rate",
                       "bottleneck_size", "bottleneck_start", "bottleneck_end"]
    tsv_dfs = []
    for tsv in sorted(eval_dir.rglob("*.tsv")):
        if "node_errors" in tsv.name:
            continue
        raw = pd.read_csv(tsv, sep="\t")
        if not all(c in raw.columns for c in pop_params_cols):
            continue
        tsv_dfs.append(raw[pop_params_cols])
    if tsv_dfs:
        params_df = pd.concat(tsv_dfs, ignore_index=True).drop_duplicates()
        df = df.merge(params_df, on=["sampling", "population_model", "mutation_signal", "tree_index"], how="inner")

    return df


def load_node_errors(eval_dir):
    """
    Read all *_node_errors.tsv files from eval_dir and concatenate into one
    DataFrame matching the structure of tree_metrics_combined, with separate
    columns for population_model, mutation_signal, and sampling parsed from
    the scenario string.

    The files are written by scripts/compute_errors.py. The root of each
    tree has node_id "internal_0", so the per-replicate root height table is
    df[df["node_id"] == "internal_0"].

    Parameters
    ----------
    eval_dir : str or Path
        Evaluation directory, searched recursively.

    Returns
    -------
    DataFrame
        One row per (replicate, node) with the per-model error medians,
        HPDs and CI-coverage flags, plus sampling, population_model and
        mutation_signal.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    eval_dir = Path(eval_dir)
    dfs = []
    for tsv in sorted(eval_dir.rglob("*_node_errors.tsv")):
        dfs.append(pd.read_csv(tsv, sep="\t"))
    if not dfs:
        raise FileNotFoundError(f"No *_node_errors.tsv files found under {eval_dir}")
    df = pd.concat(dfs, ignore_index=True)

    def parse_scenario(s):
        mutsig = s.split("_")[-1].replace("mutsig", "")
        rest   = s[: s.rfind(f"_{mutsig}mutsig")]
        for sampling in ("independenthomochronous", "linearconstant"):
            if rest.startswith(sampling):
                return sampling, rest[len(sampling) + 1:], mutsig
        return None, rest, mutsig

    parsed = df["scenario"].apply(lambda s: pd.Series(
        parse_scenario(s), index=["sampling", "population_model", "mutation_signal"]
    ))
    df = pd.concat([df.drop(columns=["scenario"]), parsed], axis=1)
    return df


def load_time_bins(eval_dir, bin_width, cutoff):
    """
    Read all time bin TSV files matching a given bin_width and cutoff from
    eval_dir and concatenate into one DataFrame. sampling, population_model,
    and mutation_signal are parsed from the scenario column.

    The files are written by scripts/compute_time_bins.py, one per scenario
    and (bin_width, cutoff) combination (see TIME_BIN_CONFIGS in the Snakefile).

    Parameters
    ----------
    eval_dir  : str or Path
        Evaluation directory, searched recursively.
    bin_width : int   e.g. 10
    cutoff    : int   e.g. 400

    Returns
    -------
    DataFrame
        One row per (scenario, model, time bin) with bin_lower, bin_upper,
        n_nodes and the error medians/HPDs, plus sampling, population_model
        and mutation_signal.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    eval_dir = Path(eval_dir)
    pattern  = f"*_time_bins_w{int(bin_width)}_c{int(cutoff)}.tsv"
    dfs = []
    for tsv in sorted(eval_dir.rglob(pattern)):
        dfs.append(pd.read_csv(tsv, sep="\t"))
    if not dfs:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found under {eval_dir}"
        )
    df = pd.concat(dfs, ignore_index=True)

    def parse_scenario(s):
        mutsig = s.split("_")[-1].replace("mutsig", "")
        rest   = s[: s.rfind(f"_{mutsig}mutsig")]
        for sampling in ("independenthomochronous", "linearconstant"):
            if rest.startswith(sampling):
                return sampling, rest[len(sampling) + 1:], mutsig
        return None, rest, mutsig

    parsed = df["scenario"].apply(lambda s: pd.Series(
        parse_scenario(s), index=["sampling", "population_model", "mutation_signal"]
    ))
    df = pd.concat([df.drop(columns=["scenario"]), parsed], axis=1)
    return df


def load_time_bins_per_tree(eval_dir, bin_width, cutoff):
    """
    Like load_time_bins but reads the per-tree files
    (*_time_bins_w{bin_width}_c{cutoff}_per_tree.tsv).

    Parameters
    ----------
    eval_dir : str or Path
        Evaluation directory, searched recursively.
    bin_width, cutoff : int
        Time binning configuration, e.g. 10 and 400.

    Returns
    -------
    DataFrame
        One row per (scenario, model, replicate tree, time bin), plus
        sampling, population_model and mutation_signal.

    Raises
    ------
    FileNotFoundError
        If no matching files are found.
    """
    eval_dir = Path(eval_dir)
    pattern  = f"*_time_bins_w{int(bin_width)}_c{int(cutoff)}_per_tree.tsv"
    dfs = []
    for tsv in sorted(eval_dir.rglob(pattern)):
        dfs.append(pd.read_csv(tsv, sep="\t"))
    if not dfs:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found under {eval_dir}"
        )
    df = pd.concat(dfs, ignore_index=True)

    def parse_scenario(s):
        mutsig = s.split("_")[-1].replace("mutsig", "")
        rest   = s[: s.rfind(f"_{mutsig}mutsig")]
        for sampling in ("independenthomochronous", "linearconstant"):
            if rest.startswith(sampling):
                return sampling, rest[len(sampling) + 1:], mutsig
        return None, rest, mutsig

    parsed = df["scenario"].apply(lambda s: pd.Series(
        parse_scenario(s), index=["sampling", "population_model", "mutation_signal"]
    ))
    df = pd.concat([df.drop(columns=["scenario"]), parsed], axis=1)
    return df


def compute_pairwise_tip_distances(tree_index, repo_base=None, run="run1",
                                   population_model=None, mutation_signal=None,
                                   sampling=None):
    """
    For one or all successful trees, compute for each pair of tips the distance
    of each tip to their MRCA, from constcoal and skyline summary trees.

    Successful trees are read from the scenario TSVs (low/med/high.tsv)
    under results/{run}/evaluation; heights are taken from the
    height_median and height_95%_HPD annotations of the
    *.combined_summary.tree files.

    Distance of tip X to MRCA = h_mrca - h_tip  (both heights in time before present).
    HPD bounds: conservative — lower = h_mrca_lo - h_tip_hi, upper = h_mrca_hi - h_tip_lo.

    Parameters
    ----------
    tree_index : int, or 'all' to process every successful tree.
    repo_base : str or Path, optional
        Project root; defaults to the parent of this file's directory.
    run : str
        Results run folder, e.g. "run1".
    population_model, mutation_signal, sampling : filter to a single scenario
        (recommended when tree_index='all' to keep runtime manageable).
        mutation_signal uses the short names "low", "med", "high".

    Returns
    -------
    DataFrame with one row per tip pair per scenario, columns:
        sampling, population_model, mutation_signal, tree_index, tip_a, tip_b,
        tip_a_time, tip_b_time (sampling times, time before present),
        dist_a_constcoal_median, dist_a_constcoal_hpd_lower, dist_a_constcoal_hpd_upper,
        dist_b_constcoal_median, dist_b_constcoal_hpd_lower, dist_b_constcoal_hpd_upper,
        dist_a_skyline_median,   dist_a_skyline_hpd_lower,   dist_a_skyline_hpd_upper,
        dist_b_skyline_median,   dist_b_skyline_hpd_lower,   dist_b_skyline_hpd_upper,
        dist_a_abs_diff  (constcoal_median - skyline_median),
        dist_a_rel_diff  ((constcoal_median - skyline_median) / skyline_median),
        dist_b_abs_diff,
        dist_b_rel_diff
    """
    if repo_base is None:
        repo_base = Path(__file__).resolve().parent.parent
    repo_base = Path(repo_base)

    eval_dir = repo_base / "results" / run / "evaluation"
    tsv_files = sorted({p for p in eval_dir.rglob("*.tsv")
                        if p.stem in ("low", "med", "high")})

    path_df = pd.concat(
        [pd.read_csv(f, sep="\t") for f in tsv_files],
        ignore_index=True
    )

    if tree_index == "all":
        subset = path_df.copy()
    else:
        subset = path_df[path_df["tree_index"] == tree_index]

    if population_model is not None:
        subset = subset[subset["population_model"] == population_model]
    if mutation_signal is not None:
        subset = subset[subset["mutation_signal"] == mutation_signal]
    if sampling is not None:
        subset = subset[subset["sampling"] == sampling]

    def _parse_height(clade):
        h_med, h_lo, h_hi = 0.0, 0.0, 0.0
        if clade.comment:
            m = re.search(r'height_median=([\d\.eE+-]+)', clade.comment)
            if m: h_med = float(m.group(1))
            m = re.search(r'height_95%_HPD=\{([\d\.eE+-]+),([\d\.eE+-]+)\}', clade.comment)
            if m: h_lo, h_hi = float(m.group(1)), float(m.group(2))
        return h_med, h_lo, h_hi

    # key: (sampling, population_model, mutation_signal, tip_a, tip_b)
    # value: dict accumulating per-model distances
    pair_data = {}

    for _, row in subset.iterrows():
        scenario_key = (row["sampling"], row["population_model"], row["mutation_signal"], row["tree_index"])

        for model, col in (("constcoal", "trees_path_constcoal"),
                           ("skyline",   "trees_path_skyline")):
            raw = row[col]
            if pd.isna(raw):
                continue
            summary_path = repo_base / str(raw).replace(".combined.trees", ".combined_summary.tree")
            if not summary_path.exists():
                continue

            tree = Phylo.read(str(summary_path), "nexus")
            tip_info  = {c.name: _parse_height(c) for c in tree.get_terminals()}
            node_info = {id(c): _parse_height(c)   for c in tree.get_nonterminals()}

            tips = list(tip_info.keys())
            for i_t, tip_a in enumerate(tips):
                for tip_b in tips[i_t + 1:]:
                    mrca = tree.common_ancestor(tip_a, tip_b)
                    mrca_med, mrca_lo, mrca_hi = node_info.get(id(mrca), (0.0, 0.0, 0.0))

                    ha_med, ha_lo, ha_hi = tip_info[tip_a]
                    hb_med, hb_lo, hb_hi = tip_info[tip_b]

                    # distance tip → MRCA = h_mrca - h_tip
                    da_med = mrca_med - ha_med
                    da_lo  = mrca_lo  - ha_hi   # conservative lower
                    da_hi  = mrca_hi  - ha_lo   # conservative upper
                    db_med = mrca_med - hb_med
                    db_lo  = mrca_lo  - hb_hi
                    db_hi  = mrca_hi  - hb_lo

                    key = scenario_key + (tip_a, tip_b)
                    if key not in pair_data:
                        pair_data[key] = {}
                        # sampling times are tree-independent; store once
                        pair_data[key]["tip_a_time"] = ha_med
                        pair_data[key]["tip_b_time"] = hb_med
                    pair_data[key][f"dist_a_{model}_median"]    = da_med
                    pair_data[key][f"dist_a_{model}_hpd_lower"] = da_lo
                    pair_data[key][f"dist_a_{model}_hpd_upper"] = da_hi
                    pair_data[key][f"dist_b_{model}_median"]    = db_med
                    pair_data[key][f"dist_b_{model}_hpd_lower"] = db_lo
                    pair_data[key][f"dist_b_{model}_hpd_upper"] = db_hi

    records = []
    for (sampling, pop_model, mut_sig, t_idx, tip_a, tip_b), vals in pair_data.items():
        rec = {
            "sampling":         sampling,
            "population_model": pop_model,
            "mutation_signal":  mut_sig,
            "tree_index":       t_idx,
            "tip_a":            tip_a,
            "tip_b":            tip_b,
        }
        rec.update(vals)
        # diffs (constcoal - skyline)
        for tip_lbl in ("a", "b"):
            cc = rec.get(f"dist_{tip_lbl}_constcoal_median", float("nan"))
            sk = rec.get(f"dist_{tip_lbl}_skyline_median",   float("nan"))
            rec[f"dist_{tip_lbl}_abs_diff"] = cc - sk
            rec[f"dist_{tip_lbl}_rel_diff"] = (cc - sk) / sk if sk != 0 else float("nan")
        records.append(rec)

    return pd.DataFrame(records)


def plot_pairwise_tip_distances(dist_df, sampling,
                                x_range=None, y_range=None,
                                alpha=0.5, errorbar_alpha=0.3, errorbar_width=0.2,
                                n_subsample=None,
                                pop_models=None, mutsigs=None,
                                figsize=None, title=None,
                                x_label="Skyline — pairwise tip distance",
                                y_label="Const. Coalescent — pairwise tip distance",
                                show_pop_model_labels=True,
                                show_mutsig_labels=True,
                                running_avg=False,
                                window_size=500, window_overlap=250,
                                running_avg_color="#c0392b",
                                running_avg_linewidth=0.8,
                                diff_plot=False,
                                diff_mode="absolute",
                                diff_axis_range=None,
                                label_fontsize=13, tick_fontsize=13, legend_fontsize=11,
                                save_path=None):
    """
    Scatter plot of skyline (x) vs constcoal (y) pairwise tip-to-MRCA distances,
    with HPD error bars. Rows = mutation signal, columns = pop model.
    Optionally adds running median line and difference panel below each column.

    Both tips of each pair contribute one point (distance of tip a and of
    tip b to their MRCA). Running medians and differences are computed in
    sliding windows over the points sorted by the skyline distance, using
    all points even if n_subsample is set.

    Parameters
    ----------
    dist_df : DataFrame
        Output of compute_pairwise_tip_distances().
    sampling : str
        Sampling scheme to plot.
    x_range, y_range : tuple, optional
        Axis limits.
    alpha, errorbar_alpha, errorbar_width : float
        Styling of the points and HPD error bars.
    n_subsample : int, optional
        Randomly draw at most this many points per subplot for the scatter.
    pop_models, mutsigs : str or list of str, optional
        Population models (columns) and mutation signals (rows) to show.
    figsize : tuple, optional
        Size of the scatter part of the figure (diff panels are added on top).
    title : str, optional
        Suptitle; defaults to "Pairwise tip distances — {sampling}".
    x_label, y_label : str
        Unused; axes are labelled "Skyline" / "Const. Coalescent".
    show_pop_model_labels, show_mutsig_labels : bool
        Toggle column titles and row labels.
    running_avg : bool
        Draw the sliding-window median of constcoal vs skyline as a step line.
    window_size, window_overlap : int
        Number of points per sliding window and overlap between windows.
    running_avg_color, running_avg_linewidth
        Styling of the running median line.
    diff_plot : bool
        Add a panel below each scatter with constcoal − skyline per point
        (grey) and per window (coloured).
    diff_mode : {"absolute", "relative"}
        constcoal − skyline, or (constcoal − skyline) / skyline.
    diff_axis_range : tuple, optional
        y-axis limits of the difference panels.
    label_fontsize, tick_fontsize, legend_fontsize : float
        Font sizes.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    """
    from matplotlib.gridspec import GridSpec

    mutsig_order       = mutsigs    if mutsigs    is not None else ["low", "med", "high"]
    growth_model_order = pop_models if pop_models is not None else ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]
    if isinstance(mutsig_order, str):       mutsig_order       = [mutsig_order]
    if isinstance(growth_model_order, str): growth_model_order = [growth_model_order]

    mutsig_labels = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}
    title_map     = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                     "uniform": "Uniform", "bottleneck": "Bottleneck"}
    color = "#555555"

    base = dist_df[dist_df["sampling"] == sampling].copy()
    tip_a_df = base.rename(columns={
        "dist_a_constcoal_median":    "constcoal_dist_median",
        "dist_a_constcoal_hpd_lower": "constcoal_dist_hpd_lower",
        "dist_a_constcoal_hpd_upper": "constcoal_dist_hpd_upper",
        "dist_a_skyline_median":      "skyline_dist_median",
        "dist_a_skyline_hpd_lower":   "skyline_dist_hpd_lower",
        "dist_a_skyline_hpd_upper":   "skyline_dist_hpd_upper",
    })
    tip_b_df = base.rename(columns={
        "dist_b_constcoal_median":    "constcoal_dist_median",
        "dist_b_constcoal_hpd_lower": "constcoal_dist_hpd_lower",
        "dist_b_constcoal_hpd_upper": "constcoal_dist_hpd_upper",
        "dist_b_skyline_median":      "skyline_dist_median",
        "dist_b_skyline_hpd_lower":   "skyline_dist_hpd_lower",
        "dist_b_skyline_hpd_upper":   "skyline_dist_hpd_upper",
    })
    keep_cols = ["sampling", "population_model", "mutation_signal", "tree_index",
                 "constcoal_dist_median", "constcoal_dist_hpd_lower", "constcoal_dist_hpd_upper",
                 "skyline_dist_median",   "skyline_dist_hpd_lower",   "skyline_dist_hpd_upper"]
    wide = pd.concat([tip_a_df[keep_cols], tip_b_df[keep_cols]], ignore_index=True)

    nrows, ncols = len(mutsig_order), len(growth_model_order)

    # layout: each mutsig row = scatter + optional diff panel
    _scatter_h = (figsize[1] if figsize else 3.5 * nrows)
    _scatter_w = (figsize[0] if figsize else 4 * ncols)
    _diff_h_per_row = 1.5  # inches, added on top of scatter height

    if diff_plot:
        gs_nrows = nrows * 2
        height_ratios = [_scatter_h / nrows, _diff_h_per_row] * nrows
        actual_figsize = (_scatter_w, _scatter_h + _diff_h_per_row * nrows)
    else:
        gs_nrows = nrows
        height_ratios = [1] * nrows
        actual_figsize = (_scatter_w, _scatter_h)

    fig = plt.figure(figsize=actual_figsize)
    gs  = GridSpec(gs_nrows, ncols, figure=fig, height_ratios=height_ratios,
                   hspace=0.22 if diff_plot else 0.3, wspace=0.3)

    axes      = np.empty((nrows, ncols), dtype=object)
    axes_diff = np.empty((nrows, ncols), dtype=object) if diff_plot else None

    for i in range(nrows):
        scatter_row = i * 2 if diff_plot else i
        for j in range(ncols):
            axes[i, j] = fig.add_subplot(gs[scatter_row, j])
            if diff_plot:
                axes_diff[i, j] = fig.add_subplot(gs[scatter_row + 1, j], sharex=axes[i, j])

    # ---- sliding window (x=skyline, y=constcoal) ----
    def _sliding_window(x, y, ws, overlap):
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        step = max(1, ws - overlap)
        xs, ys, ds, x_lefts, x_rights = [], [], [], [], []
        for start in range(0, len(x_s) - ws + 1, step):
            sl = slice(start, start + ws)
            xw, yw = x_s[sl], y_s[sl]
            xs.append(np.median(xw));  ys.append(np.median(yw))
            x_lefts.append(xw[0]);     x_rights.append(xw[-1])
            d_vals = (yw - xw) / xw if diff_mode == "relative" else yw - xw
            ds.append(np.median(d_vals))
        return (np.array(xs), np.array(ys), np.array(ds),
                np.array(x_lefts), np.array(x_rights))

    def _step_path(ys, x_lefts, x_rights):
        n = len(ys)
        if n == 0:
            return np.array([]), np.array([])
        splits = (x_rights[:-1] + x_lefts[1:]) / 2
        xp, yp = [x_lefts[0]], [ys[0]]
        for k in range(n - 1):
            xp += [splits[k], splits[k]];  yp += [ys[k], ys[k + 1]]
        xp.append(x_rights[-1]);  yp.append(ys[-1])
        return np.array(xp), np.array(yp)

    for i, mutsig in enumerate(mutsig_order):
        for j, pop_model in enumerate(growth_model_order):
            ax = axes[i, j]
            ax_d = axes_diff[i, j] if diff_plot else None

            cell_full = wide[(wide["mutation_signal"] == mutsig) &
                             (wide["population_model"] == pop_model)].dropna(
                subset=["skyline_dist_median", "constcoal_dist_median"])

            if cell_full.empty:
                ax.axis("off")
                if ax_d is not None: ax_d.axis("off")
                continue

            cell = cell_full.sample(n=n_subsample, random_state=42) if n_subsample is not None and len(cell_full) > n_subsample else cell_full

            xi = cell["skyline_dist_median"].values
            yi = cell["constcoal_dist_median"].values
            xerr = np.array([xi - cell["skyline_dist_hpd_lower"].values,
                             cell["skyline_dist_hpd_upper"].values - xi])
            yerr = np.array([yi - cell["constcoal_dist_hpd_lower"].values,
                             cell["constcoal_dist_hpd_upper"].values - yi])
            eb = ax.errorbar(xi, yi, xerr=xerr, yerr=yerr,
                             fmt="o", color=color, ecolor=color,
                             elinewidth=errorbar_width, capsize=0, markersize=1, alpha=alpha)
            for line in eb[1]: line.set_alpha(errorbar_alpha)
            for line in eb[2]: line.set_alpha(errorbar_alpha)

            if running_avg or diff_plot:
                x_all = cell_full["skyline_dist_median"].values
                y_all = cell_full["constcoal_dist_median"].values
                win_x, win_y, win_d, win_xl, win_xr = _sliding_window(x_all, y_all, window_size, window_overlap)

                if running_avg and len(win_x):
                    xp, yp = _step_path(win_y, win_xl, win_xr)
                    ax.plot(xp, yp, color=running_avg_color, linewidth=running_avg_linewidth, zorder=5)

                if diff_plot and ax_d is not None and len(win_x):
                    bg_x = cell["skyline_dist_median"].values
                    bg_d = (cell["constcoal_dist_median"].values - bg_x) / bg_x if diff_mode == "relative" else cell["constcoal_dist_median"].values - bg_x
                    ax_d.scatter(bg_x, bg_d, s=1, color="#bbbbbb", alpha=0.3, zorder=1)
                    ax_d.scatter(win_x, win_d, s=4, color=running_avg_color, zorder=3)
                    ax_d.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=4)
                    ax_d.grid(True, color="lightgray", linewidth=0.4, alpha=0.5, zorder=0)
                    if x_range: ax_d.set_xlim(x_range)
                    if diff_axis_range: ax_d.set_ylim(diff_axis_range)
                    ax_d.tick_params(labelsize=tick_fontsize - 2)
                    ylabel = "Rel. diff." if diff_mode == "relative" else "Diff. (years)"
                    if j == 0: ax_d.set_ylabel(ylabel, fontsize=label_fontsize - 1)

            ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1.0, zorder=1)
            ax.set_aspect("equal")
            ax.grid(True, color="lightgray", linewidth=0.4, alpha=0.5, zorder=0)
            if x_range: ax.set_xlim(x_range)
            if y_range: ax.set_ylim(y_range)
            ax.tick_params(labelsize=tick_fontsize)
            ax.set_xlabel("")
            if diff_plot and ax_d is not None:
                ax_d.set_xlabel("Skyline", fontsize=label_fontsize)
            elif not diff_plot:
                ax.set_xlabel("Skyline", fontsize=label_fontsize)
            if j == 0:
                ax.set_ylabel("Const. Coalescent", fontsize=label_fontsize)
            else:
                ax.set_ylabel("")
            if i == 0 and show_pop_model_labels:
                ax.set_title(title_map.get(pop_model, pop_model), fontsize=label_fontsize + 2)

            if j == 0 and i == 0:
                marker_line = Line2D([0], [0], marker="o", color=color, markersize=4,
                                     linestyle="none", alpha=1.0)
                caplines = (Line2D([0], [0], color=color, linewidth=0.4),
                            Line2D([0], [0], color=color, linewidth=0.4))
                barlines = (LineCollection([[[0, -0.3], [0, 0.3]]], colors=[color], linewidths=[0.4]),
                            LineCollection([[[-0.3, 0], [0.3, 0]]], colors=[color], linewidths=[0.4]))
                handle = ErrorbarContainer((marker_line, caplines, barlines),
                                           has_xerr=True, has_yerr=True,
                                           label="Tip dist. from\npairwise MRCA")
                ax.legend(handles=[handle], fontsize=legend_fontsize, frameon=False,
                          loc="upper left", borderpad=0.1, handletextpad=0.5,
                          handlelength=0.5, bbox_to_anchor=(0.0, 1.0))

    suptitle = title if title is not None else f"Pairwise tip distances — {sampling}"
    fig.suptitle(suptitle, fontsize=17)
    top = 0.97 if suptitle else 1.0
    fig.subplots_adjust(top=top)

    # align diff panel widths to the actual equal-aspect scatter square
    if diff_plot:
        fig.canvas.draw()
        _renderer = fig.canvas.get_renderer()
        _inv = fig.transFigure.inverted()
        for i in range(nrows):
            for j in range(ncols):
                if axes_diff[i, j] is None:
                    continue
                bb = axes[i, j].get_window_extent(_renderer)
                x0_fig, _ = _inv.transform((bb.x0, bb.y0))
                x1_fig, _ = _inv.transform((bb.x1, bb.y1))
                dp = axes_diff[i, j].get_position()
                axes_diff[i, j].set_position([x0_fig, dp.y0, x1_fig - x0_fig, dp.height])

    if show_mutsig_labels:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for i, mutsig in enumerate(mutsig_order):
            ax = axes[i][0]
            ylabel_bb = ax.yaxis.label.get_window_extent(renderer)
            x_fig = inv.transform((ylabel_bb.x0, ylabel_bb.y0 + ylabel_bb.height / 2))[0]
            pos = ax.get_position()
            y_fig = (pos.y0 + pos.y1) / 2
            fig.text(x_fig - 0.01, y_fig, mutsig_labels.get(mutsig, mutsig),
                     fontsize=label_fontsize + 2, ha="center", va="center", rotation=90,
                     transform=fig.transFigure)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    from IPython.display import display as _display
    _display(fig)
    plt.close(fig)


def plot_pairwise_tip_distances_zoom(dist_df, sampling,
                                     x_range=None, y_range=None,
                                     zoom_x_range=None, zoom_y_range=None,
                                     alpha=0.5, errorbar_alpha=0.3, errorbar_width=0.2,
                                     n_subsample=None, n_subsample_zoom=None,
                                     pop_model="bottleneck", mutsig="low",
                                     figsize=(12, 5),
                                     title=None,
                                     x_label="Skyline", y_label="Const. Coalescent",
                                     running_avg=False,
                                     window_size=500, window_overlap=250,
                                     window_size_zoom=None, window_overlap_zoom=None,
                                     running_avg_color="#c0392b",
                                     diff_plot=False,
                                     diff_mode="absolute",
                                     diff_axis_range=None,
                                     label_fontsize=13, tick_fontsize=13, legend_fontsize=11,
                                     save_path=None):
    """
    Two-panel figure: left = full scatter, right = zoomed view of a rectangular window.
    Optionally adds a running median line and a difference panel below the zoom panel.

    Shows a single scenario (pop_model × mutsig). The zoom window is marked
    by a rectangle in the main panel.

    Parameters
    ----------
    dist_df : DataFrame
        Output of compute_pairwise_tip_distances().
    sampling : str
        Sampling scheme to plot.
    x_range, y_range : tuple, optional
        Axis limits of the main panel; default to the range of the HPD bounds.
    zoom_x_range, zoom_y_range : tuple, optional
        Window shown in the zoom panel (skyline and constcoal distance).
        Without zoom_x_range only the main panel is drawn; without
        zoom_y_range the y window equals the x window.
    alpha, errorbar_alpha, errorbar_width : float
        Styling of the points and HPD error bars.
    n_subsample, n_subsample_zoom : int, optional
        Randomly draw at most this many points for the main / zoom panel.
    pop_model, mutsig : str
        Scenario to plot (mutsig uses the short names "low", "med", "high").
    figsize : tuple
        Figure size.
    title : str, optional
        Title of the main panel.
    x_label, y_label : str
        Axis labels.
    running_avg : bool
        Draw the sliding-window median as a step line (see
        plot_pairwise_tip_distances).
    window_size, window_overlap : int
        Sliding window size and overlap for the main panel.
    window_size_zoom, window_overlap_zoom : int, optional
        Sliding window size and overlap for the zoom panel; default to the
        main panel values.
    running_avg_color : str
        Colour of the running median and window differences.
    diff_plot : bool
        Add a difference panel below the zoom panel (requires zoom).
    diff_mode : 'absolute' for constcoal - skyline, 'relative' for (constcoal - skyline) / skyline.
    diff_axis_range : tuple, optional
        y-axis limits of the difference panel.
    label_fontsize, tick_fontsize, legend_fontsize : float
        Font sizes.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    """
    from matplotlib.patches import Rectangle
    import matplotlib.colors as _mcolors

    color = "#555555"
    zoom = zoom_x_range is not None

    # ---- data prep ----
    base = dist_df[dist_df["sampling"] == sampling].copy()
    tip_a_df = base.rename(columns={
        "dist_a_constcoal_median":    "constcoal_dist_median",
        "dist_a_constcoal_hpd_lower": "constcoal_dist_hpd_lower",
        "dist_a_constcoal_hpd_upper": "constcoal_dist_hpd_upper",
        "dist_a_skyline_median":      "skyline_dist_median",
        "dist_a_skyline_hpd_lower":   "skyline_dist_hpd_lower",
        "dist_a_skyline_hpd_upper":   "skyline_dist_hpd_upper",
    })
    tip_b_df = base.rename(columns={
        "dist_b_constcoal_median":    "constcoal_dist_median",
        "dist_b_constcoal_hpd_lower": "constcoal_dist_hpd_lower",
        "dist_b_constcoal_hpd_upper": "constcoal_dist_hpd_upper",
        "dist_b_skyline_median":      "skyline_dist_median",
        "dist_b_skyline_hpd_lower":   "skyline_dist_hpd_lower",
        "dist_b_skyline_hpd_upper":   "skyline_dist_hpd_upper",
    })
    keep_cols = ["sampling", "population_model", "mutation_signal", "tree_index",
                 "constcoal_dist_median", "constcoal_dist_hpd_lower", "constcoal_dist_hpd_upper",
                 "skyline_dist_median",   "skyline_dist_hpd_lower",   "skyline_dist_hpd_upper"]
    df_full = pd.concat([tip_a_df[keep_cols], tip_b_df[keep_cols]], ignore_index=True)
    df_full = df_full[(df_full["mutation_signal"] == mutsig) & (df_full["population_model"] == pop_model)]
    df_full = df_full.dropna(subset=["constcoal_dist_median", "skyline_dist_median"])

    df_scatter = df_full.sample(n=n_subsample, random_state=42) if n_subsample is not None and len(df_full) > n_subsample else df_full

    # ---- layout ----
    def _actual_pos(left, bottom, w, h, fw, fh):
        side_in = min(w * fw, h * fh)
        aw, ah = side_in / fw, side_in / fh
        return left + (w - aw) / 2, bottom + (h - ah) / 2, aw, ah

    fw, fh = figsize
    fig = plt.figure(figsize=figsize)

    if zoom:
        ml, mb, mw, mh = 0.10, 0.10, 0.48, 0.82
        ax_main = fig.add_axes([ml, mb, mw, mh])
        al, ab, aw, ah = _actual_pos(ml, mb, mw, mh, fw, fh)
        gap          = 0.06
        zoom_left    = al + aw + gap
        zoom_w       = aw * 0.70
        zoom_h       = ah * 0.70
        zoom_bottom  = ab + ah - zoom_h   # top-aligned with main
        ax_zoom = fig.add_axes([zoom_left, zoom_bottom, zoom_w, zoom_h])
        if diff_plot:
            diff_gap = 0.08
            diff_h   = zoom_bottom - ab - diff_gap
            ax_diff  = fig.add_axes([zoom_left, ab, zoom_w, diff_h])
        else:
            ax_diff = None
    else:
        ax_zoom = None
        ax_diff = None
        ax_main = fig.add_axes([0.10, 0.10, 0.82, 0.82])

    # ---- scatter helper (x=skyline, y=constcoal) ----
    def _scatter(ax_, df_):
        xi  = df_["skyline_dist_median"].values
        yi  = df_["constcoal_dist_median"].values
        xlo = df_["skyline_dist_hpd_lower"].values
        xhi = df_["skyline_dist_hpd_upper"].values
        ylo = df_["constcoal_dist_hpd_lower"].values
        yhi = df_["constcoal_dist_hpd_upper"].values
        from matplotlib.collections import LineCollection as _LC
        x_segs = np.stack([np.column_stack([xlo, yi]), np.column_stack([xhi, yi])], axis=1)
        y_segs = np.stack([np.column_stack([xi, ylo]), np.column_stack([xi, yhi])], axis=1)
        ax_.add_collection(_LC(x_segs, colors="#aaaaaa", linewidths=errorbar_width,
                               alpha=errorbar_alpha, zorder=2))
        ax_.add_collection(_LC(y_segs, colors="#aaaaaa", linewidths=errorbar_width,
                               alpha=errorbar_alpha, zorder=2))
        ax_.scatter(xi, yi, s=1, color=color, alpha=alpha, zorder=3)

    # ---- sliding window stats (x=skyline, y=constcoal) ----
    def _sliding_window(x, y, ws, overlap):
        order  = np.argsort(x)
        x_s, y_s = x[order], y[order]
        step   = max(1, ws - overlap)
        xs, ys, ds, d_los, d_his, x_lefts, x_rights = [], [], [], [], [], [], []
        for start in range(0, len(x_s) - ws + 1, step):
            sl = slice(start, start + ws)
            xw, yw = x_s[sl], y_s[sl]
            xs.append(np.median(xw))
            ys.append(np.median(yw))
            x_lefts.append(xw[0]);  x_rights.append(xw[-1])
            d_vals = (yw - xw) / xw if diff_mode == "relative" else yw - xw
            ds.append(np.median(d_vals))
        return (np.array(xs), np.array(ys), np.array(ds),
                np.array(x_lefts), np.array(x_rights))

    def _step_path(ys, x_lefts, x_rights):
        n = len(ys)
        if n == 0:
            return np.array([]), np.array([])
        splits = (x_rights[:-1] + x_lefts[1:]) / 2
        xp, yp = [x_lefts[0]], [ys[0]]
        for i in range(n - 1):
            xp += [splits[i], splits[i]];  yp += [ys[i], ys[i + 1]]
        xp.append(x_rights[-1]);  yp.append(ys[-1])
        return np.array(xp), np.array(yp)

    # ---- compute window stats on full dataset (x=skyline, y=constcoal) ----
    x_all = df_full["skyline_dist_median"].values
    y_all = df_full["constcoal_dist_median"].values
    if running_avg or diff_plot:
        win_x, win_y, win_d, win_xl, win_xr = _sliding_window(
            x_all, y_all, window_size, window_overlap)

    # ---- main panel ----
    _scatter(ax_main, df_scatter)

    if running_avg:
        xp, yp = _step_path(win_y, win_xl, win_xr)
        ax_main.plot(xp, yp, color=running_avg_color, linewidth=1.5, zorder=5, label="Running median")

    lims = list(x_range) if x_range is not None else None
    if lims is None:
        all_vals = np.concatenate([
            df_scatter["skyline_dist_hpd_lower"].values,   df_scatter["skyline_dist_hpd_upper"].values,
            df_scatter["constcoal_dist_hpd_lower"].values, df_scatter["constcoal_dist_hpd_upper"].values,
        ])
        all_vals = all_vals[np.isfinite(all_vals)]
        lims = [float(all_vals.min()), float(all_vals.max())]

    ax_main.plot(lims, lims, "--", color="black", linewidth=1, zorder=4)
    ax_main.set_xlim(lims)
    ax_main.set_ylim(lims if y_range is None else list(y_range))
    ax_main.set_xlabel(x_label, fontsize=label_fontsize)
    ax_main.set_ylabel(y_label, fontsize=label_fontsize)
    ax_main.tick_params(labelsize=tick_fontsize)
    ax_main.set_aspect("equal")
    if title is not None:
        ax_main.set_title(title, fontsize=label_fontsize + 1)

    marker_line = Line2D([0], [0], marker="o", color=color, markersize=4, linestyle="none", alpha=1.0)
    caplines = (Line2D([0], [0], color=color, linewidth=errorbar_width),
                Line2D([0], [0], color=color, linewidth=errorbar_width))
    barlines = (LineCollection([[[0, -0.3], [0, 0.3]]], colors=[color], linewidths=[errorbar_width]),
                LineCollection([[[-0.3, 0], [0.3, 0]]], colors=[color], linewidths=[errorbar_width]))
    handle = ErrorbarContainer((marker_line, caplines, barlines), has_xerr=True, has_yerr=True,
                                label="Tip dist. from\npairwise MRCA")
    ax_main.legend(handles=[handle], fontsize=legend_fontsize, frameon=False,
                   loc="upper left", borderpad=0.5, handletextpad=0.4)

    # ---- zoom panel ----
    if zoom:
        df_zoom_pool = df_full[
            (df_full["skyline_dist_median"]   >= zoom_x_range[0]) & (df_full["skyline_dist_median"]   <= zoom_x_range[1]) &
            (df_full["constcoal_dist_median"] >= zoom_y_range[0]) & (df_full["constcoal_dist_median"] <= zoom_y_range[1])
        ] if zoom_y_range is not None else df_full[
            (df_full["skyline_dist_median"] >= zoom_x_range[0]) & (df_full["skyline_dist_median"] <= zoom_x_range[1])
        ]
        df_zoom = df_zoom_pool.sample(n=n_subsample_zoom, random_state=43) if n_subsample_zoom is not None and len(df_zoom_pool) > n_subsample_zoom else df_zoom_pool
        _scatter(ax_zoom, df_zoom)

        z_xlim = list(zoom_x_range)
        z_ylim = list(zoom_y_range) if zoom_y_range is not None else z_xlim

        if running_avg or diff_plot:
            ws_z = window_size_zoom   if window_size_zoom   is not None else window_size
            ov_z = window_overlap_zoom if window_overlap_zoom is not None else window_overlap
            zwin_x, zwin_y, zwin_d, zwin_xl, zwin_xr = _sliding_window(
                x_all, y_all, ws_z, ov_z)
            inside = (zwin_xr >= zoom_x_range[0]) & (zwin_xl <= zoom_x_range[1])
            sel = np.where(inside)[0]

            if running_avg and len(sel):
                xp, yp = _step_path(zwin_y[sel], zwin_xl[sel], zwin_xr[sel])
                ax_zoom.plot(xp, yp, color=running_avg_color, linewidth=1.5, zorder=5)

            if diff_plot and ax_diff is not None and len(sel):
                bg_x = df_zoom["skyline_dist_median"].values
                bg_d = (df_zoom["constcoal_dist_median"].values - bg_x) / bg_x if diff_mode == "relative" else df_zoom["constcoal_dist_median"].values - bg_x
                if len(bg_x):
                    ax_diff.scatter(bg_x, bg_d, s=1, color="#bbbbbb", alpha=0.4, zorder=1)
                ax_diff.scatter(zwin_x[sel], zwin_d[sel], s=3**2, color=running_avg_color, zorder=3)
                ax_diff.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=4)
                ax_diff.set_xlim(z_xlim)
                if diff_axis_range is not None:
                    ax_diff.set_ylim(diff_axis_range)
                ylabel = "Rel. diff." if diff_mode == "relative" else "Diff. (years)"
                ax_diff.set_ylabel(ylabel, fontsize=label_fontsize - 1, labelpad=-2)
                ax_diff.set_xlabel(x_label, fontsize=label_fontsize - 1)
                ax_diff.tick_params(labelsize=tick_fontsize - 1)

        ax_zoom.plot(z_xlim, z_ylim, "--", color="black", linewidth=1, zorder=4)
        ax_zoom.set_xlim(z_xlim)
        ax_zoom.set_ylim(z_ylim)
        ax_zoom.set_aspect("equal")
        ax_zoom.tick_params(labelsize=tick_fontsize - 1)
        ax_zoom.set_xlabel("")
        ax_zoom.set_ylabel("")

        # rectangle + connecting lines
        zx0, zx1 = zoom_x_range
        zy0, zy1 = (zoom_y_range[0], zoom_y_range[1]) if zoom_y_range else (zx0, zx1)
        zoom_color = "#a6444f"
        _r, _g, _b, _ = _mcolors.to_rgba(zoom_color)
        rect = Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0,
                          linewidth=1.2, edgecolor=zoom_color,
                          facecolor=(_r, _g, _b, 0.15), linestyle="-", zorder=5)
        ax_main.add_patch(rect)
        for corner_main, corner_zoom in [((zx1, zy1), (zx0, zy1)),
                                          ((zx1, zy0), (zx0, zy0))]:
            con = plt.matplotlib.patches.ConnectionPatch(
                xyA=corner_main, coordsA=ax_main.transData,
                xyB=corner_zoom, coordsB=ax_zoom.transData,
                color=zoom_color, linewidth=0.8, linestyle="--", zorder=5)
            fig.add_artist(con)

    if save_path is not None:
        save_vector_figure(fig, save_path)
    from IPython.display import display as _display
    _display(fig)
    plt.close(fig)


def _ltt_events_from_summary_tree(tree_path):
    """
    Returns sorted internal node heights (time before present) and minimum tip
    height from a BEAST summary tree. Uses height_median annotations.

    Parameters
    ----------
    tree_path : str or Path
        NEXUS summary tree written by treeannotator.

    Returns
    -------
    (list of float, float) or (None, None)
        Internal node heights sorted from oldest to youngest, and the
        youngest tip height; (None, None) if no annotated internal nodes.
    """
    tree = Phylo.read(str(tree_path), "nexus")

    def _h(c):
        if c.comment:
            m = re.search(r'height_median=([\d\.eE+-]+)', c.comment)
            if m:
                return float(m.group(1))
        return None

    int_h = [_h(c) for c in tree.get_nonterminals()]
    int_h = [h for h in int_h if h is not None]
    if not int_h:
        return None, None
    tip_h = [_h(c) or 0.0 for c in tree.get_terminals()]
    return sorted(int_h, reverse=True), min(tip_h)


def _ltt_events_from_true_tree(tree_obj):
    """
    Returns sorted internal node heights and minimum tip height from a
    time-calibrated newick tree.

    Parameters
    ----------
    tree_obj : Bio.Phylo tree
        Simulated tree; heights are computed as root depth minus node depth.

    Returns
    -------
    (list of float, float)
        Internal node heights sorted from oldest to youngest, and the
        youngest tip height.
    """
    depths     = tree_obj.depths(unit_branch_lengths=False)
    root_depth = max(depths.values())
    tip_h  = [root_depth - d for c, d in depths.items() if c.is_terminal()]
    int_h  = [root_depth - d for c, d in depths.items() if not c.is_terminal()]
    return sorted(int_h, reverse=True), min(tip_h)


def _events_to_ltt(events, min_tip):
    """
    Convert sorted branching event heights to step-function arrays.

    Parameters
    ----------
    events : list of float
        Internal node heights, sorted from oldest (root) to youngest.
    min_tip : float
        Height of the youngest tip, where the curve ends.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Times (decreasing, from the root to min_tip) and the number of
        lineages from each time onwards. Lineages are counted as 1 + number
        of branching events passed, so tips sampled through time (which
        remove lineages going back in time) are not accounted for.
    """
    root_h = events[0]
    times  = [root_h]
    counts = [1]
    for h in events:
        times.append(h)
        counts.append(counts[-1] + 1)
    times.append(min_tip)
    counts.append(counts[-1])
    return np.array(times), np.array(counts)


def _interpolate_ltt(times, counts, grid):
    """
    Evaluate a step-function LTT at grid points (grid decreasing, oldest first).

    Parameters
    ----------
    times, counts : numpy.ndarray
        Step function from _events_to_ltt().
    grid : array-like
        Times at which to evaluate the step function.

    Returns
    -------
    numpy.ndarray
        Number of lineages at each grid time. Grid times older than the
        root wrap around to counts[-1] (see _interpolate_ltt_nan_safe).
    """
    # times are decreasing (past -> present), counts increasing
    return np.array([counts[np.searchsorted(-times, -t, side="right") - 1]
                     for t in grid], dtype=float)


def compute_ltt(csv_path, sim_dir,
                sampling=None, pop_models=None, mutsigs=None,
                repo_base=None,
                n_grid=1000,
                grid_end=None,
                out_per_tree=None,
                out_median=None):
    """
    Compute LTT curves for constcoal, skyline, and true trees.

    Estimated LTTs use the median node heights of the summary trees
    (*.combined_summary.tree); true LTTs use the simulated trees. Only
    replicates with a non-empty summary tree for both models are included.

    Parameters
    ----------
    csv_path : str or Path
        beast_inference/successful_mcmc_runs.csv written by the Snakefile.
    sim_dir : str or Path
        Simulated data directory containing {sampling}/{pop_model}/{pop_model}.trees.
    sampling : str, optional
        Sampling scheme; defaults to the first one found in csv_path.
    pop_models, mutsigs : str or list of str, optional
        Population models and mutation signals ("low", "med", "high") to compute.
    repo_base : str or Path, optional
        Directory the trees_path entries are relative to; defaults to three
        levels above csv_path (the project root).
    n_grid : int
        Number of time points of the shared grid used for the median.
    grid_end : float, optional
        Oldest time of the shared grid; defaults to the oldest root.
    out_per_tree, out_median : str or Path, optional
        If given, write the two result tables to these TSV files.

    Outputs
    -------
    out_per_tree : TSV with columns
        model, sampling, pop_model, mutsig, tree_index, time, lineages
    out_median : TSV with columns
        model, sampling, pop_model, mutsig, time, median_lineages
        (median over all trees evaluated on a shared grid per scenario)

    Returns (per_tree_df, median_df).
    """
    csv_path = Path(csv_path)
    sim_dir  = Path(sim_dir)
    base     = Path(repo_base) if repo_base else csv_path.parents[2]

    df  = pd.read_csv(csv_path)
    pat = re.compile(
        r'(constcoal|skyline)/([^/]+)/([^/]+)/(\w+mutsig)/\w+\.T(\d+)\.combined\.trees'
    )
    rows = []
    for p in df["trees_path"]:
        m = pat.search(p)
        if m:
            rows.append({"model": m.group(1), "sampling": m.group(2),
                         "pop_model": m.group(3),
                         "mutsig": m.group(4).replace("mutsig", ""),
                         "tree_index": int(m.group(5)), "trees_path": p})
    meta = pd.DataFrame(rows)

    if sampling is not None:
        meta = meta[meta["sampling"] == sampling]
    _sampling = sampling or (meta["sampling"].iloc[0] if not meta.empty else "")

    mutsig_order   = ([mutsigs]    if isinstance(mutsigs, str)    else mutsigs)    or ["low", "med", "high"]
    popmodel_order = ([pop_models] if isinstance(pop_models, str) else pop_models) or ["uniform", "expgrowthslow", "expgrowthfast", "bottleneck"]

    per_tree_rows = []
    median_rows   = []

    for ms in mutsig_order:
        for pm in popmodel_order:
            sim_path = sim_dir / _sampling / pm / f"{pm}.trees"
            if not sim_path.exists():
                print(f"Skipping {pm}/{ms}: sim tree not found at {sim_path}")
                continue
            true_trees = list(Phylo.parse(str(sim_path), "newick"))

            sub = meta[(meta["mutsig"] == ms) & (meta["pop_model"] == pm)]

            # accumulate per-model LTT arrays for median computation
            model_ltt_arrays = {"constcoal": [], "skyline": [], "true": []}
            all_times = {"constcoal": [], "skyline": [], "true": []}

            for t_idx, grp in sub.groupby("tree_index"):
                cc_row = grp[grp["model"] == "constcoal"]
                sk_row = grp[grp["model"] == "skyline"]
                if cc_row.empty or sk_row.empty or t_idx >= len(true_trees):
                    continue

                cc_sum = base / cc_row.iloc[0]["trees_path"].replace(".combined.trees", ".combined_summary.tree")
                sk_sum = base / sk_row.iloc[0]["trees_path"].replace(".combined.trees", ".combined_summary.tree")

                if not cc_sum.exists() or cc_sum.stat().st_size == 0:
                    continue
                if not sk_sum.exists() or sk_sum.stat().st_size == 0:
                    continue

                cc_ev, cc_tip = _ltt_events_from_summary_tree(cc_sum)
                sk_ev, sk_tip = _ltt_events_from_summary_tree(sk_sum)
                tr_ev, tr_tip = _ltt_events_from_true_tree(true_trees[t_idx])

                if cc_ev is None or sk_ev is None:
                    continue

                for label, ev, tip in [("constcoal", cc_ev, cc_tip),
                                        ("skyline",   sk_ev, sk_tip),
                                        ("true",      tr_ev, tr_tip)]:
                    times, counts = _events_to_ltt(ev, tip)
                    for t, c in zip(times, counts):
                        per_tree_rows.append({
                            "model": label, "sampling": _sampling,
                            "pop_model": pm, "mutsig": ms,
                            "tree_index": t_idx, "time": t, "lineages": c,
                        })
                    all_times[label].extend(times.tolist())
                    model_ltt_arrays[label].append((times, counts))

            # compute median on shared grid per model
            for label, ltt_list in model_ltt_arrays.items():
                if not ltt_list:
                    continue
                t_max = grid_end if grid_end is not None else max(t[0][0] for t in ltt_list)
                t_min = min(t[0][-1] for t in ltt_list)
                grid  = np.linspace(t_max, t_min, n_grid)
                matrix = np.vstack([_interpolate_ltt(t, c, grid)
                                    for t, c in ltt_list])
                med = np.median(matrix, axis=0)
                for t, m_val in zip(grid, med):
                    median_rows.append({
                        "model": label, "sampling": _sampling,
                        "pop_model": pm, "mutsig": ms,
                        "time": t, "median_lineages": m_val,
                    })

    per_tree_df = pd.DataFrame(per_tree_rows)
    median_df   = pd.DataFrame(median_rows)

    if out_per_tree:
        per_tree_df.to_csv(out_per_tree, sep="\t", index=False)
        print(f"Saved per-tree LTT to {out_per_tree}")
    if out_median:
        median_df.to_csv(out_median, sep="\t", index=False)
        print(f"Saved median LTT to {out_median}")

    return per_tree_df, median_df


def _interpolate_ltt_nan_safe(times, counts, grid):
    """
    Local variant of _interpolate_ltt used only by plot_ltt's mode="band" (not
    by compute_ltt, which keeps using the original _interpolate_ltt as-is).
    Returns NaN at grid times older than this tree's own root, instead of
    _interpolate_ltt's counts[-1]-wraparound behaviour there — so a band/HPD
    computed across trees at a given time only includes trees that actually
    exist at that time.
    """
    times = np.asarray(times)
    counts = np.asarray(counts, dtype=float)
    idx = np.searchsorted(-times, -np.asarray(grid), side="right")
    return np.where(idx > 0, counts[np.clip(idx - 1, 0, len(counts) - 1)], np.nan)


def plot_ltt(per_tree_df, median_df,
             sampling=None, pop_models=None, mutsigs=None,
             alpha=0.15, linewidth=0.5,
             median_linewidth=1.8,
             constcoal_color="#397398",
             skyline_color="#7e4d8b",
             true_color="#a6444f",
             x_range=None, y_range=None,
             figsize=None,
             title=None,
             show_pop_model_labels=True,
             show_mutsig_labels=True,
             sampling_end_time=None,
             bottleneck_window=None, bottleneck_alpha=0.2,
             label_fontsize=14, tick_fontsize=14, legend_fontsize=14,
             log_y=False,
             mode="lines",
             band_time_points=250, band_alpha=0.25, band_credible_mass=0.95,
             save_path=None):
    """
    Plot LTT curves from precomputed compute_ltt() outputs.

    mode="lines" (default): thin lines = individual trees; thick lines =
        median across trees (original behaviour).
    mode="band": instead of drawing every tree, each tree's step function
        (from per_tree_df) is resampled onto a shared time grid and, at each
        grid time, a `band_credible_mass` HPD interval across trees is
        computed (via the module-level hpd() helper) and drawn as a shaded
        ribbon around the median line — shows the spread of tree
        trajectories without the visual clutter of many overlapping thin
        lines. Grid times older than a given tree's own root are excluded
        for that tree (it doesn't exist yet), handled locally here without
        touching compute_ltt/_interpolate_ltt.

    Parameters
    ----------
    per_tree_df, median_df : DataFrame
        The two outputs of compute_ltt() (or the TSVs it writes).
    sampling : str, optional
        Sampling scheme to plot; default: all rows.
    pop_models, mutsigs : str or list of str, optional
        Population models (columns) and mutation signals (rows) to show;
        default: all in per_tree_df.
    alpha, linewidth : float
        Styling of the per-tree lines (mode="lines").
    median_linewidth : float
        Width of the median lines.
    constcoal_color, skyline_color, true_color : str
        Colours per tree type.
    x_range, y_range : tuple, optional
        Axis limits; x_range also sets the time grid in mode="band".
    figsize : tuple, optional
        Override the default figure size.
    title : str, optional
        Suptitle; defaults to the sampling scheme ("linearconstant" is not
        shown as a title).
    show_pop_model_labels, show_mutsig_labels : bool
        Toggle column titles and row labels.
    sampling_end_time : float, optional
        If given, draw an arrow labelled "sampling" from 0 to this time.
    bottleneck_window : tuple, optional
        (start, end) of the bottleneck, shaded in bottleneck panels.
    bottleneck_alpha : float
        Transparency of the bottleneck shading.
    label_fontsize, tick_fontsize, legend_fontsize : float
        Font sizes.
    log_y : bool
        Logarithmic y-axis.
    mode : {"lines", "band"}
        See above.
    band_time_points : int
        Number of grid times for mode="band".
    band_alpha : float
        Transparency of the band (fill is drawn at 0.4 × band_alpha).
    band_credible_mass : float
        Credible mass of the HPD band.
    save_path : str or Path, optional
        If given, also save the figure (see save_vector_figure).
    """
    colors = {"constcoal": constcoal_color, "skyline": skyline_color, "true": true_color}
    labels = {"constcoal": "Const. coal.", "skyline": "Skyline", "true": "True"}

    if sampling is not None:
        per_tree_df = per_tree_df[per_tree_df["sampling"] == sampling]
        median_df   = median_df[median_df["sampling"] == sampling]

    mutsig_order   = ([mutsigs]    if isinstance(mutsigs, str)    else mutsigs)    or sorted(per_tree_df["mutsig"].unique())
    popmodel_order = ([pop_models] if isinstance(pop_models, str) else pop_models) or sorted(per_tree_df["pop_model"].unique())

    title_map     = {"expgrowthfast": "Exp. Growth (fast)", "expgrowthslow": "Exp. Growth (slow)",
                     "uniform": "Uniform", "bottleneck": "Bottleneck"}
    mutsig_labels = {"low": "Low mut. signal", "med": "Med. mut. signal", "high": "High mut. signal"}

    nrows, ncols = len(mutsig_order), len(popmodel_order)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=figsize or (4 * ncols, 3.5 * nrows),
                             squeeze=False)

    for i, ms in enumerate(mutsig_order):
        for j, pm in enumerate(popmodel_order):
            ax = axes[i, j]
            pt  = per_tree_df[(per_tree_df["mutsig"] == ms) & (per_tree_df["pop_model"] == pm)]
            med = median_df[(median_df["mutsig"] == ms) & (median_df["pop_model"] == pm)]

            if pt.empty:
                ax.axis("off"); continue

            # shared time domain for this subplot, used both to clip lines
            # and (in density mode) to build the density grid
            if x_range:
                t_hi, t_lo = max(x_range), min(x_range)
            else:
                t_hi, t_lo = pt["time"].max(), pt["time"].min()

            legend_handles = {}
            for model in ("constcoal", "skyline", "true"):
                color = colors[model]
                pt_m  = pt[pt["model"] == model]

                if mode == "band" and not pt_m.empty:
                    time_grid = np.linspace(t_hi, t_lo, band_time_points)
                    curves = []
                    for t_idx, grp in pt_m.groupby("tree_index"):
                        grp = grp.sort_values("time", ascending=False)
                        curves.append(_interpolate_ltt_nan_safe(grp["time"].values, grp["lineages"].values, time_grid))
                    if not curves:
                        continue
                    curves = np.array(curves)  # (n_trees, n_time); NaN before a tree's own root

                    band_lo = np.full(band_time_points, np.nan)
                    band_hi = np.full(band_time_points, np.nan)
                    band_med = np.full(band_time_points, np.nan)
                    for col in range(band_time_points):
                        col_vals = curves[:, col]
                        col_vals = col_vals[~np.isnan(col_vals)]
                        if len(col_vals) == 0:
                            continue
                        band_med[col] = np.median(col_vals)
                        if len(col_vals) >= 2:
                            band_lo[col], band_hi[col] = hpd(np.sort(col_vals), band_credible_mass)
                        else:
                            band_lo[col] = band_hi[col] = col_vals[0]

                    valid = ~np.isnan(band_med)
                    poly = ax.fill_between(time_grid[valid], band_lo[valid], band_hi[valid],
                                           color=color, alpha=band_alpha * 0.4, zorder=1, linewidth=0)
                    ax.plot(time_grid[valid], band_lo[valid],
                            color=color, linestyle="--", linewidth=1.0, alpha=band_alpha, zorder=2)
                    ax.plot(time_grid[valid], band_hi[valid],
                            color=color, linestyle="--", linewidth=1.0, alpha=band_alpha, zorder=2)
                    legend_handles.setdefault(model, poly)
                    # this band's own median (per_tree_df-derived) replaces the
                    # median_df line for this model, so both stay consistent
                    ax.plot(time_grid[valid], band_med[valid],
                            color=color, linewidth=median_linewidth, zorder=5)
                    continue  # skip the median_df line below for this model
                elif mode == "lines":
                    # thin lines per tree
                    for t_idx, grp in pt_m.groupby("tree_index"):
                        grp = grp.sort_values("time", ascending=False)
                        l, = ax.step(grp["time"], grp["lineages"], where="post",
                                     color=color, alpha=alpha, linewidth=linewidth)
                        legend_handles.setdefault(model, l)

                # median line (drawn on top in both modes)
                med_m = med[med["model"] == model].sort_values("time", ascending=False)
                if not med_m.empty:
                    ax.plot(med_m["time"], med_m["median_lineages"],
                            color=color, linewidth=median_linewidth, zorder=5)

            ax.set_xlabel("Time (before present)", fontsize=label_fontsize)
            ax.set_ylabel("n. lineages", fontsize=label_fontsize)
            ax.invert_xaxis()
            if log_y:
                ax.set_yscale("log")
            ax.grid(True, axis="x", color="lightgray", linewidth=0.4, alpha=0.5)
            ax.tick_params(labelsize=tick_fontsize)
            if x_range: ax.set_xlim(x_range)
            if y_range:
                y_lo = max(y_range[0], 1e-1) if log_y else y_range[0]
                ax.set_ylim(y_lo, y_range[1])

            if sampling_end_time is not None:
                ylims = ax.get_ylim()
                y_arrow = ylims[0] + (ylims[1] - ylims[0]) * 0.07
                ax.annotate(
                    "", xy=(sampling_end_time, y_arrow), xytext=(0, y_arrow),
                    arrowprops=dict(arrowstyle="-|>", color="grey", lw=1.5),
                    annotation_clip=False,
                )
                ax.text(
                    sampling_end_time / 2, y_arrow + (ylims[1] - ylims[0]) * 0.02,
                    "sampling", color="grey", fontsize=label_fontsize - 1, ha="center", va="bottom",
                )

            # bottleneck period as full-height shaded band, e.g. bottleneck_window=(20, 23)
            if bottleneck_window is not None and pm.startswith("bottleneck"):
                ax.axvspan(*bottleneck_window, color="grey", alpha=bottleneck_alpha,
                           linewidth=0, zorder=0)

            if legend_handles:
                legend_lines = [
                    plt.Line2D([0], [0], color=colors[k], linewidth=2.5)
                    for k in legend_handles
                ]
                ax.legend(
                    handles=legend_lines,
                    labels=[labels[k] for k in legend_handles],
                    fontsize=legend_fontsize, frameon=False)
            if i == 0 and show_pop_model_labels and len(popmodel_order) > 1:
                ax.set_title(title_map.get(pm, pm), fontsize=label_fontsize + 1)
            if show_mutsig_labels and j == 0 and len(mutsig_order) > 1:
                ax.set_ylabel(f"{mutsig_labels.get(ms, ms)}\nn. lineages", fontsize=label_fontsize)
            else:
                ax.set_ylabel("n. lineages", fontsize=label_fontsize)

    suptitle = title if title is not None else (sampling or "")
    if suptitle and suptitle != "linearconstant":
        plt.suptitle(suptitle, fontsize=17)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout()

    if save_path is not None:
        save_vector_figure(fig, save_path)
    from IPython.display import display as _display
    _display(fig)
    plt.close(fig)

