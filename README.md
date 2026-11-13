# Molecular signal determines the sensitivity of phylogenetic node height estimates to demographic assumptions in Bayesian frameworks

Time-calibrated phylogenies are widely used to reconstruct pathogen evolution and transmission, but in a Bayesian framework node height estimates depend on both molecular information and assumptions about population dynamics. In this study, we investigate when demographic assumptions influence estimates of node heights in dated phylogenies.
We derive analytical posterior distributions for the time to the most recent common ancestor of two sequences under constant population size, exponential growth, and bottlenecks. We show that demographic assumptions can have a large impact on node height estimates, while the influence decreases with increasing molecular signal and population sizes.
We then simulate phylogenies and sequence data and re-estimate node times on fixed topologies using constant-size and skyline coalescent priors. Demographic assumptions have little effect when molecular signal is high, but increasingly affect deep node height estimates as signal declines. The skyline prior reduces these errors under strong population changes.
Finally, we analyse two case studies. For the initial SARS-CoV-2 Alpha variant wave in the UK, a misspecified constant-size prior substantially shifts the estimated root towards the past, whereas a skyline prior recovers more plausible timing. For \textit{Bordetella pertussis} in France, where transmission was halted during the COVID-19 pandemic, a constant-size prior similarly pushes internal node estimates towards older dates.
Overall, our results demonstrate that demographic misspecification can bias the dating of phylogenies, particularly when molecular signal is limited or population size undergoes rapid changes to low levels.

## Part 2: Simulation Study

This repository contains the [Snakemake](https://snakemake.readthedocs.io) workflow for the simulation study. For every scenario it

1. simulates 100 dated trees with ReMASTER (BEAST2),
2. simulates sequence alignments on each tree with seq-gen and keeps only the SNP sites (snp-sites),
3. writes BEAST XMLs that fix the true topology and re-estimate node heights under a **constant-size** (`constcoal`) and a **skyline** (`skyline`) coalescent prior,
4. runs BEAST (BEAST 1.10.4) with three seeds per XML, combines the runs and checks convergence (ESS ≥ 200),
5. compares the posterior node heights and population sizes to the simulated truth.

### Scenarios

| Factor | Values | Snakefile variable |
|---|---|---|
| Sampling scheme | `independenthomochronous`, `linearconstant` | `SAMPLING_TYPES` |
| True population model | `expgrowthfast`, `expgrowthslow`, `uniform`, `bottleneck` | `POP_MODELS` |
| Molecular signal (alignment length) | `lowmutsig` (1 Mb), `medmutsig` (5 Mb), `highmutsig` (10 Mb) | `MUTSIGS`, `MUTSIG_LENGTHS` |
| Inference prior | `constcoal`, `skyline` | `INFERENCE` |
| Replicate trees per scenario | 100 | `NREPLICATES` |
| BEAST seeds per XML | 44, 45, 46 | `SEEDS` |

This gives 2 × 4 × 3 × 2 × 100 = 4,800 XMLs and 14,400 BEAST jobs.

### Workflow overview

| Step | Rule | Output |
|---|---|---|
| 1 | `simulate_trees` | `simulated_data/{sampling}/{popmodel}/{popmodel}.trees` (one newick per line) and `.traj` |
| 2a | `simulate_alignment` | `{popmodel}_{i}_{mutsig}_snps.fasta` (temporary) |
| 2b | `snp_summary` | `simulated_data/snp_summary.csv` (only via `all_alignments`) |
| 3 | `make_beast_xml` | `beast_inference/{model}/{sampling}/{popmodel}/{mutsig}/*.T{i}.xml` |
| 4 | `run_beast` | `seed{seed}/*.T{i}.log`, `.trees` (temporary) |
| 5 | `combine_runs` | `*.T{i}.combined.log`, `.combined.trees` (10% burnin removed per seed) |
| 6 | `make_summary_tree` | `*.T{i}.combined_summary.tree` (empty if the ESS check failed) |
| 7 | `build_csv` | `beast_inference/successful_mcmc_runs.csv` |
| 8 | `process_results` | `evaluation/{sampling}/{popmodel}/{low,med,high}.tsv`, `evaluation/successful_counts.tsv` |
| 9 | `subsample_runs` | `*.T{i}.subsampled.log`, `.subsampled.trees` (~1000 samples) |
| 10 | `compute_errors` | `evaluation/.../*_node_errors.tsv`, `*_pop_summary.pkl`, `*_pop_trajectory.tsv` |
| 11 | `compute_time_bins` | `evaluation/.../*_time_bins_w{bw}_c{cutoff}[_per_tree].tsv` |

All paths are relative to `results/run1/`. A replicate is only evaluated (steps 8–11) if **both** the `constcoal` and the `skyline` run passed the ESS check.

### Installation

The pipeline is written for the ETH Euler cluster. It needs:

- **Cluster modules** (loaded automatically by each rule): BEAST2 2.7.4 with ReMASTER 2.7.4 (tree simulation), BEAST 1.10.4 with BEAGLE 3.1.2, LogCombiner and TreeAnnotator 1.10.4 (inference, combining runs, summary trees), and R 4.4.0 (ESS check). Install the R packages `coda` 0.19.4.1 and `beastio` 0.3.3 into your R library once:
  ```bash
  module load stack/2024-06 r/4.4.0
  Rscript -e 'install.packages(c("coda", "remotes")); remotes::install_github("laduplessis/beastio")'
  ```
- **Seq-Gen 1.3.5** (alignment simulation), compiled at `../Seq-Gen-1.3.5/source/seq-gen`, i.e. next to this project folder.
- **Three conda environments:**
  ```bash
  # Snakemake 9.19.0 with the SLURM executor plugin 2.6.1
  conda create -n snakemake -c conda-forge -c bioconda snakemake=9.19.0 snakemake-executor-plugin-slurm=2.6.1 pandas
  # snp-sites 2.5.1 (SNP extraction)
  conda create -n snp_sites -c conda-forge -c bioconda snp-sites=2.5.1
  # Python scripts of the pipeline and the analysis notebooks (`beast_tools` environment)
  conda env create -f environment.yaml
  ```

### Running the workflow
- From the project folder on Euler, submit the pipeline with `sbatch scripts/run_snakemake.sh`. It activates the `snakemake` environment and runs all rules as SLURM jobs using the profile in `profiles/euler`.
- All inputs and results are in the `results/run1/` folder: ReMASTER and BEAST templates (`templates/`), config files for the different settings (`config/`), the true simulated trees (`simulated_data/`), the BEAST runs (`beast_inference/`) and the evaluation tables used by the notebooks (`evaluation/`).

### Analysis of results

The pipeline output in `results/run1/evaluation/` is analysed in three Jupyter notebooks in `workflows/`, run with the `beast_tools` environment. Figures are saved to `figures/`.

- `evaluate_populationsize.ipynb`: number of successful runs per scenario, and the estimated population size trajectories (skyline and constant coalescent) compared with the true trajectory.
- `evaluate_root_heights.ipynb`: root age errors under both priors, the difference between priors, HPD coverage of the true root age, and how root age errors relate to errors in the population size estimate.
- `evaluate_branch_lengths.ipynb`: total tree length and per-branch errors, HPD coverage of branch lengths, branch length errors through time and their relation to population size errors, lineages-through-time plots, and pairwise tip-to-MRCA distances.

All loading, computation and plotting functions used by the notebooks are in `workflows/evaluation_helpers.py`, imported with `from evaluation_helpers import *`.

Some intermediate tables are computed in the notebooks, saved to `results/run1/evaluation/` and read back in later cells: `root_height_info.csv` (`evaluate_root_heights.ipynb`), and `all_node_info.csv`, `ltt_per_tree.tsv`, `ltt_median.tsv` and `pairwise_tip_distances_*.tsv` (`evaluate_branch_lengths.ipynb`). On a fresh checkout, run the cells that compute these tables before the cells that load them. The pairwise tip distance files are not tracked in git, and computing them takes a while.