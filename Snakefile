# =============================================================================
# Snakefile for the coalescent prior simulation study (Part 2)
#
# Simulates dated trees under different population models and sampling
# schemes, simulates sequence alignments of varying molecular signal on them,
# re-estimates node heights on the fixed true topology with BEAST under a
# constant-size (constcoal) and a skyline coalescent prior, and evaluates the
# node height / population size estimates against the truth.
#
# Pipeline (rule names in brackets):
#    1  Simulate 100 trees per scenario with ReMASTER        [simulate_trees]
#    2a Simulate alignments with seq-gen, keep only SNPs      [simulate_alignment]
#    2b Summarise number of SNPs per alignment (optional)     [snp_summary]
#    3  Generate one BEAST XML per tree                       [make_beast_xml]
#    4  Run BEAST once per XML and seed                       [run_beast]
#    5  Combine the seed runs with logcombiner (10% burnin)   [combine_runs]
#    6  Check ESS >= 200 and build summary tree (treeannotator) [make_summary_tree]
#    7  List all runs that passed the ESS check               [build_csv]
#    8  Build per-scenario tables of successful runs          [process_results]
#    9  Subsample combined chains to ~1000 samples            [subsample_runs]
#   10  Per-node errors and population trajectories          [compute_errors]
#   11  Time-binned error summaries                          [compute_time_bins]
#
# Wildcards used throughout:
#   sampling  - sampling scheme      (SAMPLING_TYPES)
#   popmodel  - true population model (POP_MODELS)
#   mutsig    - molecular signal level, i.e. alignment length (MUTSIGS)
#   model     - inference prior      (INFERENCE)
#   i         - replicate tree index, 0 .. NREPLICATES-1
#   seed      - BEAST seed           (SEEDS)
#
# Run from the project root, i.e. the directory containing this Snakefile,
# scripts/ and results/ (all paths below are relative to the working
# directory, except SEQGEN which is relative to this Snakefile).
# See README.md for prerequisites and the full set of run instructions.
#
# Usage:
#   Dry run:   snakemake -n
#   Cluster:   snakemake --profile profiles/euler --rerun-incomplete
#   Subset:    snakemake --profile profiles/euler all_trees   (or all_alignments, all_xmls)
# =============================================================================

# --- Simulation study design -------------------------------------------------
SAMPLING_TYPES = ["independenthomochronous", "linearconstant"]
POP_MODELS     = ["expgrowthfast", "expgrowthslow", "uniform", "bottleneck"]
NREPLICATES = 100                # trees simulated per (sampling, popmodel)
INFERENCE      = ["constcoal", "skyline"]
MUTSIGS        = ["lowmutsig", "medmutsig", "highmutsig"]
# Simulated alignment length (bp) per molecular signal level
MUTSIG_LENGTHS = {"lowmutsig": 1000000, "medmutsig": 5000000, "highmutsig": 10000000}
SEEDS          = [44, 45, 46]   # one BEAST run per seed, combined later

# --- Input / output directories (relative to the working directory) ----------
SIMDATA    = "results/run1/simulated_data"     # simulated trees and SNP alignments
BEAST_DIR  = "results/run1/beast_inference"    # BEAST XMLs, logs and trees
CONFIG_DIR = "results/run1/config"             # MakeBEASTXML config files (input)
TEMPLATES_DIR   = "results/run1/templates"     # ReMASTER template XMLs (input)

# seq-gen binary, expected next to the project directory
SEQGEN     = workflow.basedir + "/../Seq-Gen-1.3.5/source/seq-gen"
REMASTER_SEED   = 42

def config_file(wildcards):
    """Return the MakeBEASTXML config file for one (model, sampling, popmodel, mutsig) scenario."""
    return f"{CONFIG_DIR}/{wildcards.model}_{wildcards.sampling}_{wildcards.popmodel}_{wildcards.mutsig}.cfg"

def get_burnin(wildcards):
    """Return 10% burnin in states for logcombiner.

    The values must match the chain lengths set in the corresponding config
    files / BEAST templates; update both together if chain lengths change.
    """
    if wildcards.model == "constcoal":
        return 3000000        # 10% of 30M
    if wildcards.sampling == "independenthomochronous":
        if wildcards.popmodel in ("bottleneck", "bottleneck20", "bottleneck50", "bottleneck100", "bottleneck200", "bottlenecklate", "bottlenecklatesampling", "bottleneckmid50", "bottleneckmid100"):
            return 30000000  # 10% of 300M (all mutsigs)
        elif wildcards.mutsig == "highmutsig":
            return 5000000    # 10% of 50M
        elif wildcards.mutsig == "medmutsig":
            return 10000000   # 10% of 100M
        else:                 # lowmutsig
            return 30000000   # 10% of 300M
    else:  # linearconstant
        if wildcards.popmodel in ("bottleneck", "bottleneck20", "bottleneck50", "bottleneck100", "bottleneck200", "bottlenecklate", "bottlenecklatesampling", "bottleneckmid50", "bottleneckmid100"):
            return 20000000   # 10% of 200M
        elif wildcards.mutsig == "lowmutsig":
            return 12000000   # 10% of 120M
        else:
            return 8000000    # 10% of 80M


# =============================================================================
# Rule all — final targets
# Final outputs per (sampling, popmodel, mutsig) scenario in EVAL_DIR:
#   *_node_errors.tsv, *_pop_summary.pkl, *_pop_trajectory.tsv  (compute_errors)
#   *_time_bins_w{bw}_c{cutoff}[_per_tree].tsv
#                                                               (compute_time_bins)
# =============================================================================
EVAL_DIR     = BEAST_DIR.replace("beast_inference", "evaluation")
MUTSIG_SHORT = {"lowmutsig": "low", "medmutsig": "med", "highmutsig": "high"}

# Time binning configurations per population model, as (bin width, cutoff)
# in time units of the simulated trees. Bins are [0, bw, 2*bw, ..., cutoff, inf].
_CONFIGS_ALL        = [(5, 400), (5, 100), (10, 300)]
_CONFIGS_NOBOTTLE   = [(10, 100), (20, 200)]
_CONFIGS_BOTTLE     = [(1, 30), (1, 50)]

TIME_BIN_CONFIGS = {
    "uniform":       _CONFIGS_ALL + _CONFIGS_NOBOTTLE,
    "expgrowthfast": _CONFIGS_ALL + _CONFIGS_NOBOTTLE,
    "expgrowthslow": _CONFIGS_ALL + _CONFIGS_NOBOTTLE,
    "bottleneck":    _CONFIGS_ALL + _CONFIGS_BOTTLE,
}

# flat set of all (bw, cutoff) combos for wildcard_constraints
_ALL_CONFIGS = set(
    (bw, co)
    for configs in TIME_BIN_CONFIGS.values()
    for bw, co in configs
)

rule all:
    input:
        expand(
            f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{sampling}}_{{popmodel}}_{{mutsig}}_node_errors.tsv",
            sampling=SAMPLING_TYPES, popmodel=POP_MODELS, mutsig=MUTSIGS
        ),
        expand(
            f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{sampling}}_{{popmodel}}_{{mutsig}}_pop_summary.pkl",
            sampling=SAMPLING_TYPES, popmodel=POP_MODELS, mutsig=MUTSIGS
        ),
        expand(
            f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{sampling}}_{{popmodel}}_{{mutsig}}_pop_trajectory.tsv",
            sampling=SAMPLING_TYPES, popmodel=POP_MODELS, mutsig=MUTSIGS
        ),
        [
            f"{EVAL_DIR}/{sampling}/{popmodel}/{sampling}_{popmodel}_{mutsig}_time_bins_w{int(bw)}_c{int(co)}.tsv"
            for sampling in SAMPLING_TYPES
            for popmodel in POP_MODELS
            for mutsig in MUTSIGS
            for bw, co in TIME_BIN_CONFIGS[popmodel]
        ] + [
            f"{EVAL_DIR}/{sampling}/{popmodel}/{sampling}_{popmodel}_{mutsig}_time_bins_w{int(bw)}_c{int(co)}_per_tree.tsv"
            for sampling in SAMPLING_TYPES
            for popmodel in POP_MODELS
            for mutsig in MUTSIGS
            for bw, co in TIME_BIN_CONFIGS[popmodel]
        ],


# -----------------------------------------------------------------------------
# Intermediate targets, useful to run the pipeline in stages:
#   all_trees      - step 1 only (simulated trees)
#   all_alignments - steps 1-2 (SNP alignments + snp_summary.csv)
#   all_xmls       - steps 1-3 (BEAST XMLs, ready to run)
# Note: SNP alignments are temp() files and are removed by Snakemake once all
# consuming rules have finished.
# -----------------------------------------------------------------------------
rule all_trees:
    input:
        expand(
            f"{SIMDATA}/{{sampling}}/{{popmodel}}/{{popmodel}}.trees",
            sampling=SAMPLING_TYPES, popmodel=POP_MODELS
        )


rule all_alignments:
    input:
        expand(
            f"{SIMDATA}/{{sampling}}/{{popmodel}}/{{popmodel}}_{{i}}_{{mutsig}}_snps.fasta",
            sampling=SAMPLING_TYPES, popmodel=POP_MODELS, i=range(NREPLICATES), mutsig=MUTSIGS
        ),
        f"{SIMDATA}/snp_summary.csv"


rule all_xmls:
    input:
        expand(
            f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.xml",
            model=INFERENCE, sampling=SAMPLING_TYPES, popmodel=POP_MODELS, mutsig=MUTSIGS, i=range(NREPLICATES)
        )


# =============================================================================
# Step 1: Simulate trees
# Runs ReMASTER (BEAST2) on the template for one (sampling, popmodel) scenario.
# The template placeholders {$nreplicates} and {$filebase} are filled in
# with sed. Produces NREPLICATES trees in one file (converted from NEXUS to
# one plain newick tree per line; line i+1 = replicate i) and the population
# trajectory (.traj).
# =============================================================================
rule simulate_trees:
    input:
        template = f"{TEMPLATES_DIR}/remaster_{{sampling}}_{{popmodel}}.xml",
    output:
        trees = f"{SIMDATA}/{{sampling}}/{{popmodel}}/{{popmodel}}.trees",
        traj  = f"{SIMDATA}/{{sampling}}/{{popmodel}}/{{popmodel}}.traj",
    log:
        "logs/simulate_trees/{sampling}_{popmodel}.log"
    params:
        filebase    = SIMDATA,
        nreplicates = NREPLICATES,
        seed        = REMASTER_SEED,
    envmodules:
        "stack/2024-06",
        "gcc/12.2.0",
        "openjdk/21.0.3_9",
        "beast2/2.7.4",
    resources:
        mem_mb_per_cpu = 4000,
        runtime = 20,
        cpus_per_task = 1,
    shell:
        """
        mkdir -p $(dirname {output.trees})
        TMPXML=$(mktemp --suffix=.xml)
        sed 's|{{$nreplicates}}|{params.nreplicates}|g; s|{{$filebase}}|{params.filebase}|g' {input.template} > $TMPXML
        beast -overwrite -seed {params.seed} $TMPXML > {log} 2>&1
        rm $TMPXML
        # Convert NEXUS to one plain newick per line (strip beast annotations, keep only tree lines)
        NEXUS={output.trees}
        TMPNWK=$(mktemp)
        grep "^tree STATE_" "$NEXUS" \
            | sed 's/^tree STATE_[0-9]* = //' \
            | sed 's/\\[&[^]]*\\]//g' \
            > "$TMPNWK"
        mv "$TMPNWK" "$NEXUS"
        """


# =============================================================================
# Step 2a: Simulate alignment with seq-gen and extract SNPs with snp-sites
#         Full fasta is deleted immediately after SNP extraction to save space
# One job per replicate tree i and mutsig: HKY (kappa=0.5, equal base
# frequencies), substitution rate 4.6e-8 per site and time unit, alignment
# length from MUTSIG_LENGTHS. Requires the conda env "snp_sites".
# =============================================================================
rule simulate_alignment:
    input:
        trees = f"{SIMDATA}/{{sampling}}/{{popmodel}}/{{popmodel}}.trees",
    output:
        snps = temp(f"{SIMDATA}/{{sampling}}/{{popmodel}}/{{popmodel}}_{{i}}_{{mutsig}}_snps.fasta"),
    params:
        length = lambda wc: MUTSIG_LENGTHS[wc.mutsig]
    log:
        "logs/simulate_alignment/{sampling}_{popmodel}_{i}_{mutsig}.log"
    resources:
        runtime = lambda wildcards: 60 if wildcards.sampling == "independenthomochronous" else 180,
        cpus_per_task = 1,
        mem_mb_per_cpu = 8000,
    shell:
        """
        TREE=$(sed -n "$(( {wildcards.i} + 1 ))p" {input.trees})
        TMPNWK=$(mktemp --suffix=.nwk)
        TMPFASTA=$(mktemp --suffix=.fasta)
        echo -e "$TREE\n" > $TMPNWK
        {SEQGEN} -mHKY -t0.5 -f0.25,0.25,0.25,0.25 -l{params.length} -s4.6e-8 -n1 \
            < $TMPNWK > $TMPFASTA 2>> {log}
        rm $TMPNWK
        conda run -n snp_sites snp-sites -o {output.snps} $TMPFASTA >> {log} 2>&1
        rm $TMPFASTA
        """

# =============================================================================
# Step 2b: Summarize number of SNPs in each alignment and save to CSV
# Not required by rule all; build via the all_alignments target.
# =============================================================================
rule snp_summary:
    input:
        expand(
            f"{SIMDATA}/{{sampling}}/{{popmodel}}/{{popmodel}}_{{i}}_{{mutsig}}_snps.fasta",
            sampling=SAMPLING_TYPES, popmodel=POP_MODELS, i=range(NREPLICATES), mutsig=MUTSIGS
        )
    output:
        f"{SIMDATA}/snp_summary.csv"
    log:
        "logs/snp_summary/snp_summary.log"
    resources:
        mem_mb_per_cpu = 2000,
        runtime = 10,
        cpus_per_task = 1,
    run:
        import os
        rows = []
        for f in input:
            parts = f.split("/")
            sampling = parts[-3]
            popmodel = parts[-2]
            basename = os.path.basename(f).replace("_snps.fasta", "")
            # filename: {popmodel}_{i}_{mutsig}_snps.fasta
            basename_parts = basename.split("_")
            mutsig = basename_parts[-1]
            i = basename_parts[-2]
            with open(f) as fh:
                for line in fh:
                    if not line.startswith(">"):
                        n_snps = len(line.strip())
                        break
            rows.append(f"{sampling},{popmodel},{i},{mutsig},{n_snps}")
        with open(output[0], "w") as out:
            out.write("sampling,popmodel,replicate,mutsig,n_snps\n")
            out.write("\n".join(rows) + "\n")


# =============================================================================
# Step 3: Generate BEAST XML files
# One job per (model, sampling, popmodel, mutsig) scenario, writing one XML per
# replicate tree. scripts/MakeBEASTXML.py reads the scenario config file
# (CONFIG_DIR) which specifies the BEAST template, the simulated trees file,
# mutsig, clock rate, alignment length and output path. The clock rate is
# rescaled per replicate to the number of SNP sites in its alignment.
# Requires the conda env "beast_tools" (and scripts/beastutils.py).
# =============================================================================
rule make_beast_xml:
    input:
        snps   = expand(
                     f"{SIMDATA}/{{{{sampling}}}}/{{{{popmodel}}}}/{{{{popmodel}}}}_{{i}}_{{{{mutsig}}}}_snps.fasta",
                     i=range(NREPLICATES)
                 ),
        trees  = f"{SIMDATA}/{{sampling}}/{{popmodel}}/{{popmodel}}.trees",
        config = config_file,
    output:
        expand(
            f"{BEAST_DIR}/{{{{model}}}}/{{{{sampling}}}}/{{{{popmodel}}}}/{{{{mutsig}}}}/{{{{model}}}}_{{{{sampling}}}}_{{{{popmodel}}}}_{{{{mutsig}}}}.T{{i}}.xml",
            i=range(NREPLICATES)
        )
    log:
        "logs/make_beast_xml/{model}_{sampling}_{popmodel}_{mutsig}.log"
    resources:
        mem_mb_per_cpu = 2000,
        runtime = 5, # 5 min per scenario (100 xml files per scenario)
        cpus_per_task = 1,
    shell:
        "conda run -n beast_tools python scripts/MakeBEASTXML.py -c {input.config} > {log} 2>&1"


# =============================================================================
# Step 4: Run BEAST (one job per XML file per seed)
# BEAST 1.10.4 is run inside the seed-specific output directory. It is stopped
# by an internal timeout shortly before the SLURM time limit; a run that times
# out (or fails) is still accepted if non-empty .log and .trees exist, so
# convergence is checked later by the ESS check in step 6.
# Outputs are temp() and are deleted after combine_runs has used them.
# =============================================================================
rule run_beast:
    input:
        xml = f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.xml",
    output:
        log   = temp(f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/seed{{seed}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.log"),
        trees = temp(f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/seed{{seed}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.trees"),
    log:
        "logs/run_beast/{model}_{sampling}_{popmodel}_{mutsig}_T{i}_seed{seed}.log"
    envmodules:
        "stack/2024-06",
        "openjdk/21.0.3_9",
        "gcc/12.2.0",
        "beast1/1.10.4",
        "libbeagle/3.1.2",
    resources:
        mem_mb_per_cpu = 1000,
        runtime = lambda wildcards, attempt: (240 if wildcards.model == "constcoal" else 960),
        cpus_per_task = 2,
        slurm_jobname = lambda wildcards: f"{wildcards.model}_{wildcards.sampling}_{wildcards.popmodel}_{wildcards.mutsig}_T{wildcards.i}_s{wildcards.seed}",
    shell:
        r"""
        set +e

        mkdir -p $(dirname {output.log})

        # Internal timeout must be smaller than SLURM runtime.
        if [ "{wildcards.model}" = "constcoal" ]; then
            internal_timeout="238m"
        else
            internal_timeout="958m"
        fi

        cd $(dirname {output.log})

        timeout --preserve-status "$internal_timeout" \
            beast -overwrite -seed {wildcards.seed} "$OLDPWD/{input.xml}" \
            > "$OLDPWD/{log}" 2>&1

        exitcode=$?

        cd "$OLDPWD"

        if [ "$exitcode" -eq 0 ]; then
            exit 0
        fi

        if [ -s "{output.log}" ] && [ -s "{output.trees}" ]; then
            echo "BEAST exited with code $exitcode, but partial outputs exist. Accepting run." >> {log}
            exit 0
        fi

        echo "BEAST failed with code $exitcode and required outputs are missing or empty." >> {log}
        exit "$exitcode"
        """

# =============================================================================
# Step 5: Combine BEAST runs with logcombiner
# Merges the runs of all SEEDS for one replicate, removing get_burnin() states
# from the start of each run.
# =============================================================================
rule combine_runs:
    input:
        logs  = lambda wc: expand(
                    f"{BEAST_DIR}/{wc.model}/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/seed{{seed}}/{wc.model}_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{wc.i}.log",
                    seed=SEEDS
                ),
        trees = lambda wc: expand(
                    f"{BEAST_DIR}/{wc.model}/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/seed{{seed}}/{wc.model}_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{wc.i}.trees",
                    seed=SEEDS
                ),
    output:
        log   = f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined.log",
        trees = f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined.trees",
    log:
        "logs/combine_runs/{model}_{sampling}_{popmodel}_{mutsig}_T{i}.log"
    envmodules:
        "stack/2024-06",
        "openjdk/21.0.3_9",
        "gcc/12.2.0",
        "beast1/1.10.4",
        "libbeagle/3.1.2",
    params:
        burnin = get_burnin,
    resources:
        mem_mb_per_cpu = 2000,
        runtime = 60,
        cpus_per_task = 1,
    shell:
        """
        mkdir -p $(dirname {output.log})
        logcombiner -burnin {params.burnin} {input.logs} {output.log} > {log} 2>&1
        logcombiner -trees -burnin {params.burnin} {input.trees} {output.trees} >> {log} 2>&1
        """


# =============================================================================
# Step 6: Check ESS and create summary tree in one step.
# If ESS passes, treeannotator is run; otherwise an empty file is created.
# ESS threshold is 200 for all logged parameters (R packages coda + beastio).
# An empty summary tree marks a failed replicate for all downstream rules.
# =============================================================================
rule make_summary_tree:
    input:
        log   = f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined.log",
        trees = f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined.trees",
    output:
        f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined_summary.tree",
    log:
        "logs/make_summary_tree/{model}_{sampling}_{popmodel}_{mutsig}_T{i}.log"
    envmodules:
        "stack/2024-06",
        "r/4.4.0",
        "openjdk/21.0.3_9",
        "gcc/12.2.0",
        "beast1/1.10.4",
        "libbeagle/3.1.2",
    resources:
        mem_mb_per_cpu = 4000,
        runtime = 75,
        cpus_per_task = 1,
    shell:
        """
        Rscript -e "
          library(coda); library(beastio)
          mcmc <- readLog('{input.log}', burnin = 0)
          low <- checkESS(mcmc, cutoff = 200, value = TRUE)
          if (length(low) > 0) stop('ESS check failed')
        " >> {log} 2>&1 && \
        treeannotator -burnin 0 -heights median {input.trees} {output} >> {log} 2>&1 || \
        {{ echo "ESS check failed" >> {log}; touch {output}; }}
        """


# =============================================================================
# Step 7: Collect all successful summary trees into a CSV
# Writes the combined.trees path of every run with a non-empty summary tree.
# =============================================================================
rule build_csv:
    input:
        expand(
            f"{BEAST_DIR}/{{model}}/{{sampling}}/{{popmodel}}/{{mutsig}}/{{model}}_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined_summary.tree",
            model=INFERENCE, sampling=SAMPLING_TYPES, popmodel=POP_MODELS, mutsig=MUTSIGS, i=range(NREPLICATES)
        )
    output:
        csv = f"{BEAST_DIR}/successful_mcmc_runs.csv",
    run:
        import os
        rows = []
        for f in input:
            if os.path.getsize(f) > 0:
                trees_path = f.replace(".combined_summary.tree", ".combined.trees")
                rows.append(f"{trees_path}")
        with open(output.csv, "w") as out:
            out.write("trees_path\n")
            out.write("\n".join(rows) + "\n")


# =============================================================================
# Step 8: Process results into per-scenario DataFrames
# scripts/process_results.py writes EVAL_DIR/{sampling}/{popmodel}/{low,med,high}.tsv,
# one row per replicate that passed the ESS check for BOTH constcoal and
# skyline, plus successful_counts.tsv with the number of such replicates.
# Runs with the "python" on PATH (needs pandas), not in a conda env.
# =============================================================================
rule process_results:
    input:
        csv = f"{BEAST_DIR}/successful_mcmc_runs.csv",
    output:
        summary = f"{BEAST_DIR.replace('beast_inference', 'evaluation')}/successful_counts.tsv",
        tsvs    = expand(
            f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{mutsig_short}}.tsv",
            sampling=SAMPLING_TYPES, popmodel=POP_MODELS,
            mutsig_short=["low", "med", "high"]
        ),
    log:
        "logs/process_results/process_results.log"
    resources:
        mem_mb_per_cpu = 4000,
        runtime = 10,
        cpus_per_task = 1,
    shell:
        """
        python scripts/process_results.py \
            --beast_dir {BEAST_DIR} \
            > {log} 2>&1
        """


# =============================================================================
# Step 9: Subsample combined log and trees to ~1000 samples with logcombiner
# Only runs for runs that passed ESS (non-empty summary tree); otherwise
# touches empty output files so Snakemake considers the rule satisfied.
# Handles constcoal and skyline of one replicate together, since a replicate
# is only evaluated if both models passed.
# =============================================================================
rule subsample_runs:
    input:
        log_constcoal     = f"{BEAST_DIR}/constcoal/{{sampling}}/{{popmodel}}/{{mutsig}}/constcoal_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined.log",
        trees_constcoal   = f"{BEAST_DIR}/constcoal/{{sampling}}/{{popmodel}}/{{mutsig}}/constcoal_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined.trees",
        summary_constcoal = f"{BEAST_DIR}/constcoal/{{sampling}}/{{popmodel}}/{{mutsig}}/constcoal_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined_summary.tree",
        log_skyline       = f"{BEAST_DIR}/skyline/{{sampling}}/{{popmodel}}/{{mutsig}}/skyline_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined.log",
        trees_skyline     = f"{BEAST_DIR}/skyline/{{sampling}}/{{popmodel}}/{{mutsig}}/skyline_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined.trees",
        summary_skyline   = f"{BEAST_DIR}/skyline/{{sampling}}/{{popmodel}}/{{mutsig}}/skyline_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.combined_summary.tree",
        tsv               = lambda wc: f"{EVAL_DIR}/{wc.sampling}/{wc.popmodel}/{MUTSIG_SHORT[wc.mutsig]}.tsv",
    output:
        log_constcoal   = f"{BEAST_DIR}/constcoal/{{sampling}}/{{popmodel}}/{{mutsig}}/constcoal_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.subsampled.log",
        trees_constcoal = f"{BEAST_DIR}/constcoal/{{sampling}}/{{popmodel}}/{{mutsig}}/constcoal_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.subsampled.trees",
        log_skyline     = f"{BEAST_DIR}/skyline/{{sampling}}/{{popmodel}}/{{mutsig}}/skyline_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.subsampled.log",
        trees_skyline   = f"{BEAST_DIR}/skyline/{{sampling}}/{{popmodel}}/{{mutsig}}/skyline_{{sampling}}_{{popmodel}}_{{mutsig}}.T{{i}}.subsampled.trees",
    log:
        "logs/subsample_runs/{sampling}_{popmodel}_{mutsig}_T{i}.log"
    envmodules:
        "stack/2024-06",
        "openjdk/21.0.3_9",
        "gcc/12.2.0",
        "beast1/1.10.4",
        "libbeagle/3.1.2",
    resources:
        mem_mb_per_cpu = 2000,
        runtime = 120,
        cpus_per_task = 1,
    shell:
        """
        if [ ! -s "{input.summary_constcoal}" ] || [ ! -s "{input.summary_skyline}" ]; then
            echo "ESS check failed for one or both models; skipping subsampling." >> {log}
            touch {output.log_constcoal} {output.trees_constcoal} {output.log_skyline} {output.trees_skyline}
        else
            set +u
            subsample_one() {{
                local inlog=$1 intrees=$2 outlog=$3 outtrees=$4
                STATE1=$(grep -Ev "^(#|Sample|State)" "$inlog" | awk 'NR==1{{print $1}}')
                STATE2=$(grep -Ev "^(#|Sample|State)" "$inlog" | awk 'NR==2{{print $1}}')
                N_SAMPLES=$(grep -Evc "^(#|Sample|State)" "$inlog" || true)
                LOG_FREQ=$(( STATE2 - STATE1 ))
                RESAMPLE=$(( ((N_SAMPLES * LOG_FREQ) / 1000 / LOG_FREQ) * LOG_FREQ ))
                if [ "$RESAMPLE" -lt "$LOG_FREQ" ]; then RESAMPLE=$LOG_FREQ; fi
                echo "$(basename $inlog): N_SAMPLES=$N_SAMPLES LOG_FREQ=$LOG_FREQ RESAMPLE=$RESAMPLE" >> {log}
                logcombiner -burnin 0 -resample $RESAMPLE "$inlog"   "$outlog"   >> {log} 2>&1
                logcombiner -trees -burnin 0 -resample $RESAMPLE "$intrees" "$outtrees" >> {log} 2>&1
            }}
            subsample_one {input.log_constcoal} {input.trees_constcoal} {output.log_constcoal} {output.trees_constcoal}
            subsample_one {input.log_skyline}   {input.trees_skyline}   {output.log_skyline}   {output.trees_skyline}
        fi
        """


# =============================================================================
# Step 10: Compute per-node errors across posterior samples for each scenario
# scripts/compute_errors.py compares the subsampled posterior node heights to
# the simulated tree (nodes matched by preorder position, topology is fixed)
# and summarises inferred vs. true population size trajectories.
# Requires the conda env "beast_tools".
# =============================================================================
rule compute_errors:
    input:
        tsv = lambda wc: f"{EVAL_DIR}/{wc.sampling}/{wc.popmodel}/{MUTSIG_SHORT[wc.mutsig]}.tsv",
        subsampled_constcoal_logs   = lambda wc: expand(f"{BEAST_DIR}/constcoal/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/constcoal_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{{i}}.subsampled.log",   i=range(NREPLICATES)),
        subsampled_constcoal_trees  = lambda wc: expand(f"{BEAST_DIR}/constcoal/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/constcoal_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{{i}}.subsampled.trees",  i=range(NREPLICATES)),
        subsampled_skyline_logs     = lambda wc: expand(f"{BEAST_DIR}/skyline/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/skyline_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{{i}}.subsampled.log",     i=range(NREPLICATES)),
        subsampled_skyline_trees    = lambda wc: expand(f"{BEAST_DIR}/skyline/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/skyline_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{{i}}.subsampled.trees",    i=range(NREPLICATES)),
    output:
        tsv  = f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{sampling}}_{{popmodel}}_{{mutsig}}_node_errors.tsv",
        pkl  = f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{sampling}}_{{popmodel}}_{{mutsig}}_pop_summary.pkl",
        traj = f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{sampling}}_{{popmodel}}_{{mutsig}}_pop_trajectory.tsv",
    log:
        "logs/compute_errors/{sampling}_{popmodel}_{mutsig}.log"
    resources:
        mem_mb_per_cpu = 8000,
        runtime = 240,
        cpus_per_task = 1,
    shell:
        """
        conda run -n beast_tools python -u scripts/compute_errors.py \
            --tsv {input.tsv} \
            --out_dir $(dirname {output.tsv}) \
            --traj_points 1000 \
            > {log} 2>&1
        """


# =============================================================================
# Step 11: Compute time-binned summaries (median + HPD) over all posterior
#          samples × replicates for each scenario / bin-width / cutoff combo
# Nodes are binned by their true height; combos come from TIME_BIN_CONFIGS.
# Requires the conda env "beast_tools".
# =============================================================================
rule compute_time_bins:
    input:
        tsv = lambda wc: f"{EVAL_DIR}/{wc.sampling}/{wc.popmodel}/{MUTSIG_SHORT[wc.mutsig]}.tsv",
        subsampled_constcoal_logs   = lambda wc: expand(f"{BEAST_DIR}/constcoal/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/constcoal_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{{i}}.subsampled.log",   i=range(NREPLICATES)),
        subsampled_constcoal_trees  = lambda wc: expand(f"{BEAST_DIR}/constcoal/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/constcoal_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{{i}}.subsampled.trees",  i=range(NREPLICATES)),
        subsampled_skyline_logs     = lambda wc: expand(f"{BEAST_DIR}/skyline/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/skyline_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{{i}}.subsampled.log",     i=range(NREPLICATES)),
        subsampled_skyline_trees    = lambda wc: expand(f"{BEAST_DIR}/skyline/{wc.sampling}/{wc.popmodel}/{wc.mutsig}/skyline_{wc.sampling}_{wc.popmodel}_{wc.mutsig}.T{{i}}.subsampled.trees",    i=range(NREPLICATES)),
    output:
        tsv               = f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{sampling}}_{{popmodel}}_{{mutsig}}_time_bins_w{{binw}}_c{{cutoff}}.tsv",
        per_tree_tsv      = f"{EVAL_DIR}/{{sampling}}/{{popmodel}}/{{sampling}}_{{popmodel}}_{{mutsig}}_time_bins_w{{binw}}_c{{cutoff}}_per_tree.tsv",
    wildcard_constraints:
        binw   = "|".join(sorted({str(int(bw)) for bw, co in _ALL_CONFIGS}, key=lambda x: -len(x))),
        cutoff = "|".join(sorted({str(int(co)) for bw, co in _ALL_CONFIGS}, key=lambda x: -len(x))),
    log:
        "logs/compute_time_bins/{sampling}_{popmodel}_{mutsig}_w{binw}_c{cutoff}.log"
    resources:
        mem_mb_per_cpu = 16000,
        runtime = 360,
        cpus_per_task = 1,
    shell:
        """
        conda run -n beast_tools python -u scripts/compute_time_bins.py \
            --tsv {input.tsv} \
            --out_dir $(dirname {output.tsv}) \
            --config {wildcards.binw},{wildcards.cutoff} \
            > {log} 2>&1
        """
