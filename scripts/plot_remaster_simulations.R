library(ape)
library(phylodyn)
library(bdskytools)

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  if (!is.null(sys.frames()[[1]]$ofile)) return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  return(getwd())
}

script_dir <- get_script_dir()

# Trajectory functions matching REMASTER XML parameters
get_trajectory_remaster <- function(type) {
  switch(type,
    "uniform"       = function(t) rep(1000, length(t)),
    "expgrowthfast" = function(t) pmax(2000 * exp(-0.02 * t), 10),
    "expgrowthslow" = function(t) pmax(2000 * exp(-0.01 * t), 10),
    "bottleneck"    = function(t) {
      Ne        <- rep(1000.0, length(t))
      Ne[t >= 20] <- 100.0
      Ne[t >= 23] <- 1000.0
      Ne
    },
    stop(paste("Unknown trajectory type:", type))
  )
}

scenarios <- data.frame(
  row.names    = c("bottleneck", "expgrowthfast", "expgrowthslow", "uniform"),
  names        = c("Bottleneck", "Fast exponential growth", "Slow exponential growth", "Uniform"),
  maxdensity   = c(0.1, 0.1, 0.1, 0.1),
  maxlineages  = c(260, 260, 260, 260),
  maxtime_homo    = c(100, 200, 300, 500),
  maxtime_hetero  = c(100, 200, 300, 500),
  step_traj    = c(TRUE, FALSE, FALSE, FALSE),
  stringsAsFactors = FALSE
)

scenario_change_times <- list(
  bottleneck    = c(20.0, 23.0),
  expgrowthfast = numeric(0),
  expgrowthslow = numeric(0),
  uniform       = numeric(0)
)

sampling_configs <- list(
  independenthomochronous = "maxtime_homo",
  linearconstant          = "maxtime_hetero"
)

# --- Helper functions (same as simulate_trees.R) ---

getLTT <- function(tree, grid, plot = TRUE, col = pal.dark(cblue, 0.25)) {
  ltt          <- ltt.plot.coords(tree)
  lttinterp    <- approx(ltt[, 1] * -1, ltt[, 2], xout = grid, method = "constant", rule = 2)
  if (plot) lines(ltt[, 1] * -1, ltt[, 2], col = col, type = "S")
  return(lttinterp$y)
}

getHPD <- function(sims, par, plot = TRUE, col = pal.dark(cblue), fill = pal.dark(cblue, 0.5), ...) {
  densities   <- lapply(sims, function(x) density(x[[par]], ...))
  densities_y <- sapply(densities, function(d) d$y)
  par_hpd     <- getMatrixHPD(t(densities_y))
  if (plot) {
    polygon(c(densities[[1]]$x, rev(densities[[1]]$x)),
            c(par_hpd[1, ], rev(par_hpd[3, ])), border = NA, col = fill)
    lines(densities[[1]]$x, par_hpd[2, ], col = col)
  }
  return(par_hpd)
}

plotPointProcess <- function(sim, par, col = pal.dark(cblue)) {
  axis(1, lwd = 0, lwd.ticks = 1)
  abline(h = 0)
  abline(v = sim[[par]], col = col)
  abline(v = max(sim[[par]]), lty = 3)
  points(sim[[par]], rep(1, length(sim[[par]])), pch = 20, col = col)
}

# --- Core extraction ---

extract_times_from_tree <- function(tree) {
  depths         <- node.depth.edgelength(tree)
  ntips          <- length(tree$tip.label)
  tip_depths     <- depths[seq_len(ntips)]
  internal_depths <- depths[seq(ntips + 1, length(depths))]

  present    <- max(tip_depths)
  samp_times <- sort(present - tip_depths)
  coal_times <- sort(present - internal_depths)

  list(samp_times    = samp_times,
       coal_times    = coal_times,
       samp_intensity = NA,
       phylo          = tree)
}

load_remaster_sims <- function(trees_file, traj_name, gridsize = 1000) {
  trees       <- read.tree(trees_file)
  traj_fn     <- get_trajectory_remaster(traj_name)
  change_times <- scenario_change_times[[traj_name]]

  lapply(trees, function(tree) {
    sim      <- extract_times_from_tree(tree)
    maxtime  <- max(sim$coal_times)
    timegrid <- sort(unique(c(seq(0, maxtime, length.out = gridsize), change_times)))
    sim$trajgrid <- cbind(timegrid, traj_fn(timegrid))
    sim$traj     <- traj_fn
    sim
  })
}

# --- Plotting ---

summariseSims_remaster <- function(sims, maxdensity = 0.05, maxlineages = 100, maxtime = NULL, step_traj = FALSE) {

  if (is.null(maxtime) || is.na(maxtime)) maxtime <- max(sims[[1]]$coal_times)

  layout(matrix(c(1, 2, 3, 4, 5, 4, 6, 4), byrow = TRUE, ncol = 2),
         heights = c(0.5, 0.5, 0.2, 0.2))
  par(mar = c(5, 4, 2, 3) + 0.1)

  # Ne trajectory
  tg <- sims[[1]]$trajgrid
  plot(tg[, 1], tg[, 2], type = if (step_traj) "s" else "l", lwd = 2,
       xlim = c(0, maxtime), xlab = "Time", ylab = expression("True N"[e]), las = 1)

  # LTT
  plot(1, type = "n", xlim = c(0, maxtime), ylim = c(0, maxlineages),
       xlab = "Time", ylab = "LTT (95% HPD)", las = 1)
  timegrid <- seq(0, maxtime, length.out = 100)
  ltt_y    <- sapply(sims, function(x) getLTT(x$phylo, timegrid, col = pal.dark(cblue, 0.1)))
  ltt_hpd  <- getMatrixHPD(t(ltt_y))
  lines(timegrid, ltt_hpd[1, ], lty = 2, col = pal.dark(cred))
  lines(timegrid, ltt_hpd[2, ], lty = 1, col = pal.dark(cred))
  lines(timegrid, ltt_hpd[3, ], lty = 2, col = pal.dark(cred))

  # Density of events
  plot(1, type = "n", xlim = c(0, maxtime), ylim = c(0, maxdensity),
       xlab = "Time", ylab = "Density (95% HPD)", las = 1)
  getHPD(sims, par = "samp_times", to = maxtime, bw = 1,
         col = pal.dark(cred), fill = pal.dark(cred, 0.5))
  getHPD(sims, "coal_times", to = maxtime, bw = 1,
         col = pal.dark(cblue), fill = pal.dark(cblue, 0.5))
  legend("topright",
         legend = c("Coalescent events", "Sampling events"),
         border = pal.dark(c(cblue, cred)),
         fill   = pal.dark(c(cblue, cred), 0.5), bty = "n")
  density_top_ndc <- grconvertY(par("usr")[4], "user", "ndc")

  # Example tree
  tree_mar <- c(5, 4, 2, 3) + 0.1
  tree_mar[3] <- tree_mar[3] - 1  # shift plot up to align with top of density plot
  par(mar = tree_mar)
  tree_plot <- ladderize(sims[[1]]$phylo)
  plot(tree_plot, show.tip.label = FALSE,
       edge.width = 0.5, direction = "leftwards")
  axisPhylo(side = 1, backward = TRUE)
  mtext(side = 1, "Time", line = 2, cex = par("cex.lab") * par("cex"))

  # align "Example tree" label: top with top of density plot, right with the
  # rightmost (most recent) coalescent event in the tree
  lastPP     <- get("last_plot.phylo", envir = ape::.PlotPhyloEnv)
  ntip       <- length(tree_plot$tip.label)
  x_right    <- max(lastPP$xx[(ntip + 1):length(lastPP$xx)])
  label_y    <- grconvertY(density_top_ndc, "ndc", "user")
  text(x_right, label_y, "Example tree", adj = c(1, 1), xpd = NA,
       cex = 1.2, font = 1)
  par(mar = c(5, 4, 2, 3) + 0.1)

  # Point processes
  plot(1, type = "n", xlim = c(0, maxtime), ylim = c(0, 1.1),
       xlab = "Sampling events (example)", ylab = "", las = 1, bty = "n", axes = FALSE)
  plotPointProcess(sims[[1]], "samp_times", col = pal.dark(cred, 0.25))

  plot(1, type = "n", xlim = c(0, maxtime), ylim = c(0, 1.1),
       xlab = "Coalescent events (example)", ylab = "", las = 1, bty = "n", axes = FALSE)
  plotPointProcess(sims[[1]], "coal_times", col = pal.dark(cblue, 0.25))
}

# --- Main loop ---

base_dir <- file.path(script_dir, "../results/run1/simulated_data/")

cli_args     <- commandArgs(trailingOnly = TRUE)
filter_pop   <- if (length(cli_args) >= 1) cli_args[1] else NULL
filter_samp  <- if (length(cli_args) >= 2) cli_args[2] else NULL

for (sampling in names(sampling_configs)) {
  if (!is.null(filter_samp) && filter_samp != sampling) next

  maxtime_col <- sampling_configs[[sampling]]

  for (trajname in rownames(scenarios)) {
    if (!is.null(filter_pop) && filter_pop != trajname) next

    trees_file <- file.path(base_dir, sampling, trajname, paste0(trajname, ".trees"))
    if (!file.exists(trees_file)) {
      cat("Skipping (no trees file):", trees_file, "\n")
      next
    }

    cat("Processing", sampling, "/", trajname, "\n")
    sims    <- load_remaster_sims(trees_file, trajname)
    maxtime <- scenarios[trajname, maxtime_col]

    pdf_path <- file.path(base_dir, sampling, trajname, paste0(trajname, ".pdf"))
    pdf(pdf_path, width = 12, height = 10)

    summariseSims_remaster(sims,
                           maxdensity  = scenarios[trajname, "maxdensity"],
                           maxlineages = scenarios[trajname, "maxlineages"],
                           maxtime     = maxtime,
                           step_traj   = scenarios[trajname, "step_traj"])

    nrep  <- length(sims)
    nsamp <- length(sims[[1]]$samp_times)
    samp_range <- round(range(sims[[1]]$samp_times), 1)
    title(main = scenarios[trajname, "names"],
          sub  = paste0("Nr. replicates: ", nrep,
                        ", Nr. samples: ~", nsamp,
                        ", Sampling period: [", samp_range[1], ", ", samp_range[2], "]"),
          outer = TRUE, line = -1.0)

    dev.off()
    cat("  Saved:", pdf_path, "\n")
  }
}
