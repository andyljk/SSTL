#' Fit an SSTL regression model with optional empirical Bayes estimation
#'
#' Selects target-only, transfer learning, or grouped transfer learning from the
#' supplied data. When EB is enabled, estimates thresholds first and continues
#' from the fitted state with fixed thresholds in the main MCMC run. Both stages
#' run in compiled code.
#'
#' @param X,Y Target design matrix and response. For AFT families, supply
#'   observed survival times when log is TRUE, or logged times when log is FALSE.
#' @param X_s,Y_s Lists of source design matrices and responses. Empty lists
#'   select target-only regression. Predictor columns must match across studies.
#' @param group_map Optional consecutive integer group labels starting at 1,
#'   one per predictor. Supplying this selects grouped transfer learning.
#' @param family Outcome family: 'Gaussian', 'Logistic' (binary 0/1),
#'   'Student-t', 'Poisson' or 'Negative-binomial' (nonnegative integer counts,
#'   log link), 'Gamma' (positive response, log link), 'Beta' (response strictly
#'   between 0 and 1, logit link), 'Weibull', 'Lognormal', or 'Loglogistic'.
#'   Case-insensitive. Negative-binomial, gamma, and beta are not supported for
#'   grouped models.
#' @param EB Whether to estimate thresholds by empirical Bayes before MCMC.
#' @param N Number of main MCMC iterations, including any warmup the user discards.
#' @param burn Number of initial main MCMC iterations excluded from posterior
#'   summaries; defaults to half of N. All draws remain in the sampler output.
#' @param lambda Target threshold or its initial EB value; defaults to sqrt(p).
#' @param lambda_s Source thresholds or their initial EB values; defaults to 3
#'   per source.
#' @param C,C_s AFT observation indicators (1 = event, 0 = right-censored) for
#'   target and sources. NULL treats all observations as uncensored.
#' @param log If TRUE (default), log-transform target and source survival times.
#'   If FALSE, use the supplied logged times. Ignored for general outcome families.
#' @param intercept Whether to estimate an intercept in each study.
#' @param slab Slab transformation, one of 'exp', 'poly', 'nlp', 'nlp2', or
#'   'guassian'; passed to both EB and MCMC.
#' @param df Fixed positive degrees of freedom for Student-t outcomes.
#' @param EB_control Named list of EB tuning arguments. All models accept N
#'   (default 5000), burn (1000), S.max (50; 500 for grouped models), block_size
#'   (1), K_block (iterations per SAEM update, 10), gamma_power (Robbins-Monro
#'   step exponent, 0.9), lr (learning rate, 0.1), schedule (learning-rate decay
#'   exponent, 0.5), lambda_min, lambda_max (1e4), and polyak_start (0.9).
#'   Target-only models also accept optimizer ('legacy' or 'adagrad'),
#'   adagrad_eps (1e-8), and max_log_step (0.35), with lambda_min 1e-6. Transfer
#'   models also accept polyak (TRUE) and warm_start (TRUE: ten MCEM iterations of
#'   300 draws before SAEM), with lambda_min 1e-3. S.max and block_size given in
#'   ... also apply to EB unless set here.
#' @param verbose If 1, print stage labels and the EB/MCMC progress bars;
#'   if 0, run quietly.
#' @param ... Additional sampler arguments:
#'   \describe{
#'     \item{S.max}{Maximum slice iterations per update (100 for target-only,
#'       500 for transfer models).}
#'     \item{block_size}{Number of coefficients updated jointly (1).}
#'     \item{debug}{If TRUE, also return latent draws, slab scales, and slice
#'       iteration counts.}
#'     \item{b.c, sd.0, xi, sd_y, b0}{Target-only initial latent vector, prior
#'       standard deviations of the latent parameters, slab-scale shadow
#'       variable (1), outcome scale (1), and intercept (0).}
#'     \item{bt.c, bs.c, xi, xi_s, sig_T, sig_s, b0_T, b0_s}{Transfer-model initial
#'       target and source latent parameters, slab-scale shadow variables (0.5),
#'       outcome scales (1), and intercepts (0).}
#'     \item{xi_prior}{Prior standard deviation of the target slab-scale shadow
#'       variable in non-grouped transfer models (1).}
#'   }
#'   For 'Gaussian' and 'Student-t' the outcome scale is the standard deviation
#'   or scale; for 'Negative-binomial' and 'Gamma' it is the shape, and for
#'   'Beta' the precision. It is fixed at 1 for 'Logistic' and 'Poisson'. EB's
#'   final state replaces the initial MCMC states when EB is TRUE.
#' @details Shape and precision parameters are updated by log-scale
#'   Metropolis-Hastings with proposal standard deviation 0.3 (0.1 for other
#'   scales and for grouped models). The negative-binomial shape has an
#'   inverse-gamma prior with shape 0.4 and scale 0.3; gamma shape and beta
#'   precision have Gamma(shape = 0.01, rate = 0.01) priors.
#' @return A list with post_mean, post_median, post_sd, post_interval (95\% equal-tail
#'   intervals), and pip (posterior probabilities of nonzero target coefficients).
#'   Summaries exclude burn iterations. All sampler output is retained, including
#'   MC_beta and optional debug output. Additional fields are lambda
#'   (target/source thresholds), EB (the EB result or NULL), and burn.
#' @export
SSTL <- function(X, Y, X_s=list(), Y_s=list(), group_map=NULL,
                 family="Gaussian", EB=FALSE, N=5000, burn=floor(N / 2),
                 lambda=NULL, lambda_s=NULL, C=NULL, C_s=NULL, log=TRUE,
                 intercept=TRUE, slab="exp", df=4,
                 EB_control=list(), verbose=1, ...) {
  X <- as_double_matrix(X)
  Y <- as.numeric(Y)
  X_s <- lapply(X_s, as_double_matrix)
  Y_s <- lapply(Y_s, as.numeric)
  S <- length(X_s)
  p <- ncol(X)
  if (length(Y_s) != S) stop("X_s and Y_s must contain the same number of studies.")
  if (nrow(X) != length(Y) || any(vapply(X_s, nrow, numeric(1)) != lengths(Y_s))) {
    stop("Each design matrix must have one row per response.")
  }
  if (any(vapply(X_s, ncol, numeric(1)) != p)) stop("Source design matrices must have the same columns as X.")
  if (length(N) != 1 || !is.finite(N) || N < 1 || N != floor(N)) stop("N must be a positive integer.")
  if (length(burn) != 1 || !is.finite(burn) || burn < 0 || burn != floor(burn) || burn >= N) {
    stop("burn must be a nonnegative integer smaller than N.")
  }
  if (is.null(lambda)) lambda <- sqrt(p)
  if (is.null(lambda_s)) lambda_s <- rep(3, S)
  if (length(lambda_s) != S) stop("lambda_s needs one value per source.")

  grouped <- !is.null(group_map)
  transfer <- S > 0 || grouped
  aft <- tolower(family) %in% c("weibull", "lognormal", "loglogistic")
  model <- if (grouped) "Group TL" else if (transfer) "TL" else "TO"

  fam_map <- if (aft) c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
             else c("gaussian" = 1, "logistic" = 2, "student-t" = 3, "poisson" = 4,
                    "negative-binomial" = 5, "gamma" = 6, "beta" = 7)
  if (grouped && !aft) fam_map <- fam_map[1:4]
  fam_code <- unname(fam_map[tolower(family)])
  if (is.na(fam_code)) {
    stop("family must be one of ", paste0("'", c(names(fam_map), if (!aft) c("weibull", "lognormal", "loglogistic")),
                                          "'", collapse = ", "), " for this model.")
  }
  slab_code <- unname(c("exp" = 1, "poly" = 2, "nlp" = 3, "nlp2" = 4, "guassian" = 5)[tolower(slab)])
  if (is.na(slab_code)) stop("slab must be 'exp', 'poly', 'nlp', 'nlp2', or 'guassian'.")

  if (aft) {
    if (log) {
      if (any(!is.finite(Y) | Y <= 0) || any(vapply(Y_s, function(y) any(!is.finite(y) | y <= 0), logical(1)))) {
        stop("Survival times must be positive and finite when log = TRUE.")
      }
      Y <- base::log(Y)
      Y_s <- lapply(Y_s, base::log)
    }
    if (is.null(C)) C <- rep(1, length(Y))
    if (is.null(C_s)) C_s <- lapply(Y_s, function(y) rep(1, length(y)))
    if (length(C) != length(Y) || any(!C %in% c(0, 1)) ||
        length(C_s) != S || any(lengths(C_s) != lengths(Y_s)) ||
        any(vapply(C_s, function(c) any(!c %in% c(0, 1)), logical(1)))) {
      stop("C and C_s must contain 0/1 indicators matching the target and source responses.")
    }
    C <- as.numeric(C)
    C_s <- lapply(C_s, as.numeric)
    df <- 4
  } else {
    if (!is.null(C) || !is.null(C_s)) stop("C and C_s are only used for AFT families.")
    C <- numeric(0)
    C_s <- list()
    check_glm_response(fam_code, c(list(Y), Y_s))
    if (fam_code == 3 && (length(df) != 1 || !is.finite(df) || df <= 0)) {
      stop("df must be a positive number for 'Student-t'.")
    }
  }
  if (grouped) {
    group_map <- as.integer(group_map)
    validate_group_map(group_map, p)
  }

  # ---- sampler options passed through ...
  opts <- list(...)
  allowed <- c("S.max", "block_size", "debug",
               if (transfer) c("bt.c", "bs.c", "xi", "xi_s", "sig_T", "sig_s", "b0_T", "b0_s", if (!grouped) "xi_prior")
               else c("b.c", "sd.0", "xi", "sd_y", "b0"))
  if (length(opts) && (is.null(names(opts)) || any(!names(opts) %in% allowed))) {
    bad <- if (is.null(names(opts))) "(unnamed)" else setdiff(names(opts), allowed)
    stop("Unknown sampler argument(s) for the ", model, " model: ", paste(bad, collapse = ", "),
         ". Allowed: ", paste(allowed, collapse = ", "), ".")
  }
  opt <- function(name, default) if (is.null(opts[[name]])) default else opts[[name]]
  no_scale <- !aft && fam_code %in% c(2, 4)
  mcmc <- list(N = N, S.max = opt("S.max", if (transfer) 500 else 100),
               block_size = opt("block_size", 1), debug = isTRUE(opt("debug", FALSE)),
               xi_prior = opt("xi_prior", 1))

  # ---- EB settings
  eb <- NULL
  if (EB) {
    eb <- list(N = 5000, burn = 1000, S.max = if (grouped) 500 else 50, block_size = 1,
               K_block = 10, gamma_power = 0.9, lr = 0.1, schedule = 0.5,
               lambda_max = 1e4, polyak_start = 0.9)
    eb <- c(eb, if (transfer) list(lambda_min = 1e-3, polyak = TRUE, warm_start = TRUE)
                else list(lambda_min = 1e-6, optimizer = "legacy", adagrad_eps = 1e-8, max_log_step = 0.35))
    for (nm in c("S.max", "block_size")) if (!is.null(opts[[nm]])) eb[[nm]] <- opts[[nm]]
    if (length(EB_control) && (is.null(names(EB_control)) || any(!names(EB_control) %in% names(eb)))) {
      stop("EB_control must be a named list of tuning arguments: ", paste(names(eb), collapse = ", "), ".")
    }
    eb <- utils::modifyList(eb, EB_control)
    if (!transfer) eb$optimizer <- match.arg(eb$optimizer, c("legacy", "adagrad"))
    if (transfer && (!is.logical(eb$warm_start) || length(eb$warm_start) != 1 || is.na(eb$warm_start))) {
      stop("warm_start must be TRUE or FALSE.")
    }
    for (nm in c("N", "burn", "K_block")) {
      if (length(eb[[nm]]) != 1 || !is.finite(eb[[nm]]) || eb[[nm]] < 1 || eb[[nm]] != floor(eb[[nm]])) {
        stop("EB_control$", nm, " must be a positive integer.")
      }
    }
  }

  # ---- initial states, drawn in the order of the former samplers
  if (transfer) {
    G <- if (grouped) max(group_map) else p
    d <- p + G + 1
    draw_init_tl <- function(user) {
      bt <- if (is.null(user$bt.c)) rnorm(d, 0, 1) else as.numeric(user$bt.c)
      bs <- if (is.null(user$bs.c)) matrix(rnorm(d * S, 0, c(rep(0.05, p), rep(1, G + 1))), nrow = d, ncol = S)
            else matrix(as.numeric(user$bs.c), nrow = d, ncol = S)
      st <- list(bt.c = bt, bs.c = bs,
                 xi = if (is.null(user$xi)) 0.5 else user$xi,
                 xi_s = if (is.null(user$xi_s)) rep(0.5, S) else user$xi_s,
                 sig_T = if (no_scale || is.null(user$sig_T)) 1 else user$sig_T,
                 sig_s = if (no_scale || is.null(user$sig_s)) rep(1, S) else user$sig_s,
                 b0_T = if (is.null(user$b0_T)) 0 else user$b0_T,
                 b0_s = if (is.null(user$b0_s)) rep(0, S) else user$b0_s)
      if (length(st$bt.c) != d || length(st$xi_s) != S || length(st$sig_s) != S || length(st$b0_s) != S) {
        stop("Initial states have the wrong length for this model.")
      }
      if (!aft && fam_code %in% 5:7 && (!is.finite(st$sig_T) || st$sig_T <= 0 || any(!is.finite(st$sig_s) | st$sig_s <= 0))) {
        stop("sig_T and each element of sig_s must be positive; sig_s needs one value per source.")
      }
      st
    }
    init <- if (EB) {
      # EB draws its own starting values; user intercepts are used by its burn-in.
      eb$init <- draw_init_tl(opts[c("b0_T", "b0_s")])
      list(b0_T = opt("b0_T", 0), b0_s = opt("b0_s", rep(0, S)))
    } else draw_init_tl(opts)
    fit <- sstl_fit_tl(X, Y, if (aft) C else numeric(0), X_s, Y_s, C_s,
                       if (grouped) group_map else NULL,
                       fam_code, aft, df, slab_code, intercept, lambda, as.numeric(lambda_s),
                       init, mcmc, eb, paste0(model, ", ", family), verbose, getOption("width"))
  } else {
    sd0 <- opt("sd.0", rep(1, 2 * p + 1))
    draw_init_to <- function(user, sd0) {
      st <- list(b.c = if (is.null(user$b.c)) rnorm(2 * p + 1, 0, sd0) else as.numeric(user$b.c),
                 xi = if (is.null(user$xi)) 1 else user$xi,
                 sd_y = if (no_scale || is.null(user$sd_y)) 1 else user$sd_y,
                 b0 = if (is.null(user$b0)) 0 else user$b0, sd.0 = sd0)
      if (length(st$b.c) != 2 * p + 1 || length(st$sd.0) != 2 * p + 1) stop("b.c and sd.0 need 2p + 1 values.")
      if (!aft && fam_code %in% 5:7 && (length(st$sd_y) != 1 || !is.finite(st$sd_y) || st$sd_y <= 0)) {
        stop("sd_y must be a positive number for shape/precision.")
      }
      st
    }
    init <- if (EB) {
      eb$init <- draw_init_to(opts["b0"], rep(1, 2 * p + 1))
      list(b0 = opt("b0", 0), sd.0 = sd0)
    } else draw_init_to(opts, sd0)
    fit <- sstl_fit_to(X, Y, C, fam_code, aft, df, slab_code, intercept, lambda,
                       init, mcmc, eb, paste0(model, ", ", family), verbose, getOption("width"))
  }

  eb_fit <- fit$EB
  lam <- fit$lambda
  fit$EB <- NULL
  fit$lambda <- NULL
  names(lam) <- c("target", if (transfer && S > 0) paste0("source", seq_len(S)))
  out <- fit
  out$lambda <- lam
  out["EB"] <- list(eb_fit)
  out$burn <- burn
  beta_draws <- out$MC_beta[seq.int(burn + 1, N), , drop=FALSE]
  colnames(beta_draws) <- if (is.null(colnames(X))) paste0("X", seq_len(p)) else colnames(X)
  c(list(post_mean=colMeans(beta_draws),
         post_median=apply(beta_draws, 2, stats::median),
         post_sd=apply(beta_draws, 2, stats::sd),
         post_interval=t(apply(beta_draws, 2, stats::quantile, probs=c(0.025, 0.975))),
         pip=colMeans(beta_draws != 0)), out)
}

#' @keywords internal
#' @noRd
check_glm_response <- function(fam_code, ys) {
  bad <- function(test) any(vapply(ys, function(y) any(test(y)), logical(1)))
  if (fam_code == 2 && bad(function(y) !y %in% c(0, 1))) {
    stop("Target and source responses must contain only 0 and 1 for 'Logistic'.")
  }
  if (fam_code %in% c(4, 5) && bad(function(y) !is.finite(y) | y < 0 | y != floor(y))) {
    stop("Target and source responses must contain only nonnegative integer counts for 'Poisson' or 'Negative-binomial'.")
  }
  if (fam_code == 6 && bad(function(y) !is.finite(y) | y <= 0)) {
    stop("Target and source responses must contain only positive values for 'Gamma'.")
  }
  if (fam_code == 7 && bad(function(y) !is.finite(y) | y <= 0 | y >= 1)) {
    stop("Target and source responses must contain only values strictly between 0 and 1 for 'Beta'.")
  }
}
