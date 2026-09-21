#' Fit an SSTL regression model with optional empirical Bayes estimation
#'
#' Selects target-only, transfer learning, or grouped transfer learning from the
#' supplied data. When EB is enabled, estimates thresholds first and continues
#' from the fitted state with fixed thresholds in the main MCMC run.
#'
#' @param X,Y Target design matrix and response. For AFT families, supply
#'   observed survival times when log is TRUE, or logged times when log is FALSE.
#' @param X_s,Y_s Lists of source design matrices and responses. Empty lists
#'   select target-only regression. Predictor columns must match across studies.
#' @param group_map Optional consecutive integer group labels starting at 1,
#'   one per predictor. Supplying this selects grouped transfer learning.
#' @param family Outcome family: 'Gaussian', 'Logistic', 'Student-t', 'Poisson',
#'   'Negative-binomial', 'Gamma', 'Beta', 'Weibull', 'Lognormal', or 'Loglogistic'.
#'   Negative-binomial, gamma, and beta are not supported for grouped models.
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
#' @param slab Slab transformation, passed to both EB and MCMC.
#' @param df Fixed degrees of freedom for Student-t outcomes.
#' @param EB_control Named list of EB tuning arguments, such as N, burn, K_block,
#'   lr, or warm_start (TL only). Other defaults follow the selected EB function.
#' @param verbose If 1, print stage labels and the existing EB/MCMC progress bars;
#'   if 0, run quietly.
#' @param ... Additional arguments to the selected MCMC function, such as
#'   S.max, block_size, or debug. Arguments also supported by EB are passed to
#'   that stage. EB's final state replaces initial MCMC states when EB is TRUE.
#' @return A list with post_mean, post_median, post_sd, post_interval (95\% equal-tail
#'   intervals), and pip (posterior probabilities of nonzero target coefficients).
#'   Summaries exclude burn iterations. All fields from the selected sampler are
#'   retained, including MC_beta and optional debug output. Additional fields are
#'   lambda (target/source thresholds), EB (the EB result or NULL), and burn.
#' @export
SSTL <- function(X, Y, X_s=list(), Y_s=list(), group_map=NULL,
                 family="Gaussian", EB=FALSE, N=5000, burn=floor(N / 2),
                 lambda=NULL, lambda_s=NULL, C=NULL, C_s=NULL, log=TRUE,
                 intercept=TRUE, slab="exp", df=4,
                 EB_control=list(), verbose=1, ...) {
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  X_s <- lapply(X_s, as.matrix)
  Y_s <- lapply(Y_s, as.numeric)
  S <- length(X_s)
  if (length(Y_s) != S) stop("X_s and Y_s must contain the same number of studies.")
  if (length(burn) != 1 || !is.finite(burn) || burn < 0 || burn != floor(burn) || burn >= N) {
    stop("burn must be a nonnegative integer smaller than N.")
  }
  if (is.null(lambda)) lambda <- sqrt(ncol(X))
  if (is.null(lambda_s)) lambda_s <- rep(3, S)

  grouped <- !is.null(group_map)
  transfer <- S > 0 || grouped
  aft <- tolower(family) %in% c("weibull", "lognormal", "loglogistic")
  if (aft && log) {
    if (any(!is.finite(Y) | Y <= 0) || any(vapply(Y_s, function(y) any(!is.finite(y) | y <= 0), logical(1)))) {
      stop("Survival times must be positive and finite when log = TRUE.")
    }
    Y <- base::log(Y)
    Y_s <- lapply(Y_s, base::log)
  }
  model <- if (grouped) "Group TL" else if (transfer) "TL" else "TO"
  suffix <- paste0(if (transfer) "_TL" else "",
                   if (grouped) "_Group" else "",
                   if (aft) "_AFT" else "_General")
  mcmc_fun <- get(paste0("ESS_Gibbs", suffix), mode="function")
  eb_fun <- get(paste0("EB_SAEM", suffix), mode="function")

  args <- list(family=family, intercept=intercept, slab=slab, verbose=verbose)
  if (transfer) {
    args <- c(args, list(X_T=X, Y_T=Y, X_s=X_s, Y_s=Y_s,
                         lambda_T=lambda, lambda_s=lambda_s))
    if (grouped) args$group_map <- group_map
  } else {
    args <- c(args, list(X=X, Y=Y, lambda=lambda))
  }
  if (aft) {
    if (is.null(C)) C <- rep(1, length(Y))
    if (is.null(C_s)) C_s <- lapply(Y_s, function(y) rep(1, length(y)))
    if (length(C) != length(Y) || any(!C %in% c(0, 1)) ||
        length(C_s) != S || any(lengths(C_s) != lengths(Y_s)) ||
        any(vapply(C_s, function(c) any(!c %in% c(0, 1)), logical(1)))) {
      stop("C and C_s must contain 0/1 indicators matching the target and source responses.")
    }
    if (transfer) args <- c(args, list(C_T=C, C_s=C_s)) else args$C <- C
  } else {
    if (!is.null(C) || !is.null(C_s)) stop("C and C_s are only used for AFT families.")
    args$df <- df
  }

  mc_args <- c(args, list(N=N), list(...))
  eb_fit <- NULL
  if (EB) {
    eb_args <- mc_args[names(mc_args) %in% names(formals(eb_fun))]
    eb_args$N <- 5000
    if (!transfer) eb_args$lambda_init <- lambda
    tuning <- setdiff(names(formals(eb_fun)),
                      c(names(args), "lambda_init", "bt.c", "bs.c", "xi", "xi_s", "b0", "b0_T", "b0_s"))
    if (length(EB_control) && (is.null(names(EB_control)) || any(!names(EB_control) %in% tuning))) {
      stop("EB_control must be a named list of tuning arguments for the selected EB function.")
    }
    eb_args <- utils::modifyList(eb_args, EB_control)
    if (verbose == 1) cat("Empirical Bayes (", model, ", ", family, ")\n", sep="")
    eb_fit <- do.call(eb_fun, eb_args)
    if (verbose == 1) cat("\n")

    if (transfer) {
      lambda <- eb_fit$final_lam[1]
      lambda_s <- eb_fit$final_lam[-1]
      mc_args$lambda_T <- lambda
      mc_args$lambda_s <- lambda_s
      mc_args$bt.c <- eb_fit$bt_c
      mc_args$bs.c <- eb_fit$bs_c
      mc_args$xi <- eb_fit$xi
      mc_args$xi_s <- eb_fit$xi_s
      mc_args$sig_T <- eb_fit$sig_T
      mc_args$sig_s <- eb_fit$sig_s
      if (intercept) {
        mc_args$b0_T <- eb_fit$b0_T
        mc_args$b0_s <- eb_fit$b0_s
      }
    } else {
      lambda <- eb_fit$lambda_final
      mc_args$lambda <- lambda
      mc_args$b.c <- eb_fit$b_c
      mc_args$xi <- eb_fit$xi
      mc_args$sd_y <- eb_fit$sd_y
      if (intercept) mc_args$b0 <- eb_fit$b0_final
    }
  }

  if (verbose == 1) cat("Main MCMC (", model, ", ", family, ")\n", sep="")
  out <- do.call(mcmc_fun, mc_args)
  if (verbose == 1) cat("\n")
  out$lambda <- c(lambda, if (transfer) lambda_s)
  names(out$lambda) <- c("target", if (transfer && S > 0) paste0("source", seq_len(S)))
  out["EB"] <- list(eb_fit)
  out$burn <- burn
  beta_draws <- out$MC_beta[seq.int(burn + 1, N), , drop=FALSE]
  colnames(beta_draws) <- if (is.null(colnames(X))) paste0("X", seq_len(ncol(X))) else colnames(X)
  c(list(post_mean=colMeans(beta_draws),
         post_median=apply(beta_draws, 2, stats::median),
         post_sd=apply(beta_draws, 2, stats::sd),
         post_interval=t(apply(beta_draws, 2, stats::quantile, probs=c(0.025, 0.975))),
         pip=colMeans(beta_draws != 0)), out)
}
