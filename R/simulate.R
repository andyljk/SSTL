#' Simulate target and source datasets for SSTL
#'
#' Generates a target study and K source studies with the design of the SSTL
#' simulation studies: Gaussian covariates with AR(1) (Toeplitz) correlation,
#' s0 nonzero target coefficients in the first positions, and sources that
#' either share the target coefficients (informative) or keep half of the
#' target signals and gain 2 * s0 new ones (non-informative). AFT outcomes are
#' right-censored by independent exponential censoring times whose rate is
#' chosen to give the requested censoring proportion in each study.
#'
#' @param n_T Target sample size.
#' @param p Number of predictors.
#' @param n_k Source sample sizes: one value used for every source, or one
#'   value per source. Ignored when K = 0.
#' @param K Number of source studies; 0 generates the target study only.
#' @param family Outcome family, as in [SSTL()]: 'Gaussian', 'Logistic',
#'   'Student-t', 'Poisson', 'Negative-binomial', 'Gamma', 'Beta', 'Weibull',
#'   'Lognormal', or 'Loglogistic'. Case-insensitive.
#' @param s0 Number of nonzero target coefficients (the first s0 predictors).
#' @param signal Magnitude of the nonzero target coefficients: one value, or
#'   s0 values.
#' @param random_sign If TRUE, each nonzero coefficient gets a random sign;
#'   otherwise all are positive.
#' @param rho Correlation parameter; covariates have covariance
#'   rho^|j - j'|.
#' @param K_informative Number of informative sources (the first ones), whose
#'   coefficients equal the target's. The remaining sources keep the first
#'   floor(s0 / 2) target signals, have 2 * s0 new signals of size
#'   `het_signal` with random signs in the following positions, and zeros
#'   elsewhere. Defaults to floor(K / 2); set it to K for all-informative
#'   sources.
#' @param het_signal Magnitude of the new signals in non-informative sources.
#' @param intercept Intercept of the linear predictor: one value for all
#'   studies, or K + 1 values (target first).
#' @param scale Error scale for 'Gaussian' (standard deviation), 'Student-t',
#'   and the AFT families. Default 1.
#' @param df Degrees of freedom for 'Student-t'. Default 4.
#' @param shape Shape for 'Negative-binomial' (size) and 'Gamma'. Default 2.
#' @param precision Precision for 'Beta'. Default 10.
#' @param cens Target censoring proportion in each study for AFT families,
#'   in \[0, 1). Default 0.2; 0 gives no censoring.
#' @param n_test Size of an optional test set drawn from the target model
#'   after the training data; 0 (default) skips it.
#' @param log_time For AFT families, return log times (TRUE, default; fit
#'   with `SSTL(..., log = FALSE)`) or times (FALSE; fit with the default
#'   `log = TRUE`).
#' @param seed Optional seed passed to [set.seed()] before generating.
#' @details Per study, the linear predictor is eta = intercept + X b. Outcomes are
#'   eta + scale * e with e standard normal ('Gaussian'), t with `df` degrees
#'   of freedom ('Student-t'), minimum extreme value ('Weibull', log T),
#'   standard normal ('Lognormal', log T), or standard logistic
#'   ('Loglogistic', log T); Bernoulli(plogis(eta)) ('Logistic');
#'   Poisson(exp(eta)); negative binomial with mean exp(eta) and size `shape`;
#'   gamma with mean exp(eta) and shape `shape`; and beta with mean
#'   plogis(eta) and precision `precision`, kept within \[1e-10, 1 - 1e-10\].
#'   These match the likelihoods used by [SSTL()].
#'
#'   Draws follow the order of the SSTL simulation scripts (target signs,
#'   target covariates, errors, censoring, then each source), and Weibull
#'   errors are drawn as with `extRemes::revd`, so the same RNG state gives
#'   the same datasets as those scripts.
#' @return A list with `X_T`, `Y_T`, `C_T` (target design, response, and
#'   event indicator: 1 = event, 0 = censored), `X_s`, `Y_s`, `C_s` (lists over
#'   sources; empty when K = 0), `b_T` (target coefficients), `b_s` (p x K
#'   matrix of source coefficients), `informative` (logical, per source), and
#'   `settings` (the values used). `C_T` and `C_s` are NULL for non-AFT
#'   families. When `n_test > 0`, also `X_test`, `Y_test`, `C_test`, and
#'   `T_test` (the uncensored test response, on the same scale as `Y_test`).
#' @examples
#' dat <- simulate_SSTL(n_T = 60, p = 50, n_k = 80, K = 2, family = "Weibull",
#'                      s0 = 4, seed = 1)
#' fit <- SSTL(dat$X_T, dat$Y_T, X_s = dat$X_s, Y_s = dat$Y_s,
#'             C = dat$C_T, C_s = dat$C_s, family = "Weibull", log = FALSE,
#'             N = 500, verbose = 0)
#' @export
simulate_SSTL <- function(n_T = 100, p = 500, n_k = 100, K = 2, family = "Gaussian",
                          s0 = 10, signal = 0.5, random_sign = TRUE, rho = 0.7,
                          K_informative = floor(K / 2), het_signal = 0.5, intercept = 0,
                          scale = NULL, df = NULL, shape = NULL, precision = NULL,
                          cens = NULL, n_test = 0, log_time = TRUE, seed = NULL) {
  count <- function(x, name, min = 0) {
    if (length(x) != 1 || !is.finite(x) || x < min || x != floor(x)) {
      stop(name, " must be an integer of at least ", min, ".")
    }
    as.integer(x)
  }
  n_T <- count(n_T, "n_T", 1)
  p <- count(p, "p", 1)
  K <- count(K, "K")
  s0 <- count(s0, "s0")
  n_test <- count(n_test, "n_test")
  K_informative <- count(K_informative, "K_informative")
  if (s0 > p) stop("s0 must not exceed p.")
  if (K_informative > K) stop("K_informative must not exceed K.")
  if (K > 0) {
    if (length(n_k) == 1) n_k <- rep(n_k, K)
    if (length(n_k) != K) stop("n_k must have length 1 or K.")
    n_k <- vapply(n_k, count, integer(1), name = "Each n_k", min = 1)
  } else {
    n_k <- integer(0)
  }
  if (K_informative < K && floor(s0 / 2) + 2 * s0 > p) {
    stop("Non-informative sources need p >= floor(s0 / 2) + 2 * s0.")
  }
  if (!length(signal) %in% c(1, s0) || any(!is.finite(signal))) stop("signal must have length 1 or s0.")
  if (length(het_signal) != 1 || !is.finite(het_signal)) stop("het_signal must be a single number.")
  if (length(rho) != 1 || !is.finite(rho) || abs(rho) >= 1) stop("rho must be in (-1, 1).")
  if (!length(intercept) %in% c(1, K + 1) || any(!is.finite(intercept))) {
    stop("intercept must have length 1 or K + 1.")
  }
  intercept <- rep_len(intercept, K + 1)

  fam <- tolower(family)
  families <- c("gaussian", "logistic", "student-t", "poisson", "negative-binomial", "gamma", "beta",
                "weibull", "lognormal", "loglogistic")
  if (!fam %in% families) stop("family must be one of: ", paste0("'", families, "'", collapse = ", "), ".")
  aft <- fam %in% c("weibull", "lognormal", "loglogistic")

  # Family parameters: defaults when used, an error when supplied but unused
  uses <- list(scale = c("gaussian", "student-t", "weibull", "lognormal", "loglogistic"),
               df = "student-t", shape = c("negative-binomial", "gamma"), precision = "beta",
               cens = c("weibull", "lognormal", "loglogistic"))
  defaults <- list(scale = 1, df = 4, shape = 2, precision = 10, cens = 0.2)
  given <- list(scale = scale, df = df, shape = shape, precision = precision, cens = cens)
  par <- list()
  for (nm in names(uses)) {
    used <- fam %in% uses[[nm]]
    if (!used && !is.null(given[[nm]])) stop("`", nm, "` is not used for family '", family, "'.")
    if (used) {
      par[[nm]] <- if (is.null(given[[nm]])) defaults[[nm]] else given[[nm]]
      ok <- length(par[[nm]]) == 1 && is.finite(par[[nm]]) &&
        if (nm == "cens") par[[nm]] >= 0 && par[[nm]] < 1 else par[[nm]] > 0
      if (!ok) stop("`", nm, "` must be ", if (nm == "cens") "in [0, 1)." else "a positive number.")
    }
  }

  if (!is.null(seed)) set.seed(seed)

  Sigma <- stats::toeplitz(rho^(0:(p - 1)))
  draw_X <- function(n) MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
  # Latent response (uncensored for AFT) given the linear predictor
  draw_y <- function(eta) {
    n <- length(eta)
    switch(fam,
      gaussian = eta + par$scale * stats::rnorm(n),
      logistic = stats::rbinom(n, 1, stats::plogis(eta)),
      `student-t` = eta + par$scale * stats::rt(n, par$df),
      poisson = stats::rpois(n, exp(eta)),
      `negative-binomial` = stats::rnbinom(n, size = par$shape, mu = exp(eta)),
      gamma = stats::rgamma(n, shape = par$shape, rate = par$shape / exp(eta)),
      beta = {
        mu <- stats::plogis(eta)
        y <- stats::rbeta(n, mu * par$precision, (1 - mu) * par$precision)
        pmin(pmax(y, 1e-10), 1 - 1e-10)
      },
      weibull = eta + par$scale * log(stats::rexp(n)), # = eta - extRemes::revd(n, 0, scale)
      lognormal = eta + par$scale * stats::rnorm(n),
      loglogistic = eta + par$scale * stats::rlogis(n))
  }
  draw_study <- function(n, b, b0) {
    X <- draw_X(n)
    y <- draw_y(b0 + as.numeric(X %*% b))
    if (!aft) return(list(X = X, Y = y, C = NULL, T = y))
    log_C <- generate_logC(y, par$cens)
    Y <- pmin(y, log_C)
    out_scale <- if (log_time) identity else exp
    list(X = X, Y = out_scale(Y), C = as.integer(y <= log_C), T = out_scale(y))
  }

  # Target
  b_T <- rep(0, p)
  if (s0 > 0) {
    signs <- if (random_sign) stats::rbinom(s0, 1, 0.5) * 2 - 1 else rep(1, s0)
    b_T[seq_len(s0)] <- signs * rep_len(signal, s0)
  }
  target <- draw_study(n_T, b_T, intercept[1])

  # Sources
  b_s <- matrix(0, p, K)
  informative <- seq_len(K) <= K_informative
  X_s <- Y_s <- C_s <- vector("list", K)
  for (k in seq_len(K)) {
    if (informative[k]) {
      b_s[, k] <- b_T
    } else {
      keep <- s0 %/% 2
      new <- (2 * stats::rbinom(2 * s0, 1, 0.5) - 1) * het_signal
      b_s[, k] <- c(b_T[seq_len(keep)], new, rep(0, p - keep - 2 * s0))
    }
    src <- draw_study(n_k[k], b_s[, k], intercept[k + 1])
    X_s[[k]] <- src$X
    Y_s[[k]] <- src$Y
    if (aft) C_s[[k]] <- src$C
  }

  out <- list(X_T = target$X, Y_T = target$Y, C_T = target$C,
              X_s = X_s, Y_s = Y_s, C_s = if (aft) C_s else NULL,
              b_T = b_T, b_s = b_s, informative = informative)
  if (n_test > 0) {
    test <- draw_study(n_test, b_T, intercept[1])
    out$X_test <- test$X
    out$Y_test <- test$Y
    out$C_test <- test$C
    out$T_test <- test$T
  }
  out$settings <- c(list(family = family, n_T = n_T, p = p, n_k = n_k, K = K, s0 = s0,
                         signal = signal, random_sign = random_sign, rho = rho,
                         K_informative = K_informative, het_signal = het_signal,
                         intercept = intercept),
                    par, list(n_test = n_test, log_time = if (aft) log_time, seed = seed))
  out
}
