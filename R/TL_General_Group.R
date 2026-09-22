#' ESS-within-Gibbs sampler for grouped Bayesian transfer learning of uncensored regression models
#'
#' Runs elliptical slice sampling updates for target parameters and source biases.
#'
#' @param X_T Target design matrix.
#' @param Y_T Target response.
#' @param X_s List of source design matrices.
#' @param Y_s List of source responses.
#' @param group_map The groups for covariates.
#' @param bt.c,bs.c Initial states for target and source parameters.
#' @param lambda_T,lambda_s Threshold parameters. The target defaults to sqrt(p), where p is the number of predictors; each source defaults to 3.
#' @param xi,xi_s Initial states of shadow variables for the slab scale parameter.
#' @param sig_T,sig_s Initial target/source outcome standard deviations for 'Gaussian' or scales for 'Student-t'. Fixed at 1 and unused for 'Logistic' and 'Poisson'.
#' @param b0_T,b0_s Initial value for the intercept term, default is 0.
#' @param intercept Whether to estimate the intercept term in the MCMC algorithm, default is TRUE.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param family Outcome distribution: 'Gaussian', 'Logistic' (binary 0/1), 'Student-t', or 'Poisson' (nonnegative integer counts, log link). Case-insensitive; default is 'Gaussian'.
#' @param df Fixed positive degrees of freedom for 'Student-t', default is 4. Ignored for other families.
#' @param slab Slab transformation: 'exp', 'poly', 'nlp', or 'guassian'.
#' @param verbose Verbosity flag.
#' @param debug Optional returning of MCMC runs other than the coefficient itself.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs_TL_Group_General <- function(X_T, Y_T,
                                   X_s, Y_s,
                                   group_map,
                                   bt.c=NULL, bs.c=NULL,
                                   lambda_T=NULL, lambda_s=NULL,
                                   xi=NULL, xi_s=NULL,
                                   sig_T = NULL, sig_s = NULL,
                                   b0_T = NULL, b0_s = NULL, intercept = FALSE,
                                   N=5000, S.max=500, block_size=1,
                                   family="Gaussian", df=4, slab = "exp",
                                   verbose=1, debug=FALSE) {
  fam_map <- c("gaussian" = 1, "logistic" = 2, "student-t" = 3, "poisson" = 4)
  fam_code <- fam_map[tolower(family)]
  if (is.na(fam_code)) stop("Family must be 'Gaussian', 'Logistic', 'Student-t', or 'Poisson'")

  if (fam_code == 3 && (length(df) != 1 || is.na(df) || df <= 0)) {
    stop("df must be a positive number for 'Student-t'.")
  }

  slab_map <- c("exp" = 1, "poly" = 2, "nlp" = 3, "nlp2" = 4, "guassian" = 5)
  slab_code <- slab_map[tolower(slab)]
  if (is.na(slab_code)) stop("Slab must be 'exp', 'poly', 'nlp', or 'guassian'")

  X_T <- as.matrix(X_T)
  Y_T <- as.numeric(Y_T)
  X_s <- lapply(X_s, as.matrix)
  Y_s <- lapply(Y_s, as.numeric)

  if (fam_code == 2 && (any(!Y_T %in% c(0, 1)) || any(vapply(Y_s, function(y) any(!y %in% c(0, 1)), logical(1))))) {
    stop("Target and source responses must contain only 0 and 1 for 'Logistic'.")
  }

  if (fam_code == 4 && (any(!is.finite(Y_T) | Y_T < 0 | Y_T != floor(Y_T)) ||
                       any(vapply(Y_s, function(y) any(!is.finite(y) | y < 0 | y != floor(y)), logical(1))))) {
    stop("Target and source responses must contain only nonnegative integer counts for 'Poisson'.")
  }

  p <- ncol(X_T)
  S <- length(X_s)
  group_map <- as.integer(group_map)
  validate_group_map(group_map, p)
  G <- max(group_map)
  id <- build_group_id(group_map)

  sd_T <- sqrt(c(rep(1, p), rep(1, G), 1))
  if (is.null(bt.c)) bt.c <- rnorm(p + G + 1, 0, sd_T)
  if (is.null(bs.c)) {
    bs.c <- matrix(rnorm((p + G + 1) * S, 0, c(rep(0.05, p), rep(1, G + 1))),
                   nrow = p + G + 1, ncol = S)
  }
  if (is.null(lambda_T)) lambda_T <- sqrt(p)
  if (is.null(lambda_s)) lambda_s <- rep(3, S)
  if (is.null(xi)) xi <- 0.5
  if (is.null(xi_s)) xi_s <- rep(0.5, S)
  if (fam_code %in% c(2, 4) || is.null(sig_T)) sig_T <- 1
  if (fam_code %in% c(2, 4) || is.null(sig_s)) sig_s <- rep(1, S)
  if (is.null(b0_T)) b0_T <- 0
  if (is.null(b0_s)) b0_s <- rep(0, S)

  d <- length(bt.c)
  MC.beta <- matrix(NA, N, p)
  mc.sig_T <- rep(NA, N)
  mc.sig_s <- array(NA, dim = c(S, N))

  if (debug) {
    N.t <- matrix(NA, N, p + G + 1)
    N.s <- array(NA, dim = c(N, p + G + 1))
    mc.bt <- matrix(NA, N, d)
    mc.bs <- array(NA, dim = c(p + G + 1, S, N))
    mc.tau_T <- rep(NA, N)
    mc.tau_S <- array(NA, dim = c(N, S))
  }

  if (intercept) {
    MC.b0_T <- rep(NA, N)
    MC.b0_S <- array(NA, dim = c(N, S))
  }

  beta_T_init <- calc_beta_group(bt.c, group_map, lambda_T, abs(xi), slab_code)
  resid_T <- as.numeric(Y_T - b0_T - X_T %*% beta_T_init)
  if (S > 0) {
    bias_s_init <- lapply(seq_len(S), function(s) {
      calc_beta_group(bs.c[, s], group_map, lambda_s[s], abs(xi_s[s]), slab_code)
    })
    resid_S <- lapply(seq_len(S), function(s) {
      as.numeric(Y_s[[s]] - b0_s[s] - X_s[[s]] %*% (beta_T_init + bias_s_init[[s]]))
    })
  } else {
    resid_S <- list()
  }

  if (verbose == 1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for (i in seq_len(N)) {
    if (intercept) {
      int_res_T <- update_target_intercept_tl_general(b0_T_curr = b0_T,
                                                  resid_T = resid_T, Y_T = Y_T,
                                                  sd_y_T = sig_T,
                                                  fam_code = fam_code, df=df,
                                                  sd_prior = 2.0)
      b0_T <- int_res_T$b0_T
      resid_T <- int_res_T$resid_T
    }

    cpp_res_T <- update_target_group_general(bt_c = bt.c,
                                         resid_T = resid_T, Y_T = Y_T, resid_S_list = resid_S, Y_s_list = Y_s,
                                         X_T = X_T,
                                         X_S_list = X_s,
                                         id = id, group_map = group_map,
                                         sd_T = sd_T,
                                         lambda_T = lambda_T,
                                         tau = abs(xi),
                                         sd_y_T = sig_T, sd_y_S = sig_s,
                                         S_max = S.max, fam_code = fam_code, df=df,
                                         slab_code = slab_code)
    bt.c <- cpp_res_T$bt_c
    resid_T <- cpp_res_T$resid_T
    resid_S <- cpp_res_T$resid_S

    Y_T_scale <- Y_T - b0_T
    if (S > 0) {
      bias_s_curr <- lapply(seq_len(S), function(s) {
        calc_beta_group(bs.c[, s], group_map, lambda_s[s], abs(xi_s[s]), slab_code)
      })
      Y_s_target_scale <- lapply(seq_len(S), function(s) {
        as.numeric(Y_s[[s]] - b0_s[s] - X_s[[s]] %*% bias_s_curr[[s]])
      })
    } else {
      Y_s_target_scale <- list()
    }

    scale_res_T <- update_target_scale_general(xi_t_curr = xi,
                                           sd_0 = 2.0,
                                           Y_T_scale = Y_T_scale,
                                           resid_T = resid_T, Y_T = Y_T,
                                           Y_s_scale_list = Y_s_target_scale,
                                           resid_S_list = resid_S, Y_s_list = Y_s,
                                           bt_c = bt.c,
                                           X_T = X_T,
                                           lambda_T = lambda_T,
                                           X_s_list = X_s,
                                           sd_y_T = sig_T, sd_y_S = sig_s,
                                           fam_code = fam_code, df=df, slab_code = slab_code)
    xi <- scale_res_T$xi_t
    resid_T <- scale_res_T$resid_T
    resid_S <- scale_res_T$resid_S
    tau <- abs(xi)
    beta_T_curr <- calc_beta_group(bt.c, group_map, lambda_T, tau, slab_code)

    if (!fam_code %in% c(2, 4)) {
      sig_T <- update_sigma_target_tl_general(resid = resid_T, Y = Y_T,
                                          current_sigma = sig_T,
                                          fam_code = fam_code, df=df, step_size = 0.1)
    }

    mc.sig_T[i] <- sig_T
    if (debug) {
      mc.bt[i, ] <- bt.c
      N.t[i, ] <- cpp_res_T$N_t
      mc.tau_T[i] <- tau
    }

    if (S > 0) {
      if (intercept) {
        int_res_S <- update_source_intercepts_tl_general(b0_s_curr = b0_s,
                                                     resid_S_list = resid_S, Y_s_list = Y_s,
                                                     sd_y_S = sig_s,
                                                     fam_code = fam_code, df=df,
                                                     sd_prior = 2.0)
        b0_s <- int_res_S$b0_s
        resid_S <- int_res_S$resid_S
      }

      cpp_res_S <- update_source_joint_group_general(bs_c = bs.c,
                                                 resid_S_list = resid_S, Y_s_list = Y_s,
                                                 X_s_list = X_s,
                                                 id = id, group_map = group_map,
                                                 lambda_S = lambda_s, tau_S = abs(xi_s),
                                                 sd_y_S = sig_s,
                                                 S_max = S.max,
                                                 fam_code = fam_code, df=df,
                                                 slab_code = slab_code)
      bs.c <- cpp_res_S$bs_c
      resid_S <- cpp_res_S$resid_S

      Y_s_source_scale <- lapply(seq_len(S), function(s) {
        as.numeric(Y_s[[s]] - b0_s[s] - X_s[[s]] %*% beta_T_curr)
      })
      scale_res_S <- update_source_scales_general(xi_s_curr = xi_s,
                                              sd_0 = 0.2,
                                              Y_s_scale_list = Y_s_source_scale,
                                              resid_S_list = resid_S, Y_s_list = Y_s,
                                              bs_c = bs.c,
                                              X_s_list = X_s,
                                              lambda_S = lambda_s,
                                              sd_y_S = sig_s,
                                              fam_code = fam_code, df=df,
                                              slab_code = slab_code)
      xi_s <- scale_res_S$xi_s
      resid_S <- scale_res_S$resid_S
      tau_s <- abs(xi_s)

      for (s in seq_len(S)) {
        if (!fam_code %in% c(2, 4)) {
          sig_s[s] <- update_sigma_source_tl_general(resid = resid_S[[s]], Y = Y_s[[s]],
                                                 current_sigma = sig_s[s],
                                                 fam_code = fam_code, df=df,
                                                 step_size = 0.1)
        }
        mc.sig_s[s, i] <- sig_s[s]
      }

      if (debug) {
        mc.bs[, , i] <- bs.c
        N.s[i, ] <- cpp_res_S$N_s
        mc.tau_S[i, ] <- tau_s
      }
    }

    if (intercept) {
      MC.b0_T[i] <- b0_T
      if (S > 0) MC.b0_S[i, ] <- b0_s
    }

    MC.beta[i, ] <- beta_T_curr

    if (verbose == 1) setTxtProgressBar(pb, i)
  }

  if (debug) {
    out <- list(mc_bt = mc.bt, mc_bs = mc.bs, MC_beta = MC.beta, n_t = N.t, n_s = N.s,
                mc_sig_T = mc.sig_T, mc_sig_s = mc.sig_s,
                mc_tau_T = mc.tau_T, mc_tau_S = mc.tau_S)
    if (intercept) {
      out$mc_b0_T <- MC.b0_T
      out$mc_b0_S <- MC.b0_S
    }
    return(out)
  }

  out <- list(MC_beta = MC.beta,
              mc_sig_T = mc.sig_T, mc_sig_s = mc.sig_s)
  if (intercept) {
    out$MC_b0_T <- MC.b0_T
    out$MC_b0_S <- MC.b0_S
  }
  out
}


#' stochastic EB estimation
#'
#' @inheritParams ESS_Gibbs_TL_Group_General
#' @param gamma_power Robbins–Monro step-size exponent for SAEM updates.
#' @param lr Learning rate for lambda updates.
#' @param K_block Block size (iterations) per SAEM update.
#' @param schedule Exponent controlling learning-rate decay (e.g., lr / t^schedule).
#' @param lambda_min,lambda_max Minimum and maximum values of lambda.
#' @param warm_start Whether to run ten fixed MCEM warm-start iterations before SAEM.
#' @param polyak,polyak_start Boolean to return average or not, starting from a percentage of the run.
#' @param verbose 1 or 0 to show progress bar or not.
#' @return A list containing MCMC draws and lambda trajectories.
#' @export
EB_SAEM_TL_Group_General <- function(X_T, Y_T,
                                 X_s, Y_s,
                                 group_map,
                                 bt.c=NULL, bs.c=NULL,
                                 lambda_T=NULL, lambda_s=NULL,
                                 xi=NULL, xi_s=NULL,
                                 b0_T=NULL, b0_s=NULL, intercept=FALSE,
                                 N=5000, burn = 1000,
                                 S.max=500, block_size=1,
                                 family="Gaussian", df=4, slab = "poly",
                                 gamma_power = 0.9,
                                 lr = 0.1, K_block = 10, schedule=0.5,
                                 lambda_min = 1e-3,
                                 lambda_max = 1e4,
                                 polyak = TRUE, polyak_start = 0.9,
                                 warm_start = TRUE,
                                 verbose=1) {

  # Type safety
  X_T <- as.matrix(X_T)
  Y_T <- as.numeric(Y_T)
  X_s <- lapply(X_s, as.matrix)
  Y_s <- lapply(Y_s, as.numeric)

  p <- ncol(X_T)
  S <- length(X_s)
  group_map <- as.integer(group_map)
  validate_group_map(group_map, p)
  G <- max(group_map)
  a0_idx <- p + G + 1
  n_blocks <- floor(N / K_block)
  if (!is.logical(warm_start) || length(warm_start) != 1 || is.na(warm_start)) {
    stop("warm_start must be TRUE or FALSE.")
  }

  if (is.null(lambda_T)) lambda_T <- sqrt(p)
  if (is.null(lambda_s)) lambda_s <- rep(3, S)
  lambda_T <- min(lambda_max, max(lambda_min, lambda_T))
  lambda_s <- pmin(lambda_max, pmax(lambda_min, lambda_s))
  lambda_active <- c(lambda_T, lambda_s) > lambda_min & c(lambda_T, lambda_s) < lambda_max
  lambda_fixed <- !lambda_active

  res <- ESS_Gibbs_TL_Group_General(
    X_T = X_T, Y_T = Y_T,
    X_s = X_s, Y_s = Y_s,
    group_map = group_map,
    lambda_T = lambda_T, lambda_s = lambda_s,
    b0_T = b0_T, b0_s = b0_s, intercept = intercept,
    N = burn, S.max = S.max, block_size = block_size,
    family = family, df=df, slab = slab,
    verbose = 0, debug = TRUE
  )

  bt.c <- res$mc_bt[burn, ]
  bs.c <- matrix(res$mc_bs[, , burn], nrow = p + G + 1, ncol = S)
  xi <- abs(res$mc_tau_T[burn])
  xi_s <- abs(res$mc_tau_S[burn, ])
  sig_T <- res$mc_sig_T[burn]
  sig_s <- res$mc_sig_s[, burn]
  if (intercept) {
    b0_T <- res$mc_b0_T[burn]
    b0_s <- if (S > 0) res$mc_b0_S[burn, ] else numeric(0)
  } else {
    b0_T <- 0
    b0_s <- rep(0, S)
  }

  mc.lam <- array(NA, dim = c(n_blocks, S + 1))
  colnames(mc.lam) <- c("lambda_T", if (S > 0) paste0("lambda_s", seq_len(S)) else NULL)
  warm_start_iter <- 10
  warm_start_N <- 300
  mc_lam_mcem <- array(NA, dim=c(if (warm_start) warm_start_iter else 0, S+1))
  colnames(mc_lam_mcem) <- colnames(mc.lam)

  tail_idx <- seq.int(max(1, burn - 999), burn)
  B_hat_T <- mean(pnorm(res$mc_bt[tail_idx, a0_idx], log.p = TRUE))
  B_hat_s <- if (S > 0) {
    sapply(seq_len(S), function(s) mean(pnorm(res$mc_bs[a0_idx, s, tail_idx], log.p = TRUE)))
  } else {
    numeric(0)
  }

  # Polyak-Ruppert average state
  polyak_from <- max(1, floor(polyak_start * n_blocks))
  lam_avg <- rep(0, S + 1)
  lam_avg_n <- 0

  # warm start using MCEM - good for fast approximation to solution neighborhood
  if (warm_start) {
    for (mcem_iter in seq_len(warm_start_iter)) {
      if (verbose == 1) cat("warm start iteration: ", mcem_iter, "/", warm_start_iter, "\n", sep="")
      res <- ESS_Gibbs_TL_Group_General(X_T=X_T, Y_T=Y_T,
                                    X_s=X_s, Y_s=Y_s,
                                    group_map=group_map,
                                    bt.c=bt.c, bs.c=bs.c,
                                    lambda_T=lambda_T, lambda_s=lambda_s,
                                    xi=xi, xi_s=xi_s,
                                    sig_T=sig_T, sig_s=sig_s,
                                    b0_T=b0_T, b0_s=b0_s, intercept=intercept,
                                    N=warm_start_N, S.max=S.max, block_size=block_size,
                                    family=family, df=df, slab=slab,
                                    verbose=0, debug=TRUE)
      bt.c  <- res$mc_bt[warm_start_N, ]
      xi    <- abs(res$mc_tau_T[warm_start_N])
      sig_T <- res$mc_sig_T[warm_start_N]
      if (intercept) b0_T <- res$mc_b0_T[warm_start_N]

      bs.c  <- matrix(res$mc_bs[,,warm_start_N], nrow=p+G+1, ncol=S)
      xi_s  <- abs(res$mc_tau_S[warm_start_N, ])
      sig_s <- res$mc_sig_s[, warm_start_N]
      if (intercept && S > 0) b0_s <- res$mc_b0_S[warm_start_N, ]

      mean_logPhi_T <- mean(pnorm(res$mc_bt[,a0_idx], log.p=TRUE))
      mean_logPhi_s <- if (S > 0) {
        sapply(seq_len(S), function(s) mean(pnorm(res$mc_bs[a0_idx, s,], log.p=TRUE)))
      } else {
        numeric(0)
      }
      mean_logPhi_vec <- pmax(pmin(c(mean_logPhi_T, mean_logPhi_s), -1e-6), -30)
      lam_vec <- c(lambda_T, lambda_s)
      active_idx <- which(lambda_active)
      if (length(active_idx) > 0) {
        lam_vec[active_idx] <- -lam_vec[active_idx] / mean_logPhi_vec[active_idx]
        lam_vec[active_idx] <- pmin(lambda_max, pmax(lambda_min, lam_vec[active_idx]))
        lambda_active[active_idx] <- lam_vec[active_idx] > lambda_min & lam_vec[active_idx] < lambda_max
        lambda_fixed <- !lambda_active
      }
      lambda_T <- lam_vec[1]
      lambda_s <- lam_vec[-1]
      mc_lam_mcem[mcem_iter, ] <- lam_vec
      B_hat_T <- mean_logPhi_vec[1]
      B_hat_s <- mean_logPhi_vec[-1]
    }
  }

  if (verbose == 1) pb <- txtProgressBar(min = 0, max = n_blocks, style = 3)
  for (t_block in seq_len(n_blocks)) {
    res <- ESS_Gibbs_TL_Group_General(
      X_T = X_T, Y_T = Y_T,
      X_s = X_s, Y_s = Y_s,
      group_map = group_map,
      bt.c = bt.c, bs.c = bs.c,
      lambda_T = lambda_T, lambda_s = lambda_s,
      xi = xi, xi_s = xi_s,
      sig_T = sig_T, sig_s = sig_s,
      b0_T = b0_T, b0_s = b0_s, intercept = intercept,
      N = K_block, S.max = S.max, block_size = block_size,
      family = family, df=df, slab = slab,
      verbose = 0, debug = TRUE
    )

    bt.c <- res$mc_bt[K_block, ]
    xi <- abs(res$mc_tau_T[K_block])
    sig_T <- res$mc_sig_T[K_block]
    if (intercept) b0_T <- res$mc_b0_T[K_block]

    bs.c <- matrix(res$mc_bs[, , K_block], nrow = p + G + 1, ncol = S)
    xi_s <- abs(res$mc_tau_S[K_block, ])
    sig_s <- res$mc_sig_s[, K_block]
    if (intercept && S > 0) b0_s <- res$mc_b0_S[K_block, ]

    # SAEM update
    gamma_t <- t_block^(-gamma_power)
    mean_logPhi_T <- mean(pnorm(res$mc_bt[, a0_idx], log.p = TRUE))
    if (lambda_active[1]) {
      B_hat_T      <- (1 - gamma_t) * B_hat_T + gamma_t * mean_logPhi_T
      B_hat_T      <- max(min(B_hat_T, -1e-6), -30)  # clip
    }

    for (s in seq_len(S)) {
      mean_logPhi_s <- mean(pnorm(res$mc_bs[a0_idx, s,], log.p = TRUE))
      if (lambda_active[s + 1]) {
        B_hat_s[s] <- (1 - gamma_t) * B_hat_s[s] + gamma_t * mean_logPhi_s
        B_hat_s[s] <- max(min(B_hat_s[s], -1e-6), -30)
      }
    }

    lr_t <- lr / (t_block^schedule)
    lam_vec <- c(lambda_T, lambda_s)
    B_hat_vec <- c(B_hat_T, B_hat_s)
    active_idx <- which(lambda_active)
    if (length(active_idx) > 0) {
      lambda_target <- -lam_vec[active_idx] / B_hat_vec[active_idx]
      lam_vec[active_idx] <- (1-lr_t) * lam_vec[active_idx] + lr_t * lambda_target
      lam_vec[active_idx] <- pmin(lambda_max, pmax(lambda_min, lam_vec[active_idx]))
      lambda_active[active_idx] <- lam_vec[active_idx] > lambda_min & lam_vec[active_idx] < lambda_max
      lambda_fixed <- !lambda_active
    }
    lambda_T = lam_vec[1]; lambda_s = lam_vec[-1]

    lam_now <- c(lambda_T, lambda_s)
    mc.lam[t_block, ] <- lam_now
    if (polyak && t_block >= polyak_from) {
      lam_avg_n <- lam_avg_n + 1
      lam_avg <- lam_avg + (lam_now - lam_avg) / lam_avg_n
    }

    if (verbose == 1) setTxtProgressBar(pb, t_block)
  }

  final_lam <- if (polyak && lam_avg_n > 0) lam_avg else c(lambda_T, lambda_s)
  final_lam[lambda_fixed] <- c(lambda_T, lambda_s)[lambda_fixed]
  out <- list(mc.lam = mc.lam, final_lam = final_lam,
              bt_c = bt.c, bs_c = bs.c,
              xi = xi, xi_s = xi_s,
              sig_T = sig_T, sig_s = sig_s)
  if (intercept) {
    out$b0_T <- b0_T
    out$b0_s <- b0_s
  }
  out$mc_lam_mcem <- mc_lam_mcem
  return(out)
}
