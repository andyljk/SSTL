#' ESS-within-Gibbs sampler for Bayesian transfer learning of AFT models
#'
#' Runs elliptical slice sampling updates for target parameters and source biases.
#'
#' @param X_T Target design matrix.
#' @param Y_T Target response.
#' @param C_T Target bbservation indicator.
#' @param X_s List of source design matrices.
#' @param Y_s List of source responses.
#' @param C_s List of source observation indicator.
#' @param group_map The groups for covariates.
#' @param bt.c,bs.c Initial states for target and source parameters.
#' @param lambda_T,lambda_s Threshold parameters.
#' @param xi,xi_s Initial states of shadow variables for the slab scale parameter.
#' @param b0_T,b0_s Initial value for the intercept term, default is 0.
#' @param intercept Whether to estimate the intercept term in the MCMC algorithm, default is TRUE.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param family Specification of outcome model, one of 'Weibull', 'Lognormal', 'Loglogistic'. Default is 'Weibull'.
#' @param slab Specification of slab type, one of 'exp', 'poly', 'nlp'. Default is 'exp'.
#' @param verbose Verbosity flag.
#' @param debug Optional returning of MCMC runs other than the coefficient itself.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs_TL_Group_AFT <- function(X_T, Y_T, C_T=NULL,
                                   X_s, Y_s, C_s=NULL,
                                   group_map,
                                   bt.c=NULL, bs.c=NULL,
                                   lambda_T=NULL, lambda_s=NULL,
                                   xi=NULL, xi_s=NULL,
                                   sig_T = NULL, sig_s = NULL,
                                   b0_T = NULL, b0_s = NULL, intercept = FALSE,
                                   N=5000, S.max=500, block_size=1,
                                   family="Weibull", slab = "exp",
                                   verbose=1, debug=FALSE) {
  fam_map <- c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
  fam_code <- fam_map[tolower(family)]
  if (is.na(fam_code)) stop("Family must be 'weibull', 'loglogistic', or 'lognormal'")

  slab_map <- c("exp" = 1, "poly" = 2, "nlp1" = 3, "nlp2" = 4)
  slab_code <- slab_map[tolower(slab)]
  if (is.na(slab_code)) stop("Slab must be 'exp', 'poly', 'nlp1', or 'nlp2'")

  X_T <- as.matrix(X_T)
  Y_T <- as.numeric(Y_T)
  C_T <- as.numeric(C_T)
  X_s <- lapply(X_s, as.matrix)
  Y_s <- lapply(Y_s, as.numeric)
  C_s <- lapply(C_s, as.numeric)

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
  if (is.null(lambda_T) | is.null(lambda_s)) {
    lambda_T <- p^0.5
    lambda_s <- rep(3, S)
  }
  if (is.null(xi)) xi <- 0.5
  if (is.null(xi_s)) xi_s <- rep(0.5, S)
  if (is.null(sig_T)) sig_T <- 1
  if (is.null(sig_s)) sig_s <- rep(1, S)
  if (is.null(b0_T)) b0_T <- 0
  if (is.null(b0_s)) b0_s <- rep(0, S)

  d <- length(bt.c)
  MC.beta <- matrix(NA, N, p)

  if (debug) {
    N.t <- matrix(NA, N, p + G + 1)
    N.s <- array(NA, dim = c(N, p + G + 1))
    mc.bt <- matrix(NA, N, d)
    mc.bs <- array(NA, dim = c(p + G + 1, S, N))
    mc.sig_T <- rep(NA, N)
    mc.sig_s <- array(NA, dim = c(S, N))
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
      int_res_T <- update_target_intercept_tl_aft(b0_T_curr = b0_T,
                                                  resid_T = resid_T, C_T = C_T,
                                                  sd_y_T = sig_T,
                                                  fam_code = fam_code,
                                                  sd_prior = 2.0)
      b0_T <- int_res_T$b0_T
      resid_T <- int_res_T$resid_T
    }

    cpp_res_T <- update_target_group_aft(bt_c = bt.c,
                                         resid_T = resid_T, resid_S_list = resid_S,
                                         X_T = X_T, C_T = C_T,
                                         X_S_list = X_s, C_S_list = C_s,
                                         id = id, group_map = group_map,
                                         sd_T = sd_T,
                                         lambda_T = lambda_T,
                                         tau = abs(xi),
                                         sd_y_T = sig_T, sd_y_S = sig_s,
                                         S_max = S.max, fam_code = fam_code,
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

    scale_res_T <- update_target_scale_aft(xi_t_curr = xi,
                                           sd_0 = 2.0,
                                           Y_T = Y_T_scale,
                                           resid_T = resid_T,
                                           Y_s_list = Y_s_target_scale,
                                           resid_S_list = resid_S,
                                           bt_c = bt.c,
                                           X_T = X_T, C_T = C_T,
                                           lambda_T = lambda_T,
                                           X_s_list = X_s, C_s_list = C_s,
                                           sd_y_T = sig_T, sd_y_S = sig_s,
                                           fam_code = fam_code, slab_code = slab_code)
    xi <- scale_res_T$xi_t
    resid_T <- scale_res_T$resid_T
    resid_S <- scale_res_T$resid_S
    tau <- abs(xi)
    beta_T_curr <- calc_beta_group(bt.c, group_map, lambda_T, tau, slab_code)

    sig_T <- update_sigma_target_tl_aft(resid = resid_T, C = C_T,
                                        current_sigma = sig_T,
                                        fam_code = fam_code, step_size = 0.1)

    if (debug) {
      mc.bt[i, ] <- bt.c
      N.t[i, ] <- cpp_res_T$N_t
      mc.tau_T[i] <- tau
      mc.sig_T[i] <- sig_T
    }

    if (S > 0) {
      if (intercept) {
        int_res_S <- update_source_intercepts_tl_aft(b0_s_curr = b0_s,
                                                     resid_S_list = resid_S,
                                                     C_s_list = C_s,
                                                     sd_y_S = sig_s,
                                                     fam_code = fam_code,
                                                     sd_prior = 2.0)
        b0_s <- int_res_S$b0_s
        resid_S <- int_res_S$resid_S
      }

      cpp_res_S <- update_source_joint_group_aft(bs_c = bs.c,
                                                 resid_S_list = resid_S,
                                                 X_s_list = X_s, C_s_list = C_s,
                                                 id = id, group_map = group_map,
                                                 lambda_S = lambda_s, tau_S = abs(xi_s),
                                                 sd_y_S = sig_s,
                                                 S_max = S.max,
                                                 fam_code = fam_code,
                                                 slab_code = slab_code)
      bs.c <- cpp_res_S$bs_c
      resid_S <- cpp_res_S$resid_S

      Y_s_source_scale <- lapply(seq_len(S), function(s) {
        as.numeric(Y_s[[s]] - b0_s[s] - X_s[[s]] %*% beta_T_curr)
      })
      scale_res_S <- update_source_scales_aft(xi_s_curr = xi_s,
                                              sd_0 = 0.2,
                                              Y_s_list = Y_s_source_scale,
                                              resid_S_list = resid_S,
                                              bs_c = bs.c,
                                              X_s_list = X_s, C_s_list = C_s,
                                              lambda_S = lambda_s,
                                              sd_y_S = sig_s,
                                              fam_code = fam_code,
                                              slab_code = slab_code)
      xi_s <- scale_res_S$xi_s
      resid_S <- scale_res_S$resid_S
      tau_s <- abs(xi_s)

      for (s in seq_len(S)) {
        sig_s[s] <- update_sigma_source_tl_aft(resid = resid_S[[s]], C = C_s[[s]],
                                               current_sigma = sig_s[s],
                                               fam_code = fam_code,
                                               step_size = 0.1)
        if (debug) mc.sig_s[s, i] <- sig_s[s]
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

  out <- list(MC_beta = MC.beta)
  if (intercept) {
    out$MC_b0_T <- MC.b0_T
    out$MC_b0_S <- MC.b0_S
  }
  out
}


#' stochastic EB estimation
#'
#' @inheritParams ESS_Gibbs_TL_Group_AFT
#' @param gamma_power Robbins–Monro step-size exponent for SAEM updates.
#' @param lr Learning rate for lambda updates.
#' @param K_block Block size (iterations) per SAEM update.
#' @param schedule Exponent controlling learning-rate decay (e.g., lr / t^schedule).
#' @param optimizer "legacy" is the plain doubly smoothed MCEM, "adagrad" for adaptive step sizes.
#' @param polyak,polyak_start Boolean to return average or not, starting from a percentage of the run.
#' @return A list containing MCMC draws and lambda trajectories.
#' @export
EB_SAEM_TL_Group_AFT <- function(X_T, Y_T, C_T=NULL,
                                 X_s, Y_s, C_s=NULL,
                                 group_map,
                                 bt.c=NULL, bs.c=NULL,
                                 lambda_T=NULL, lambda_s=NULL,
                                 xi=NULL, xi_s=NULL,
                                 b0_T=NULL, b0_s=NULL, intercept=FALSE,
                                 N=5000, burn = 1000,
                                 S.max=500, block_size=1,
                                 family="Weibull", slab = "poly",
                                 gamma_power = 0.9,
                                 lr = 0.1, K_block = 10, schedule=0.5,
                                 optimizer = c("legacy", "adagrad"),
                                 max_log_step = 0.35,
                                 polyak = TRUE, polyak_start = 0.5,
                                 verbose=1) {
  optimizer <- match.arg(optimizer)

  X_T <- as.matrix(X_T)
  Y_T <- as.numeric(Y_T)
  C_T <- as.numeric(C_T)
  X_s <- lapply(X_s, as.matrix)
  Y_s <- lapply(Y_s, as.numeric)
  C_s <- lapply(C_s, as.numeric)

  p <- ncol(X_T)
  S <- length(X_s)
  group_map <- as.integer(group_map)
  validate_group_map(group_map, p)
  G <- max(group_map)
  a0_idx <- p + G + 1
  n_blocks <- floor(N / K_block)

  if (is.null(lambda_T)) lambda_T <- 9
  if (is.null(lambda_s)) lambda_s <- rep(3, S)

  res <- ESS_Gibbs_TL_Group_AFT(
    X_T = X_T, Y_T = Y_T, C_T = C_T,
    X_s = X_s, Y_s = Y_s, C_s = C_s,
    group_map = group_map,
    lambda_T = lambda_T, lambda_s = lambda_s,
    b0_T = b0_T, b0_s = b0_s, intercept = intercept,
    N = burn, S.max = S.max, block_size = block_size,
    family = family, slab = slab,
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

  tail_idx <- seq.int(max(1, burn - 999), burn)
  B_hat_T <- mean(pnorm(res$mc_bt[tail_idx, a0_idx], log.p = TRUE))
  B_hat_s <- if (S > 0) {
    sapply(seq_len(S), function(s) mean(pnorm(res$mc_bs[a0_idx, s, tail_idx], log.p = TRUE)))
  } else {
    numeric(0)
  }

  theta <- log(c(lambda_T, lambda_s))
  G_acc <- rep(0, S + 1)

  polyak_from <- max(1, floor(polyak_start * n_blocks))
  lam_avg <- rep(0, S + 1)
  lam_avg_n <- 0

  if (verbose == 1) pb <- txtProgressBar(min = 0, max = n_blocks, style = 3)
  for (t_block in seq_len(n_blocks)) {
    res <- ESS_Gibbs_TL_Group_AFT(
      X_T = X_T, Y_T = Y_T, C_T = C_T,
      X_s = X_s, Y_s = Y_s, C_s = C_s,
      group_map = group_map,
      bt.c = bt.c, bs.c = bs.c,
      lambda_T = lambda_T, lambda_s = lambda_s,
      xi = xi, xi_s = xi_s,
      sig_T = sig_T, sig_s = sig_s,
      b0_T = b0_T, b0_s = b0_s, intercept = intercept,
      N = K_block, S.max = S.max, block_size = block_size,
      family = family, slab = slab,
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

    gamma_t <- t_block^(-gamma_power)
    mean_logPhi_T <- mean(pnorm(res$mc_bt[, a0_idx], log.p = TRUE))
    B_hat_T <- (1 - gamma_t) * B_hat_T + gamma_t * mean_logPhi_T
    B_hat_T <- max(min(B_hat_T, -1e-6), -30)

    for (s in seq_len(S)) {
      mean_logPhi_s <- mean(pnorm(res$mc_bs[a0_idx, s, ], log.p = TRUE))
      B_hat_s[s] <- (1 - gamma_t) * B_hat_s[s] + gamma_t * mean_logPhi_s
      B_hat_s[s] <- max(min(B_hat_s[s], -1e-6), -30)
    }

    if (optimizer == "legacy") {
      lr_t <- lr / (t_block^schedule)
      lambda_target <- -c(lambda_T, lambda_s) / c(B_hat_T, B_hat_s)
      lam_vec <- (1 - lr_t) * c(lambda_T, lambda_s) + lr_t * lambda_target
      lambda_T <- lam_vec[1]
      lambda_s <- lam_vec[2:(S + 1)]
      theta <- log(c(lambda_T, lambda_s))
    } else {
      d_vec <- pmax(-max_log_step, pmin(max_log_step, log(-1 / c(B_hat_T, B_hat_s))))
      G_acc <- G_acc + d_vec^2
      lr_vec <- lr / ((t_block^schedule) * sqrt(G_acc + 1e-8))
      step_theta <- pmax(-max_log_step, pmin(max_log_step, lr_vec * d_vec))
      theta <- theta + step_theta
      lam_vec <- exp(theta)
      lambda_T <- lam_vec[1]
      lambda_s <- lam_vec[2:(S + 1)]
    }

    lam_now <- c(lambda_T, lambda_s)
    mc.lam[t_block, ] <- lam_now
    if (polyak && t_block >= polyak_from) {
      lam_avg_n <- lam_avg_n + 1
      lam_avg <- lam_avg + (lam_now - lam_avg) / lam_avg_n
    }

    if (verbose == 1) setTxtProgressBar(pb, t_block)
  }

  final_lam <- if (polyak && lam_avg_n > 0) lam_avg else c(lambda_T, lambda_s)
  out <- list(mc.lam = mc.lam, final_lam = final_lam,
              bt_c = bt.c, bs_c = bs.c,
              xi = xi, xi_s = xi_s,
              sig_T = sig_T, sig_s = sig_s)
  if (intercept) {
    out$b0_T <- b0_T
    out$b0_s <- b0_s
  }
  out
}
