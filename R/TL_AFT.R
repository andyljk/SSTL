#' ESS-within-Gibbs sampler for Bayesian transfer learning of AFT models
#'
#' Runs elliptical slice sampling updates for target parameters and source biases.
#'
#' @param X_T Target design matrix.
#' @param Y_T Target response.
#' @param C_T Target bbservation indicator.
#' @param X_s List of source design matrices.
#' @param Y_s List of source responses.
#' @param C_s List of source observation indicator
#' @param bt.c,bs.c Initial states for target and source parameters.
#' @param lambda_T,lambda_s Threshold parameters.
#' @param xi,xi_s Initial states of shadow variables for the slab scale parameter.
#' @param b0_T,b0_s Initial value for the intercept term, default is 0.
#' @param intercept Whether to estimate the intercept term in the MCMC algorithm, default is TRUE.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param family Specification of outcome model, one of 'Weibull', 'Lognormal', 'Loglogistic'. Default is 'Weibull'.
#' @param slab Specification of slab type, one of 'exp', 'poly', 'nlp', 'guassian'. Default is 'exp'.
#' @param verbose Verbosity flag.
#' @param debug Optional returning of MCMC runs other than the coefficient itself.
#' @return A list containing MCMC draws and diagnostics.
#' @export
# elliptical slice sampling within gibbs function
ESS_Gibbs_TL_AFT <- function(X_T, Y_T, C_T=NULL, # Target Data
                             X_s, Y_s, C_s=NULL, # Source Data (Optional)
                             bt.c=NULL, bs.c=NULL,
                             lambda_T=NULL, lambda_s=NULL,
                             xi=NULL, xi_s=NULL, xi_prior=1,
                             sig_T = NULL, sig_s = NULL,
                             b0_T = NULL, b0_s = NULL, intercept = TRUE,
                             N=5000, S.max=500, block_size=1,
                             family="Weibull", slab = "exp",
                             verbose=1, debug=F) {

  fam_map <- c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
  fam_code <- fam_map[tolower(family)]
  if(is.na(fam_code)) stop("Family must be 'weibull', 'loglogistic', or 'lognormal'")

  slab_map <- c("exp" = 1, "poly" = 2, "nlp1" = 3, "nlp2" = 4, "guassian" = 5)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'poly', 'nlp1/nlp2', or 'guassian'")

  # Type Safety
  X_T <- as.matrix(X_T); Y_T <- as.numeric(Y_T); C_T <- as.numeric(C_T)
  X_s <- lapply(X_s, as.matrix); Y_s <- lapply(Y_s, as.numeric); C_s <- lapply(C_s, as.numeric)

  p = ncol(X_T); n_t = nrow(X_T)
  S = length(X_s)
  # setup inputs for MCMC
  id <- lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx <- min(start_idx + block_size - 1, p)
    w_idx <- start_idx:end_idx; a_idx <- w_idx + p
    return(c(w_idx, a_idx))
  })
  id <- c(id, list(2*p + 1))

  # Defaults
  sd_T = sqrt(c(rep(1, p), rep(1,p), 1))
  if (is.null(bt.c)) bt.c = c(rnorm(2*p+1, 0, sd_T))
  if (is.null(bs.c)) bs.c = matrix(rnorm((2*p+1)*S, 0, c(rep(0.05,p),rep(1,p+1))), nrow=2*p+1, ncol=S)
  if(is.null(lambda_T) | is.null(lambda_s)){ lambda_T = p^0.5; lambda_s = rep(3,S) }

  d    <- length(bt.c)                   # nr of parameters
  K    <- length(id)                     # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  MC.beta = matrix(NA, N, p)
  mc.sig_T = rep(NA,N); mc.sig_s = array(NA, dim=c(S,N))
  if (debug){
    N.t  <- matrix(NA, N, K)               # nr of slice sampling itr at each MCMC-itr
    N.s  <- array(NA, dim=c(N, (2*p+1)))   # nr of slice sampling itr at each MCMC-itr
    mc.bt <- matrix(NA, N, d)              # storage for the target parameter
    mc.bs = array(NA, dim=c(2*p+1, S, N))  # storage for the bias parameters
    mc.tau_T = rep(NA,N)
    mc.tau_S = array(NA,dim=c(N,S))
  }
  if (intercept) {
    MC.b0_T <- rep(NA, N)
    MC.b0_S <- array(NA, dim = c(N, S))
  }
  if (is.null(xi)) xi = 0.5
  if (is.null(xi_s)) xi_s = rep(0.5,S)
  if (is.null(sig_T)) sig_T = 1
  if (is.null(sig_s)) sig_s = rep(1,S) # initialize scale parameters
  if (is.null(b0_T)) b0_T = 0
  if (is.null(b0_s)) b0_s = rep(0, S)

  beta_T_init <- calc_beta(bt.c, lambda_T, abs(xi), p, slab_code)
  resid_T <- as.numeric(Y_T - b0_T - X_T %*% beta_T_init)
  if (S > 0) {
    bias_s_init <- lapply(seq_len(S), function(s) {
      calc_beta(bs.c[, s], lambda_s[s], abs(xi_s[s]), p, slab_code)
    })
    resid_S <- lapply(seq_len(S), function(s) {
      as.numeric(Y_s[[s]] - b0_s[s] - X_s[[s]] %*% (beta_T_init + bias_s_init[[s]]))
    })
  } else {
    resid_S <- list()
  }


  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    # ---------------------------------------------------------
    # STEP A: Update Target (Always Runs)
    # ---------------------------------------------------------
    if (intercept) {
      int_res_T <- update_target_intercept_tl_aft(b0_T_curr = b0_T,
                                                  resid_T = resid_T, C_T = C_T,
                                                  sd_y_T = sig_T,
                                                  fam_code = fam_code,
                                                  sd_prior = 2.0)
      b0_T <- int_res_T$b0_T
      resid_T <- int_res_T$resid_T
    }

    # Note: C++ function naturally handles S=0 (loops over sources won't run)
    cpp_res_T <- update_target_aft(bt_c = bt.c,
                                   resid_T = resid_T, resid_S_list = resid_S,
                                   X_T = X_T, C_T = C_T,
                                   X_S_list = X_s, C_S_list = C_s,
                                   id = id, sd_T = sd_T,
                                   lambda_T = lambda_T,
                                   tau = abs(xi),
                                   sd_y_T = sig_T, sd_y_S = sig_s,
                                   S_max = S.max, fam_code=fam_code, slab_code=slab_code)
    bt.c <- cpp_res_T$bt_c
    resid_T <- cpp_res_T$resid_T
    resid_S <- cpp_res_T$resid_S

    # Update Target Scale
    Y_T_scale <- Y_T - b0_T
    if (S > 0) {
      bias_s_curr <- lapply(seq_len(S), function(s) {
        calc_beta(bs.c[, s], lambda_s[s], abs(xi_s[s]), p, slab_code)
      })
      Y_s_target_scale <- lapply(seq_len(S), function(s) {
        as.numeric(Y_s[[s]] - b0_s[s] - X_s[[s]] %*% bias_s_curr[[s]])
      })
    } else {
      Y_s_target_scale <- list()
    }
    scale_res_T <- update_target_scale_aft(xi_t_curr = xi,
                                           sd_0 = xi_prior,
                                           Y_T = Y_T_scale,
                                           resid_T = resid_T,
                                           Y_s_list = Y_s_target_scale,
                                           resid_S_list = resid_S,
                                           bt_c = bt.c,
                                           X_T = X_T, C_T = C_T,
                                           lambda_T = lambda_T,
                                           X_s_list = X_s, C_s_list = C_s,
                                           sd_y_T = sig_T, sd_y_S = sig_s,
                                           fam_code=fam_code, slab_code = slab_code)
    xi <- scale_res_T$xi_t
    resid_T <- scale_res_T$resid_T
    resid_S <- scale_res_T$resid_S
    tau <- abs(xi)
    beta_T_curr <- calc_beta(bt.c, lambda_T, tau, p, slab_code)

    # update prior variance of w in target
    # tau2_w = 1/rgamma(1, shape = 3 + p/2, 2 + sum(bt.c[1:p]^2)/2)
    # sd_T[1:p] = tau2_w^0.5; mc.tau2_wT[i] = tau2_w

    # update scale parameter for target
    sig_T = update_sigma_target_tl_aft(resid = resid_T, C = C_T,
                                       current_sigma = sig_T,
                                       fam_code=fam_code, step_size=0.1)

    if (debug){
      mc.bt[i, ] <- bt.c
      N.t[i,] = cpp_res_T$N_t
      mc.tau_T[i] <- tau
    }
    mc.sig_T[i] = sig_T


    # ---------------------------------------------------------
    # STEP B: Update Sources (Run ONLY if S > 0)
    # ---------------------------------------------------------
    if (S > 0){
      if (intercept) {
        int_res_S <- update_source_intercepts_tl_aft(b0_s_curr = b0_s,
                                                     resid_S_list = resid_S, C_s_list = C_s,
                                                     sd_y_S = sig_s,
                                                     fam_code = fam_code,
                                                     sd_prior = 2.0)
        b0_s <- int_res_S$b0_s
        resid_S <- int_res_S$resid_S
      }

      cpp_res_S <- update_source_joint_aft(bs_c = bs.c,
                                           resid_S_list = resid_S,
                                           X_s_list = X_s, C_s_list = C_s,
                                           lambda_S = lambda_s, tau_S = abs(xi_s),
                                           sd_y_S = sig_s,
                                           S_max = S.max,
                                           fam_code=fam_code, slab_code=slab_code)
      bs.c <- cpp_res_S$bs_c
      resid_S <- cpp_res_S$resid_S

      # Update Source Scales (Jointly with Independent Prior)
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
                                              fam_code = fam_code, slab_code = slab_code)
      xi_s <- scale_res_S$xi_s
      resid_S <- scale_res_S$resid_S
      tau_s <- abs(xi_s)

      # update scale parameter for sources
      for (s in 1:S){
        sig_s[s] <- update_sigma_source_tl_aft(resid = resid_S[[s]], C = C_s[[s]],
                                               current_sigma = sig_s[s],
                                               fam_code=fam_code,
                                               step_size=0.1)
        mc.sig_s[s, i] <- sig_s[s]
      }

      if (debug){
        mc.bs[,,i] <- bs.c
        N.s[i,] = cpp_res_S$N_s
        mc.tau_S[i, ] <- tau_s
      }
    }

    if (intercept) {
      MC.b0_T[i] <- b0_T
      if (S > 0) MC.b0_S[i, ] <- b0_s
    }

    MC.beta[i,] = beta_T_curr

    if (verbose==1) setTxtProgressBar(pb, i)
  }
  if (debug){
    out <- list(mc_bt=mc.bt, mc_bs=mc.bs, MC_beta = MC.beta, n_t=N.t, n_s=N.s,
                mc_sig_T = mc.sig_T, mc_sig_s = mc.sig_s,
                mc_tau_T = mc.tau_T, mc_tau_S = mc.tau_S)
    if (intercept) {
      out$mc_b0_T <- MC.b0_T
      out$mc_b0_S <- MC.b0_S
    }
    return(out)
  }else{
    out <- list(MC_beta = MC.beta,
                mc_sig_T = mc.sig_T, mc_sig_s = mc.sig_s)
    if (intercept) {
      out$MC_b0_T <- MC.b0_T
      out$MC_b0_S <- MC.b0_S
    }
    return(out)
  }
}

#' stochastic version
#'
#' @inheritParams ESS_Gibbs_TL_AFT
#' @param gamma_power Robbins–Monro step-size exponent for SAEM updates.
#' @param lr Learning rate for lambda updates.
#' @param K_block Block size (iterations) per SAEM update.
#' @param schedule Exponent controlling learning-rate decay (e.g., lr / t^schedule).
#' @param lambda_min,lambda_max Minimum and maximum values of lambda.
#' @param warm_start Whether to run five fixed MCEM warm-start iterations before SAEM.
#' @param polyak,polyak_start Boolean to return average or not, starting from a percentage of the run.
#' @param verbose 1 or 0 to show progress bar or not.
#' @return A list containing MCMC draws and lambda trajectories.
#' @export
EB_SAEM_TL_AFT = function(X_T, Y_T, C_T=NULL, # Target Data
                          X_s, Y_s, C_s=NULL, # Source Data (Optional)
                          bt.c=NULL, bs.c=NULL,
                          lambda_T=NULL, lambda_s=NULL,
                          xi=NULL, xi_s=NULL,
                          b0_T=NULL, b0_s=NULL, intercept=FALSE,
                          N=5000, burn = 1000,
                          S.max=50, block_size=1,
                          family="Weibull", slab = "poly",
                          gamma_power = 0.9,
                          lr = 0.1, K_block = 10, schedule=0.5,
                          lambda_min = 1e-3,
                          lambda_max = 1e4,
                          polyak = TRUE, polyak_start = 0.9,
                          warm_start = TRUE,
                          verbose=1){

  # Type safety
  X_T <- as.matrix(X_T); Y_T <- as.numeric(Y_T); C_T <- as.numeric(C_T)
  X_s <- lapply(X_s, as.matrix); Y_s <- lapply(Y_s, as.numeric); C_s <- lapply(C_s, as.numeric)

  p <- ncol(X_T)
  S <- length(X_s)
  n_blocks <- floor(N / K_block)
  if (!is.logical(warm_start) || length(warm_start) != 1 || is.na(warm_start)) {
    stop("warm_start must be TRUE or FALSE.")
  }

  # Init lambdas
  if (is.null(lambda_T)) lambda_T <- 9
  if (is.null(lambda_s)) lambda_s <- rep(3, S)
  lambda_T <- min(lambda_max, max(lambda_min, lambda_T))
  lambda_s <- pmin(lambda_max, pmax(lambda_min, lambda_s))
  lambda_active <- c(lambda_T, lambda_s) > lambda_min & c(lambda_T, lambda_s) < lambda_max
  lambda_fixed <- !lambda_active

  # Burn-in
  res <- ESS_Gibbs_TL_AFT(
    X_T = X_T, Y_T = Y_T, C_T = C_T,
    X_s = X_s, Y_s = Y_s, C_s = C_s,
    lambda_T = lambda_T, lambda_s = lambda_s,
    b0_T = b0_T, b0_s = b0_s, intercept = intercept,
    N = burn, S.max = S.max, block_size = block_size,
    family = family, slab = slab,
    verbose = 0, debug = TRUE
  )

  # Initialize the parameters from burn in
  bt.c = res$mc_bt[burn,]
  bs.c = matrix(res$mc_bs[,,burn], nrow=2*p+1, ncol=S)
  xi = abs(res$mc_tau_T[burn]); xi_s = abs(res$mc_tau_S[burn,])
  sig_T = res$mc_sig_T[burn]; sig_s = res$mc_sig_s[,burn]
  if (intercept) {
    b0_T = res$mc_b0_T[burn]
    b0_s = if (S > 0) res$mc_b0_S[burn, ] else numeric(0)
  } else {
    b0_T = 0
    b0_s = rep(0, S)
  }

  mc.lam = array(NA, dim=c(n_blocks,S+1))
  colnames(mc.lam) <- c("lambda_T", if (S > 0) paste0("lambda_s", seq_len(S)) else NULL)
  warm_start_iter <- 10
  warm_start_N <- 300
  mc_lam_mcem <- array(NA, dim=c(if (warm_start) warm_start_iter else 0, S+1))
  colnames(mc_lam_mcem) <- colnames(mc.lam)

  tail_idx <- seq.int(max(1, burn - 999), burn)
  B_hat_T <- mean(pnorm(res$mc_bt[tail_idx,2*p+1],log.p=T))
  B_hat_s <- if (S > 0) {
    sapply(seq_len(S), function(s) mean(pnorm(res$mc_bs[2*p+1, s, tail_idx], log.p=T)))
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
      res <- ESS_Gibbs_TL_AFT(X_T=X_T, Y_T=Y_T, C_T=C_T,
                              X_s=X_s, Y_s=Y_s, C_s=C_s,
                              bt.c=bt.c, bs.c=bs.c,
                              lambda_T=lambda_T, lambda_s=lambda_s,
                              xi=xi, xi_s=xi_s,
                              sig_T=sig_T, sig_s=sig_s,
                              b0_T=b0_T, b0_s=b0_s, intercept=intercept,
                              N=warm_start_N, S.max=S.max, block_size=block_size,
                              family=family, slab=slab,
                              verbose=0, debug=TRUE)
      bt.c  <- res$mc_bt[warm_start_N, ]
      xi    <- abs(res$mc_tau_T[warm_start_N])
      sig_T <- res$mc_sig_T[warm_start_N]
      if (intercept) b0_T <- res$mc_b0_T[warm_start_N]

      bs.c  <- matrix(res$mc_bs[,,warm_start_N], nrow=2*p+1, ncol=S)
      xi_s  <- abs(res$mc_tau_S[warm_start_N, ])
      sig_s <- res$mc_sig_s[, warm_start_N]
      if (intercept && S > 0) b0_s <- res$mc_b0_S[warm_start_N, ]

      mean_logPhi_T <- mean(pnorm(res$mc_bt[,2*p+1], log.p=TRUE))
      mean_logPhi_s <- if (S > 0) {
        sapply(seq_len(S), function(s) mean(pnorm(res$mc_bs[2*p+1, s,], log.p=TRUE)))
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

  if (verbose==1) pb <- txtProgressBar(min = 0, max = n_blocks, style = 3)
  for(t_block in seq_len(n_blocks)){

    res <- ESS_Gibbs_TL_AFT(X_T=X_T, Y_T=Y_T, C_T=C_T,
                            X_s=X_s, Y_s=Y_s, C_s=C_s,
                            bt.c=bt.c, bs.c=bs.c,
                            lambda_T=lambda_T, lambda_s=lambda_s,
                            xi=xi, xi_s=xi_s,
                            sig_T=sig_T, sig_s=sig_s,
                            b0_T=b0_T, b0_s=b0_s, intercept=intercept,
                            N=K_block, S.max=S.max, block_size=block_size,
                            family=family, slab=slab,
                            verbose=0, debug=TRUE)
    bt.c  <- res$mc_bt[K_block, ]
    xi    <- abs(res$mc_tau_T[K_block])
    sig_T <- res$mc_sig_T[K_block]
    if (intercept) b0_T <- res$mc_b0_T[K_block]

    bs.c  <- matrix(res$mc_bs[,,K_block], nrow=2*p+1, ncol=S)
    xi_s  <- abs(res$mc_tau_S[K_block, ])
    sig_s <- res$mc_sig_s[, K_block]
    if (intercept && S > 0) b0_s <- res$mc_b0_S[K_block, ]

    # SAEM update
    gamma_t <- t_block^(-gamma_power)     # Robbins step size
    mean_logPhi_T  <- mean(pnorm(res$mc_bt[,2*p+1],log.p=T))
    if (lambda_active[1]) {
      B_hat_T      <- (1 - gamma_t) * B_hat_T + gamma_t * mean_logPhi_T
      B_hat_T      <- max(min(B_hat_T, -1e-6), -30)  # clip
    }

    for (s in seq_len(S)) {
      mean_logPhi_s <- mean(pnorm(res$mc_bs[2*p+1, s,], log.p = TRUE))
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

    if (verbose==1) setTxtProgressBar(pb, t_block)
  }
  final_lam <- if (polyak && lam_avg_n > 0) lam_avg else c(lambda_T, lambda_s)
  final_lam[lambda_fixed] <- c(lambda_T, lambda_s)[lambda_fixed]
  out <- list(mc.lam = mc.lam, final_lam = final_lam,
              bt_c = bt.c, bs_c = bs.c,
              xi = xi, xi_s = xi_s,
              sig_T = sig_T, sig_s = sig_s)
  if (intercept) {
    out$b0_T = b0_T
    out$b0_s = b0_s
  }
  out$mc_lam_mcem <- mc_lam_mcem
  return(out)
}
