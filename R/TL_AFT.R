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
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param family Specification of outcome model, one of 'Weibull', 'Lognormal', 'Loglogistic'. Default is 'Weibull'.
#' @param slab Specification of slab type, one of 'exp', 'poly', 'nlp'. Default is 'exp'.
#' @param verbose Verbosity flag.
#' @param debug Optional returning of MCMC runs other than the coefficient itself.
#' @return A list containing MCMC draws and diagnostics.
#' @export
# elliptical slice sampling within gibbs function
ESS_Gibbs_TL_AFT <- function(X_T, Y_T, C_T=NULL, # Target Data
                             X_s, Y_s, C_s=NULL, # Source Data (Optional)
                             bt.c=NULL, bs.c=NULL,
                             lambda_T=NULL, lambda_s=NULL,
                             xi=NULL, xi_s=NULL,
                             sig_T = NULL, sig_s = NULL,
                             N=5000, S.max=500, block_size=1,
                             family="Weibull", slab = "poly",
                             verbose=1, debug=F) {

  fam_map <- c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
  fam_code <- fam_map[tolower(family)]
  if(is.na(fam_code)) stop("Family must be 'weibull', 'loglogistic', or 'lognormal'")

  slab_map <- c("exp" = 1, "poly" = 2, "nlp1" = 3, "nlp2" = 4)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'slab', or 'nlp1/nlp2'")

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
  if (debug){
    N.t  <- matrix(NA, N, K)               # nr of slice sampling itr at each MCMC-itr
    N.s  <- array(NA, dim=c(N, (2*p+1)))   # nr of slice sampling itr at each MCMC-itr
    mc.bt <- matrix(NA, N, d)              # storage for the target parameter
    mc.bs = array(NA, dim=c(2*p+1, S, N))  # storage for the bias parameters
    mc.sig_T = rep(NA,N); mc.sig_s = array(NA, dim=c(S,N))
    mc.tau_T = rep(NA,N)
    mc.tau_S = array(NA,dim=c(N,S))
  }
  if (is.null(xi)) xi = 1
  if (is.null(xi_s)) xi_s = rep(0.1,S)
  if (is.null(sig_T)) sig_T = 1
  if (is.null(sig_s)) sig_s = rep(1,S) # initialize scale parameters


  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    # ---------------------------------------------------------
    # STEP A: Update Target (Always Runs)
    # ---------------------------------------------------------
    # Note: C++ function naturally handles S=0 (loops over sources won't run)
    cpp_res_T <- update_target_aft(bt_c = bt.c,
                                   X_T = X_T, Y_T = Y_T, C_T = C_T,
                                   X_S_list = X_s, Y_S_list = Y_s, C_S_list = C_s,
                                   bs_c = bs.c,
                                   id = id, sd_T = sd_T,
                                   lambda_T = lambda_T, lambda_S = lambda_s,
                                   tau = abs(xi), tau_S = abs(xi_s),
                                   sd_y_T = sig_T, sd_y_S = sig_s,
                                   S_max = S.max, fam_code=fam_code, slab_code=slab_code)
    bt.c <- cpp_res_T$bt_c

    # Update Target Scale
    xi <- update_target_scale_aft(xi_t_curr = xi,
                                  sd_0 = 2.0,
                                  bt_c = bt.c,
                                  bs_c = bs.c,
                                  X_T = X_T, Y_T = Y_T, C_T = C_T,
                                  X_s_list = X_s, Y_s_list = Y_s, C_s_list = C_s,
                                  lambda_T = lambda_T, lambda_S = lambda_s,
                                  sd_y_T = sig_T, sd_y_S = sig_s,
                                  tau_S = abs(xi_s),
                                  fam_code=fam_code, slab_code = slab_code)
    tau <- abs(xi)

    # update prior variance of w in target
    # tau2_w = 1/rgamma(1, shape = 3 + p/2, 2 + sum(bt.c[1:p]^2)/2)
    # sd_T[1:p] = tau2_w^0.5; mc.tau2_wT[i] = tau2_w

    # update scale parameter for target
    sig_T = update_sigma_target_tl_aft(bt.c, X_T, Y_T, C_T,
                                       sig_T, lambda_T, abs(xi),
                                       fam_code=fam_code, slab_code=slab_code, step_size=0.1)

    if (debug){
      mc.bt[i, ] <- bt.c
      N.t[i,] = cpp_res_T$N_t
      mc.tau_T[i] <- tau
      mc.sig_T[i] = sig_T
    }



    # ---------------------------------------------------------
    # STEP B: Update Sources (Run ONLY if S > 0)
    # ---------------------------------------------------------
    if (S > 0){
      beta_Tc = calc_beta(bt.c, lambda_T, abs(xi), p, slab_code)

      cpp_res_S <- update_source_joint_aft(bs_c = bs.c,
                                           X_s_list = X_s, Y_s_list = Y_s, C_s_list = C_s,
                                           beta_T = beta_Tc,
                                           lambda_S = lambda_s, tau_S = abs(xi_s),
                                           sd_y_S = sig_s,
                                           S_max = S.max,
                                           fam_code=fam_code, slab_code=slab_code)
      bs.c <- cpp_res_S$bs_c

      # Update Source Scales (Jointly with Independent Prior)
      xi_s <- update_source_scales_aft(xi_s_curr = xi_s,
                                       sd_0 = 2.0,
                                       bs_c = bs.c,
                                       X_s_list = X_s, Y_s_list = Y_s, C_s_list = C_s,
                                       beta_T = beta_Tc,
                                       lambda_S = lambda_s,
                                       sd_y_S = sig_s,
                                       fam_code = fam_code, slab_code = slab_code)
      tau_s <- abs(xi_s)

      # update scale parameter for sources
      beta_Tc_curr <- calc_beta(bt.c, lambda_T, abs(xi), p, slab_code)
      for (s in 1:S){
        bs_col <- bs.c[, s] # Extract column for source s
        sig_s[s] <- update_sigma_source_tl_aft(bs_col, beta_Tc_curr,
                                               X_s[[s]], Y_s[[s]], C_s[[s]],
                                               sig_s[s], lambda_s[s], abs(xi_s[s]),
                                               fam_code=fam_code, slab_code=slab_code,
                                               step_size=0.1)
        if (debug) mc.sig_s[s, i] <- sig_s[s]
      }

      if (debug){
        mc.bs[,,i] <- bs.c
        N.s[i,] = cpp_res_S$N_s
        mc.tau_S[i, ] <- tau_s
      }

      MC.beta[i,] = beta_Tc_curr
    }

    if (verbose==1) setTxtProgressBar(pb, i)
  }
  if (debug){
    return(list(mc_bt=mc.bt, mc_bs=mc.bs, MC_beta = MC.beta, n_t=N.t, n_s=N.s,
                mc_sig_T = mc.sig_T, mc_sig_s = mc.sig_s,
                mc_tau_T = mc.tau_T, mc_tau_S = mc.tau_S))
  }else{
    return(list(MC_beta = MC.beta))
  }
}

#' stochastic version
#'
#' @inheritParams ESS_Gibbs_TL_AFT
#' @param gamma_power Robbins–Monro step-size exponent for SAEM updates.
#' @param lr Learning rate for lambda updates.
#' @param K_block Block size (iterations) per SAEM update.
#' @param schedule Exponent controlling learning-rate decay (e.g., lr / t^schedule).
#' @param use_approx,k_start,k_end,approx_burnin Temporary parameters that might be eliminated in final package.
#' @return A list containing MCMC draws and lambda trajectories.
#' @export
EB_SAEM_TL_AFT = function(X_T, Y_T, C_T=NULL, # Target Data
                          X_s, Y_s, C_s=NULL, # Source Data (Optional)
                          bt.c=NULL, bs.c=NULL,
                          lambda_T=NULL, lambda_s=NULL,
                          xi=NULL, xi_s=NULL,
                          N=5000, burn = 1000,
                          S.max=500, block_size=1,
                          family="Weibull", slab = "poly",
                          gamma_power = 0.9,
                          lr = 0.1, K_block = 10, schedule=0.5,
                          use_approx=F, k_start=2.0, k_end=50.0, approx_burnin=0.5,
                          verbose=1){
  # b_T, b_s:     Current state (vector) for par of interest and source study biases
  # LL:  Function to compute log-likelihood
  # sd_0:    sd vector of the Gaussian prior
  # N:       Number of iterations
  # S.max    max nr of slice itr for each angle-sampling step

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

  # run burn in
  if (is.null(lambda_T)) lambda_T <- 9
  if (is.null(lambda_s)) lambda_s <- rep(3, S)

  res <- ESS_Gibbs_TL_AFT(X_T=X_T, Y_T=Y_T, C_T=C_T,
                          X_s=X_s, Y_s=Y_s, C_s=C_s,
                          lambda_T=lambda_T, lambda_s=lambda_s,
                          N=burn, S.max=S.max, block_size=block_size,
                          family=family, slab=slab,
                          verbose=0, debug=T)

  # Initialize the parameters
  bt.c = res$mc_bt[burn,]
  bs.c = matrix(res$mc_bs[,,burn], nrow=2*p+1, ncol=S)
  xi = res$mc_tau_T[burn]; xi_s = res$mc_tau_S[burn,]
  sig_T = res$mc_sig_T[burn]; sig_s = res$mc_sig_s[,burn] # initialize scale parameters

  d    <- length(bt.c)                   # nr of parameters
  K    <- length(id)                     # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  mc.bt <- matrix(NA, N, d)              # storage for the target parameter
  mc.bs = array(NA, dim=c(2*p+1, S, N))  # storage for the bias parameters
  mc.lam = array(NA, dim=c(N/K_block,S+1))

  B_hat_T <- mean(pnorm(res$mc_bt[(burn-1000):burn,2*p+1],log=T))
  B_hat_s <- sapply(1:S, function(s) mean(pnorm(res$mc_bs[2*p+1, s, (burn-1000):burn], log=T)))

  burnin_iters <- floor(N * approx_burnin)

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N/K_block, style = 3)
  for(t_block in 1:(N/K_block)){
    current_approx <- F
    current_k <- 10.0

    if (use_approx && slab_code %in% c(3, 4)) {
      if (i <= burnin_iters) {
        current_approx <- T
        # Linearly step up k to sharpen the threshold as iterations progress
        current_k <- k_start + (k_end - k_start) * (i / burnin_iters)
      }
    }

    res <- ESS_Gibbs_TL_AFT(X_T=X_T, Y_T=Y_T, C_T=C_T,
                            X_s=X_s, Y_s=Y_s, C_s=C_s,
                            bt.c=bt.c, bs.c=bs.c,
                            lambda_T=lambda_T, lambda_s=lambda_s,
                            xi=xi, xi_s=xi_s,
                            sig_T=sig_T, sig_s=sig_s,
                            N=K_block, S.max=S.max, block_size=block_size,
                            family=family, slab=slab,
                            verbose=0, debug=TRUE)
    bt.c  <- res$mc_bt[K_block, ]
    xi    <- res$mc_tau_T[K_block]
    sig_T <- res$mc_sig_T[K_block]

    bs.c  <- matrix(res$mc_bs[,,K_block], nrow=2*p+1, ncol=S)
    xi_s  <- res$mc_tau_S[K_block, ]
    sig_s <- res$mc_sig_s[, K_block]

    # SAEM update
    lr_t = lr / (t_block^schedule)
    gamma_t <- t_block^(-gamma_power)     # Robbins step size
    zT_block       <- res$mc_bt[,2*p+1]
    mean_logPhi_T  <- mean(pnorm(zT_block,log.p=T))
    B_hat_T        <- (1 - gamma_t) * B_hat_T + gamma_t * mean_logPhi_T
    B_hat_T        <- max(min(B_hat_T, -1e-6), -3)  # clip

    lambda_T_target  <- -lambda_T / B_hat_T
    lambda_T       <- max(1e-6, (1 - lr_t)*lambda_T + lr_t*lambda_T_target)

    for (s in 1:S) {
      zS_block          <- res$mc_bs[2*p+1, s,]
      mean_logPhi_s     <- mean(pnorm(zS_block,log.p=T))
      B_hat_s[s]        <- (1 - gamma_t) * B_hat_s[s] + gamma_t * mean_logPhi_s
      B_hat_s[s]        <- max(min(B_hat_s[s], -0.3), -3)

      lambda_s_target     <- -lambda_s[s] / B_hat_s[s]
      lambda_s[s]       <- max(1e-1, (1 - lr_t)*lambda_s[s] + lr_t*lambda_s_target)
    }
    mc.lam[t_block, ] <- c(lambda_T, lambda_s)

    if (verbose==1) setTxtProgressBar(pb, t_block)
  }
  return(list(mc.lam = mc.lam, final_lam = c(lambda_T,lambda_s),
              bt_c = bt.c, bs_c = bs.c,
              xi = xi, xi_s = xi_s,
              sig_T = sig_T, sig_s = sig_s))
}
