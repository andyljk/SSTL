#' ESS-within-Gibbs sampler for Bayesian transfer learning of HD linear regression models.
#'
#' Runs elliptical slice sampling updates for target parameters and source biases.
#'
#' @param X_T Target design matrix.
#' @param Y_T Target response.
#' @param X_s List of source design matrices.
#' @param Y_s List of source responses.
#' @param bt.c,bs.c Initial states for target and source parameters.
#' @param xi,xi_s Initial states for target and source scale parameters.
#' @param sig2_T,sig2_s Initial states for target and source variance parameters.
#' @param lambda_T,lambda_s Threshold parameters.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param slab Type of slab distribution to use, includes 'exp', 'poly', 'nlp'. Default is 'exp'.
#' @param verbose Verbosity flag.
#' @param debug Optional returning of MCMC runs other than the coefficient itself.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs_TL <- function(X_T,Y_T,X_s,Y_s,
                         bt.c=NULL, bs.c=NULL,
                         xi=NULL, xi_s=NULL,
                         sig2_T=NULL, sig2_s=NULL,
                         lambda_T=NULL, lambda_s=NULL,
                         N=5000, S.max=500, block_size=1, slab = "exp",
                         verbose=1, debug=F) {

  slab_map <- c("exp" = 1, "poly" = 2, "nlp" = 3, "nlp2" = 4)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'slab', or 'nlp'")

  # b_T, b_s:     Current state (vector) for par of interest and source study biases
  # LL:  Function to compute log-likelihood
  # sd_0:    sd vector of the Gaussian prior
  # N:       Number of iterations
  # S.max    max nr of slice itr for each angle-sampling step

  X_T <- as.matrix(X_T); Y_T <- as.numeric(Y_T)
  X_s <- lapply(X_s, as.matrix); Y_s <- lapply(Y_s, as.numeric)

  p = ncol(X_T); n_t = nrow(X_T)
  S = length(X_s)
  # setup inputs for MCMC
  id <- lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx <- min(start_idx + block_size - 1, p)
    w_idx <- start_idx:end_idx; a_idx <- w_idx + p
    return(c(w_idx, a_idx))
  })
  id <- c(id, list(2*p + 1))

  sd_T = sqrt(c(rep(1, p), rep(1,p), 1))

  if (is.null(bt.c)){
    bt.c = c(rnorm(2*p+1, 0, sd_T)) # current state for target par: [w_T,a_T,a_0]
    bs.c = matrix(rnorm((2*p+1)*S, 0, c(rep(0.05,p),rep(1,p+1))), nrow=2*p+1, ncol=S) # [[w_1^T,a_1^T,a_10^T]^T,..,[w_S^T,a_S^T,a_S0^T]^T]
  }

  if(is.null(lambda_T) | is.null(lambda_s)){
    lambda_T = p^0.5; lambda_s = rep(3,S)
  }


  d    <- length(bt.c)                   # nr of parameters
  K    <- length(id)                     # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  MC.beta = matrix(NA, N, p)
  if (debug){
    N.t  <- matrix(NA, N, K)               # nr of slice sampling itr at each MCMC-itr
    N.s  <- array(NA, dim=c(N, (2*p+1)))   # nr of slice sampling itr at each MCMC-itr
    mc.bt <- matrix(NA, N, d)              # storage for the target parameter
    mc.bs = array(NA, dim=c(2*p+1, S, N))  # storage for the bias parameters
    mc.sig2_T = rep(NA,N); mc.sig2_s = array(NA, dim=c(S,N))
    mc.tau_T = rep(NA,N)
    mc.tau_S = array(NA,dim=c(N,S))
  }

  if (is.null(xi)) xi = 1
  if (is.null(xi_s)) xi_s = rep(0.1,S)
  if (is.null(sig2_T)) sig2_T = 1
  if (is.null(sig2_s)) sig2_s = rep(1,S) # initialize noise variance parameters

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){#  loop over iteration
    # update target parameter
    cpp_res_T <- update_target_cpp(bt_c = bt.c,
                                   X_T = X_T, Y_T = Y_T,
                                   X_S_list = X_s, Y_S_list = Y_s,
                                   bs_c = bs.c,
                                   id = id, sd_T = sd_T,
                                   lambda_T = lambda_T, lambda_S = lambda_s,
                                   tau = abs(xi), tau_S = abs(xi_s),
                                   sd_y_T = sqrt(sig2_T), sd_y_S = sqrt(sig2_s),
                                   S_max = S.max, slab_code = slab_code)
    bt.c <- cpp_res_T$bt_c

    # Update Target Scale
    xi <- update_target_scale_cpp(xi_t_curr = xi,
                                  sd_0 = 10.0,
                                  bt_c = bt.c,
                                  bs_c = bs.c,
                                  X_T = X_T, Y_T = Y_T,
                                  X_s_list = X_s, Y_s_list = Y_s,
                                  lambda_T = lambda_T,
                                  lambda_S = lambda_s,
                                  sd_y_T = sqrt(sig2_T), sd_y_S = sqrt(sig2_s),
                                  tau_S = abs(xi_s),
                                  slab_code = slab_code)
    tau <- abs(xi)

    # update noise variance for target
    beta_Tc = calc_beta(bt.c, lambda_T, abs(xi), p, slab_code)
    sig2_T = 1/rgamma(1, shape = 0.001 + length(Y_T)/2,
                      rate = 0.001 + 0.5*sum((Y_T-X_T%*%beta_Tc)^2))

    MC.beta[i,] = beta_Tc
    if (debug){
      N.t[i,] = cpp_res_T$N_t
      mc.bt[i, ]    <- bt.c                  # Store the sample for target parameter
      mc.sig2_T[i] = sig2_T
      mc.tau_T[i] <- tau
    }


    # Calls optimized C++ function (Joint Row-wise updates)
    cpp_res_S <- update_source_joint_cpp(bs_c = bs.c,
                                         X_s_list = X_s, Y_s_list = Y_s,
                                         beta_T = beta_Tc,
                                         lambda_S = lambda_s,
                                         tau_S = abs(xi_s),
                                         sd_y_S = sqrt(sig2_s),
                                         S_max = S.max,
                                         slab_code = slab_code)
    bs.c <- cpp_res_S$bs_c

    # Update Source Scales (Jointly with Independent Prior)
    xi_s <- update_source_scales_cpp(xi_s_curr = xi_s,
                                     sd_0 = 10.0,
                                     bs_c = bs.c,
                                     X_s_list = X_s, Y_s_list = Y_s,
                                     beta_T = beta_Tc,
                                     lambda_S = lambda_s,
                                     sd_y_S = sqrt(sig2_s),
                                     slab_code = slab_code)
    tau_s <- abs(xi_s)

    # update noise variance for sources
    biases <- calc_bias(bs.c,p,lambda_s,tau_s,slab_code)
    for (s in 1:S){
      beta_sc = biases[,s] + beta_Tc
      sig2_s[s] = 1/rgamma(1, shape = 0.01 + length(Y_s[[s]])/2,
                           rate = 0.01 + 0.5*sum((Y_s[[s]]-X_s[[s]]%*%beta_sc)^2))
      if (debug) mc.sig2_s[s,i] = sig2_s[s]
    }

    if (debug){
      mc.bs[,,i]    <- bs.c                  # Store the sample for source bias
      N.s[i,] = cpp_res_S$N_s
      mc.tau_S[i, ] <- tau_s
    }

    if (verbose==1) setTxtProgressBar(pb, i)
  }
  if (debug){
    return(list(mc.bt=mc.bt, mc.bs=mc.bs,
                MC_beta = MC.beta,
                n.t=N.t, n.s=N.s,
                mc.s2_T = mc.sig2_T, mc.s2_s = mc.sig2_s,
                mc.tau_T = mc.tau_T, mc.tau_S = mc.tau_S))
  }else{
    return(list(MC_beta = MC.beta))
  }
}


#' Empirical Bayes estimation of prior spike probabilities of the SpSL model
#'
#' Run a stochastic EM algorithm
#'
#' @inheritParams ESS_Gibbs_TL
#' @param gamma_power Robbins–Monro step-size exponent for SAEM updates.
#' @param lr Learning rate for lambda updates.
#' @param K_block Block size (iterations) per SAEM update.
#' @param schedule Exponent controlling learning-rate decay (e.g., lr / t^schedule).
#' @return A list containing MCMC draws and lambda trajectories.
#' @export
EB_Gibbs_SAEM = function(X_T,Y_T,X_s,Y_s,
                         bt.c=NULL, bs.c=NULL,
                         xi=NULL, xi_s=NULL,
                         N=5000, S.max=500, block_size=1,
                         gamma_power = 0.9,
                         lr = 0.1,
                         K_block = 10,
                         schedule=0.5, slab = "exp",
                         verbose=1){

  slab_map <- c("exp" = 1, "poly" = 2, "nlp" = 3, "nlp2" = 4)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'slab', or 'nlp'")

  # b_T, b_s:     Current state (vector) for par of interest and source study biases
  # LL:  Function to compute log-likelihood
  # sd_0:    sd vector of the Gaussian prior
  # N:       Number of iterations
  # S.max    max nr of slice itr for each angle-sampling step

  X_T <- as.matrix(X_T); Y_T <- as.numeric(Y_T)
  X_s <- lapply(X_s, as.matrix); Y_s <- lapply(Y_s, as.numeric)

  p = ncol(X_T); n_t = nrow(X_T)
  S = length(X_s)
  # setup inputs for MCMC
  id <- lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx <- min(start_idx + block_size - 1, p)
    w_idx <- start_idx:end_idx; a_idx <- w_idx + p
    return(c(w_idx, a_idx))
  })
  id <- c(id, list(2*p + 1))

  sd_T = sqrt(c(rep(1, p), rep(1,p), 1))
  if (is.null(bt.c)){
    bt.c = c(rnorm(2*p+1, 0, sd_T)) # current state for target par: [w_T,a_T,a_0]
    bs.c = matrix(rnorm((2*p+1)*S, 0, c(rep(0.05,p),rep(1,p+1))), nrow=2*p+1, ncol=S) # [[w_1^T,a_1^T,a_10^T]^T,..,[w_S^T,a_S^T,a_S0^T]^T]
  }
  if (is.null(xi)) xi = 1; if (is.null(xi_s)) xi_s = rep(0.1,S)

  d    <- length(bt.c)                   # nr of parameters
  K    <- length(id)                     # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  mc.bt <- matrix(NA, N, d)              # storage for the target parameter
  mc.bs = array(NA, dim=c(2*p+1, S, N))  # storage for the bias parameters
  if (is.null(xi)) xi = 1
  if (is.null(xi_s)) xi_s = rep(0.1,S)
  mc.lam = array(NA, dim=c(N/K_block,S+1))

  sig2_T = 1; sig2_s = rep(1,S) # initialize noise variance parameters

  lambda_T <- 9
  lambda_s <- rep(3, S)
  B_hat_T <- -1
  B_hat_s <- rep(-1, S)

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){
    # update target parameter
    cpp_res_T <- update_target_cpp(bt_c = bt.c,
                                   X_T = X_T, Y_T = Y_T,
                                   X_S_list = X_s, Y_S_list = Y_s,
                                   bs_c = bs.c,
                                   id = id, sd_T = sd_T,
                                   lambda_T = lambda_T, lambda_S = lambda_s,
                                   tau = abs(xi), tau_S = abs(xi_s),
                                   sd_y_T = sqrt(sig2_T), sd_y_S = sqrt(sig2_s),
                                   S_max = S.max, slab_code = slab_code)
    bt.c <- cpp_res_T$bt_c

    mc.bt[i, ]    <- bt.c                  # Store the sample for target parameter

    # update noise variance for target
    beta_Tc = calc_beta(bt.c, lambda_T, abs(xi), p, slab_code)
    sig2_T = 1/rgamma(1, shape = 0.001 + length(Y_T)/2,
                      rate = 0.001 + 0.5*sum((Y_T-X_T%*%beta_Tc)^2))

    # Update Target Scale
    xi <- update_target_scale_cpp(xi_t_curr = xi,
                                  sd_0 = 2.0,
                                  bt_c = bt.c,
                                  bs_c = bs.c,
                                  X_T = X_T, Y_T = Y_T,
                                  X_s_list = X_s, Y_s_list = Y_s,
                                  lambda_T = lambda_T,
                                  lambda_S = lambda_s,
                                  sd_y_T = sqrt(sig2_T), sd_y_S = sqrt(sig2_s),
                                  tau_S = abs(xi_s),
                                  slab_code = slab_code)
    tau <- abs(xi)

    # Calls optimized C++ function (Joint Row-wise updates)
    cpp_res_S <- update_source_joint_cpp(bs_c = bs.c,
                                         X_s_list = X_s, Y_s_list = Y_s,
                                         beta_T = beta_Tc,
                                         lambda_S = lambda_s,
                                         tau_S = abs(xi_s),
                                         sd_y_S = sqrt(sig2_s),
                                         S_max = S.max,
                                         slab_code = slab_code)
    bs.c <- cpp_res_S$bs_c
    mc.bs[,,i]    <- bs.c                  # Store the sample for source bias

    # Update Source Scales (Jointly with Independent Prior)
    xi_s <- update_source_scales_cpp(xi_s_curr = xi_s,
                                     sd_0 = 2.0,
                                     bs_c = bs.c,
                                     X_s_list = X_s, Y_s_list = Y_s,
                                     beta_T = beta_Tc,
                                     lambda_S = lambda_s,
                                     sd_y_S = sqrt(sig2_s),
                                     slab_code = slab_code)
    tau_s <- abs(xi_s)

    # update noise variance for sources
    biases <- calc_bias(bs.c,p,lambda_s,tau_s,slab_code)
    for (s in 1:S){
      beta_sc = biases[,s] + beta_Tc
      sig2_s[s] = 1/rgamma(1, shape = 0.01 + length(Y_s[[s]])/2,
                           rate = 0.01 + 0.5*sum((Y_s[[s]]-X_s[[s]]%*%beta_sc)^2))
    }

    if (i%%K_block==0){

      t_block = i/K_block
      lr_t = lr / (t_block^schedule)
      gamma_t <- t_block^(-gamma_power)     # Robbins step size
      zT_block       <- mc.bt[(i-K_block+1):i, 2*p+1]
      mean_logPhi_T  <- mean(log(pnorm(zT_block)))
      B_hat_T        <- (1 - gamma_t) * B_hat_T + gamma_t * mean_logPhi_T
      B_hat_T        <- max(min(B_hat_T, -0.3), -3)  # clip

      lambda_T_target  <- -lambda_T / B_hat_T
      lambda_T       <- max(1e-6, (1 - lr_t)*lambda_T + lr_t*lambda_T_target)

      for (s in 1:S) {
        zS_block          <- mc.bs[2*p+1, s, (i-K_block+1):i]
        mean_logPhi_s     <- mean(log(pnorm(zS_block)))
        B_hat_s[s]        <- (1 - gamma_t) * B_hat_s[s] + gamma_t * mean_logPhi_s
        B_hat_s[s]        <- max(min(B_hat_s[s], -0.3), -3)

        lambda_s_target     <- -lambda_s[s] / B_hat_s[s]
        lambda_s[s]       <- max(1e-1, (1 - lr_t)*lambda_s[s] + lr_t*lambda_s_target)
      }
      mc.lam[t_block, ] <- c(lambda_T, lambda_s)
    }
    if (verbose==1) setTxtProgressBar(pb, i)
  }
  return(list(mc.lam = mc.lam, final_lam = c(lambda_T,lambda_s),
              bt_c = bt.c, bs_c = bs.c,
              xi = xi, xi_s = xi_s,
              sig2_T = sig2_T, sig2_s = sig2_s))
}
