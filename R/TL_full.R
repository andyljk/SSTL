
# elliptical slice sampling within gibbs function
#'
#' @param X_T Target design matrix.
#' @param Y_T Target response.
#' @param X_s List of source design matrices.
#' @param Y_s List of source responses.
#' @param bt.c,bs.c Initial states for target and source parameters.
#' @param sd_T Prior SD vector for target latent parameters.
#' @param cov_W Prior covariance for source weight correlation.
#' @param lambda_T,lambda_s Threshold parameters.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param verbose Verbosity flag.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs_TL <- function(X_T,Y_T,X_s,Y_s,
                         bt.c=NULL, bs.c=NULL,
                         sd_T=NULL, cov_W=NULL,
                         lambda_T=NULL, lambda_s=NULL,
                         N=5000, S.max=500, block_size=3,
                         verbose=1) {
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

  if (is.null(sd_T)) sd_T = sqrt(c(rep(0.15, p), rep(1,p), 1))

  if (is.null(bt.c)){
    bt.c = c(rnorm(2*p+1, 0, sd_T)) # current state for target par: [w_T,a_T,a_0]
    bs.c = matrix(rnorm((2*p+1)*S, 0, c(rep(0.05,p),rep(1,p+1))), nrow=2*p+1, ncol=S) # [[w_1^T,a_1^T,a_10^T]^T,..,[w_S^T,a_S^T,a_S0^T]^T]
  }

  # same prior variances for latent variables in all data sources
  if (is.null(cov_W)) cov_W = diag(0.05^2, S)

  W_0 = diag(0.05^2, S); v_0 = S+1

  if(is.null(lambda_T) | is.null(lambda_s)){
    lambda_T = p^0.5; lambda_s = rep(3,S)
  }


  d    <- length(bt.c)                   # nr of parameters
  K    <- length(id)                     # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  N.t  <- matrix(NA, N, K)               # nr of slice sampling itr at each MCMC-itr
  N.s  <- array(NA, dim=c(N, (2*p+1)))   # nr of slice sampling itr at each MCMC-itr
  mc.bt <- matrix(NA, N, d)              # storage for the target parameter
  mc.bs = array(NA, dim=c(2*p+1, S, N))  # storage for the bias parameters
  mc.W = array(NA, dim=c(S,S,N))         # storage for covariances
  mc.sig2_T = rep(NA,N); mc.sig2_s = array(NA, dim=c(S,N))
  mc.tau2_wT = rep(NA,N)

  sig2_T = 1; sig2_s = rep(1,S) # initialize noise variance parameters

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){#  loop over iteration
    # update target parameter
    cpp_res_T <- update_target_cpp(bt_c = bt.c,
                                   X_T = X_T, Y_T = Y_T,
                                   X_S_list = X_s, Y_S_list = Y_s,
                                   bs_c = bs.c,
                                   id = id, sd_T = sd_T,
                                   lambda_T = lambda_T, lambda_S = lambda_s,
                                   sd_y_T = sqrt(sig2_T), sd_y_S = sqrt(sig2_s),
                                   S_max = S.max)
    bt.c <- cpp_res_T$bt_c
    N.t[i,] = cpp_res_T$N_t

    mc.bt[i, ]    <- bt.c                  # Store the sample for target parameter

    # update noise variance for target
    beta_Tc = beta_Tc <- beta(bt.c, lambda_T, p)
    sig2_T = 1/rgamma(1, shape = 0.001 + length(Y_T)/2,
                      rate = 0.001 + 0.5*sum((Y_T-X_T%*%beta_Tc)^2))
    mc.sig2_T[i] = sig2_T

    # update prior variance of w in target
    tau2_w = 1/rgamma(1, shape = 3 + p/2, 2 + sum(bt.c[1:p]^2)/2)
    sd_T[1:p] = tau2_w^0.5; mc.tau2_wT[i] = tau2_w

    # update source biases
    # Pre-calculate Cholesky for C++ (Lower Triangular)
    chol_W <- t(chol(cov_W))

    # Calls optimized C++ function (Joint Row-wise updates)
    cpp_res_S <- update_source_joint_cpp(bs_c = bs.c,
                                         X_s_list = X_s, Y_s_list = Y_s,
                                         beta_T = beta_Tc,
                                         chol_cov_W = chol_W,
                                         lambda_S = lambda_s,
                                         sd_y_S = sqrt(sig2_s),
                                         S_max = S.max)
    bs.c <- cpp_res_S$bs_c
    mc.bs[,,i]    <- bs.c                  # Store the sample for source bias
    N.s[i,] = cpp_res_S$N_s

    # draw conditional covariances for w's in source
    term_b = matrix(bs.c[1:p,], nrow=p)                            # handle edge case when S=1
    cov_W <- MCMCpack::riwish(v_0 + p, t(term_b) %*% term_b + W_0)
    mc.W[,,i] = cov_W

    # update noise variance for sources
    biases <- calc_bias(bs.c,p,lambda_s)
    for (s in 1:S){
      beta_sc = biases[,s] + beta_Tc
      sig2_s[s] = 1/rgamma(1, shape = 0.01 + length(Y_s[[s]])/2,
                           rate = 0.01 + 0.5*sum((Y_s[[s]]-X_s[[s]]%*%beta_sc)^2))
      mc.sig2_s[s,i] = sig2_s[s]
    }

    if (verbose==1) setTxtProgressBar(pb, i)
  }

  # calculate beta
  MC.beta  = t(apply(mc.bt, 1, function(b) b[1:p] * T.n(b[(p+1):(2*p)] - a0_star(b[2*p+1], lambda_T) )  ))
  MC.alp = t(apply(mc.bt, 1, function(b) T.n(b[(p+1):(2*p)] - a0_star(b[2*p+1], lambda_T) )  ))

  return(list(MC_beta = MC.beta,
              MC_alpha = MC.alp,
              mc.bt=mc.bt,
              mc.bs=mc.bs,
              n.t=N.t, n.s=N.s,
              mc.s2_T = mc.sig2_T, mc.s2_s = mc.sig2_s,
              mc.tau2_wT = mc.tau2_wT, mc.covW = mc.W,
              lambda=c(lambda_T,lambda_s))
  )
}

# stochastic version
#'
#' @inheritParams ESS_Gibbs_TL
#' @return A list containing MCMC draws and lambda trajectories.
#' @export
EB_Gibbs_SAEM = function(X_T,Y_T,X_s,Y_s,
                         bt.c=NULL, bs.c=NULL,
                         sd_T=NULL, cov_W=NULL,
                         N=5000, S.max=500, block_size=3,
                         gamma_power = 0.9,
                         lr = 0.1,
                         K_block = 10,
                         schedule=0.5,
                         verbose=1){
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

  if (is.null(sd_T)) sd_T = sqrt(c(rep(0.15, p), rep(1,p), 1))

  if (is.null(bt.c)){
    bt.c = c(rnorm(2*p+1, 0, sd_T)) # current state for target par: [w_T,a_T,a_0]
    bs.c = matrix(rnorm((2*p+1)*S, 0, c(rep(0.05,p),rep(1,p+1))), nrow=2*p+1, ncol=S) # [[w_1^T,a_1^T,a_10^T]^T,..,[w_S^T,a_S^T,a_S0^T]^T]
  }

  # same prior variances for latent variables in all data sources
  if (is.null(cov_W)) cov_W = diag(0.05^2, S)

  W_0 = diag(0.05^2, S); v_0 = S+1

  d    <- length(bt.c)                   # nr of parameters
  K    <- length(id)                     # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  v    <- sapply(id, length)             # length of each block b.j
  N.t  <- matrix(NA, N, K)               # nr of slice sampling itr at each MCMC-itr
  N.s  <- array(NA, dim=c(N, (2*p+1)))   # nr of slice sampling itr at each MCMC-itr
  mc.bt <- matrix(NA, N, d)              # storage for the target parameter
  mc.bs = array(NA, dim=c(2*p+1, S, N))  # storage for the bias parameters
  mc.W = array(NA, dim=c(S,S,N))         # storage for covariances
  mc.sig2_T = rep(NA,N); mc.sig2_s = array(NA, dim=c(S,N))
  mc.tau2_wT = rep(NA,N)
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
                                   sd_y_T = sqrt(sig2_T), sd_y_S = sqrt(sig2_s),
                                   S_max = S.max)
    bt.c <- cpp_res_T$bt_c

    mc.bt[i, ]    <- bt.c                  # Store the sample for target parameter

    # update noise variance for target
    beta_Tc = beta_Tc <- beta(bt.c, lambda_T, p)
    sig2_T = 1/rgamma(1, shape = 0.001 + length(Y_T)/2,
                      rate = 0.001 + 0.5*sum((Y_T-X_T%*%beta_Tc)^2))
    mc.sig2_T[i] = sig2_T

    # update prior variance of w in target
    tau2_w = 1/rgamma(1, shape = 3 + p/2, 2 + sum(bt.c[1:p]^2)/2)
    sd_T[1:p] = tau2_w^0.5; mc.tau2_wT[i] = tau2_w

    # update source biases
    # Pre-calculate Cholesky for C++ (Lower Triangular)
    chol_W <- t(chol(cov_W))

    # Calls optimized C++ function (Joint Row-wise updates)
    cpp_res_S <- update_source_joint_cpp(bs_c = bs.c,
                                         X_s_list = X_s, Y_s_list = Y_s,
                                         beta_T = beta_Tc,
                                         chol_cov_W = chol_W,
                                         lambda_S = lambda_s,
                                         sd_y_S = sqrt(sig2_s),
                                         S_max = S.max)
    bs.c <- cpp_res_S$bs_c
    mc.bs[,,i]    <- bs.c                  # Store the sample for source bias

    # draw conditional covariances for w's in source
    term_b = matrix(bs.c[1:p,], nrow=p)                            # handle edge case when S=1
    cov_W <- MCMCpack::riwish(v_0 + p, t(term_b) %*% term_b + W_0)
    mc.W[,,i] = cov_W

    # update noise variance for sources
    biases <- calc_bias(bs.c,p,lambda_s)
    for (s in 1:S){
      beta_sc = biases[,s] + beta_Tc
      sig2_s[s] = 1/rgamma(1, shape = 0.01 + length(Y_s[[s]])/2,
                           rate = 0.01 + 0.5*sum((Y_s[[s]]-X_s[[s]]%*%beta_sc)^2))
      mc.sig2_s[s,i] = sig2_s[s]
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
  return(list(mc.bt=mc.bt, mc.bs=mc.bs, n.t=N.t, n.s=N.s,
              mc.s2_T = mc.sig2_T, mc.s2_s = mc.sig2_s,
              mc.tau2_wT = mc.tau2_wT, mc.covW = mc.W,
              mc.lam = mc.lam, final_lam = c(lambda_T,lambda_s)))
}


calc_bias <- function(mat,p,lam_s){
  w_mat = mat[1:p, , drop=FALSE]
  alp_mat = mat[(p+1):(2*p), , drop=FALSE]
  thres_vec = rep(a0_star(mat[2*p + 1, ], lam_s), each = p)
  thres_mat = matrix(thres_vec, nrow = p, ncol = ncol(mat))
  return(w_mat * pmax(alp_mat - thres_mat, 0))
}
# mat[1:p, ] * pmax(mat[(p+1):(2*p), ] - rep(a0_star(mat[2*p + 1, ],lam_s), each = p), 0)

a0_star <- function(a0_raw,lam) {
  # This calculates a_0* = Phi_inv( Phi(a0_raw)^(1/p) )
  return(qnorm(pnorm(a0_raw)^(1/lam)))
}

T.n   = function(b) pmax(b, 0)
beta  = function(b,lambda,p) b[1:p]*T.n(b[(p+1):(2*p)]-qnorm(pnorm(b[2*p+1])^(1/lambda)))


# function to generate correlated covariates
#'
#' Generates an \(n \times p\) design matrix with AR(1) correlation
#' structure among covariates.
#'
#' @param n Number of observations.
#' @param p Number of covariates.
#' @param rho AR(1) correlation parameter (must be in (-1, 1)).
#'
#' @return A numeric matrix of dimension n x p.
#' @export
Gen_AR1 <- function(n,p,rho) {
  if (abs(rho) >= 1) stop("rho must be in (-1, 1).")
  Sigma  <- toeplitz(rho^(0:(p -1)))          # AR(1) covariance
  mvnfast::rmvn(n, mu=rep(0,p), sigma=Sigma)
}

#' Simulate Target and Source Data for Transfer Learning
#'
#' Simulates one target dataset and multiple source datasets for
#' high-dimensional regression transfer learning experiments.
#'
#' @param p Number of covariates.
#' @param n_t Target sample size.
#' @param n_s Source sample size (per source).
#' @param S Number of source studies.
#' @param sparse_level Proportion of non-zero coefficients in the target.
#' @param effect_size Signal strength of non-zero coefficients.
#' @param bias_level Magnitude of source-specific bias.
#'
#' @return A list with elements:
#' \itemize{
#'   \item X_T: Target design matrix
#'   \item Y_T: Target response vector
#'   \item X_s: List of source design matrices
#'   \item Y_s: List of source response vectors
#'   \item b_T: True target coefficient vector
#'   \item b_s: True source coefficient matrix
#' }
#' @export
sim_data = function(p, n_t, n_s, S, info_set = round(S/2),
                    sparse_level=0.1, effect_size=0.5, X_cor=0.5,
                    bias_level=5, bad_bias = 5){
  p_0   = round(p*sparse_level) # non-zero true coefficients
  sd_y = 1
  good = c(1:info_set)
  b_T    = c(rep(effect_size,p_0), rep(0, p-p_0)) # true parameter, first 5 nonzero, the rest 100 are zero
  X_T    = Gen_AR1(n_t,p,rho=X_cor) # AR1 correlated covars
  Y_T    = X_T%*%b_T + rnorm(n_t,0,sd_y) # true response data

  b_s = array(NA,dim=c(p,S))
  X_s = vector(mode="list",length=S)
  Y_s = vector(mode="list",length=S)

  # generate source data
  for (s in 1:S){
    b_s[,s] = b_T + bad_bias
    if (s %in% good) b_s[,s] = b_T + rnorm(p,0,bias_level/p)
    X_s[[s]] = Gen_AR1(n_s,p,rho=X_cor) # AR1
    Y_s[[s]] = X_s[[s]] %*% b_s[,s] + rnorm(n_s,0,1)
  }
  return(list(X_T=X_T,
              Y_T=Y_T,
              X_s=X_s,
              Y_s=Y_s,
              b_T=b_T,
              b_s=b_s))
}

