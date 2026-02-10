#' ESS-within-Gibbs sampler for single source Bayesian HD linear regression models.
#'
#' Runs elliptical slice sampling updates for target parameters and source biases.
#'
#' @importFrom truncnorm rtruncnorm
#' @param X Design matrix.
#' @param Y Response vector.
#' @param b.c Initial states for latent parameters.
#' @param sd.0 Prior SD vector for target latent parameters.
#' @param lambda Threshold parameter.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param slab Specification of the slab distribution. One of 'exp', 'poly', 'nlp'. Default is 'exp'.
#' @param verbose Verbosity flag.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs <- function(X,Y,b.c=NULL, sd.0=NULL, lambda=NULL, tau=NULL,
                      N=5000, block_size=1, S.max=500, slab = "poly", verbose=1) {

  slab_map <- c("exp" = 1, "poly" = 2, "nlp1" = 3, "nlp2" = 4)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'slab', or 'nlp1/nlp2'")

  p = ncol(X)
  if (is.null(sd.0)) sd.0  = sqrt(c(rep(1, p), rep(1,p), 1))
  if (is.null(b.c)) b.c = rnorm(2*p+1,0,sd.0)

  id = lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx = min(start_idx + block_size - 1, p)
    w_idx = start_idx:end_idx; a_idx = w_idx + p
    return(c(w_idx, a_idx))
  })
  id = c(id, list(2*p + 1))
  if (is.null(lambda)) lambda = p^0.5

  # b.c:     Current state (vector)
  # LL.blg:  Function to compute log-likelihood
  # sd.0:    sd vector of the Gaussian prior
  # N:       Number of iterations
  # S.max    max nr of slice itr for each angle-sampling step

  d    <- length(b.c)                   # nr of parameters
  K    <- length(id)                    # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  v    <- sapply(id, length)            # length of each block b.j
  N.s  <- matrix(NA, N, K)              # nr of slice sampling itr at each MCMC-itr
  mc.b <- matrix(NA, N, d)              # storage
  MC.beta = MC.alpha = matrix(NA, N, p)
  mc.tau = rep(NA,N); if (is.null(tau)) tau = 1
  mc.var = rep(NA,N); sd_y = 1

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    cpp_res <- update_blocks_cpp(b_c = b.c,
                                 X = X,
                                 Y = Y,
                                 id = id,
                                 sd_0 = sd.0,
                                 lambda = lambda,
                                 tau = tau,
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 slab_code = slab_code)
    b.c <- cpp_res$b_c
    N.s[i, ] <- cpp_res$N_s

    eta_c = X%*%calc_beta(b.c,lambda,tau,p,slab_code)
    # update noise variance
    var_y = 1/rgamma(1, shape=0.01 + 0.5*length(Y), 0.01 + 0.5*sum((Y-eta_c)^2))
    sd_y = var_y^0.5; mc.var[i] = var_y

    # update scale parameter for w
    tau = rtruncnorm(1,a=0,b=Inf,
                     mean=1/sd_y^2*sum(Y*eta_c)*(1/2 + 1/sd_y^2 * sum(eta_c^2))^(-1),
                     sd=(1/2 + 1/sd_y^2 * sum(eta_c^2))^(-1))
    mc.tau[i] = tau

    mc.b[i, ]    <- b.c                  # Store the sample
    MC.beta[i,] = calc_beta(b.c,lambda,tau,p,slab_code)
    if (verbose==1) setTxtProgressBar(pb, i)
  }



  return(list(mc.b=mc.b, n.s=N.s,
              beta = MC.beta,
              mc.tau = mc.tau,
              mc.var = mc.var))
}

#' Empirical Bayes estimation of prior spike probabilities of the SpSL model
#'
#' Run a stochastic EM algorithm
#'
#' @inheritParams ESS_Gibbs
#' @param gamma_power Robbins–Monro step-size exponent for SAEM updates.
#' @param lr Learning rate for lambda updates.
#' @param K_block Block size (iterations) per SAEM update.
#' @param schedule Exponent controlling learning-rate decay (e.g., lr / t^schedule).
#' @return A list containing MCMC draws and lambda trajectories.
#' @export
ESS_Gibbs_SAEM <- function(X,Y,b.c=NULL,sd.0=NULL, lambda=NULL, tau=NULL,
                           N=5000, block_size=3,S.max=100,
                           lambda_init = 9,
                           gamma_power = 0.9,
                           lr = 0.1,
                           K_block = 10,
                           schedule = 0.5, slab = "poly",
                           verbose=1) {

  slab_map <- c("exp" = 1, "poly" = 2, "nlp1" = 3, "nlp2" = 4)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'slab', or 'nlp1/nlp2'")

  p = ncol(X)
  if (is.null(sd.0)) sd.0  = sqrt(c(rep(1, p), rep(1,p), 1))
  if (is.null(b.c)) b.c = rnorm(2*p+1,0,sd.0)
  id = lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx = min(start_idx + block_size - 1, p)
    w_idx = start_idx:end_idx; a_idx = w_idx + p
    return(c(w_idx, a_idx))
  })
  id = c(id, list(2*p + 1))

  # N: total SAEM iterations
  # gamma_k = k^{-gamma_power}, with 0.5 < gamma_power <= 1
  # per-parameter adaptive lr state

  d    <- length(b.c)                   # nr of parameters
  K    <- length(id)                    # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  v    <- sapply(id, length)            # length of each block b.j
  mc.b <- matrix(NA, N, d)              # storage
  mc.lam = rep(NA,N/K_block); lambda=lambda_init
  if (is.null(tau)) tau = 1; sd_y = 1

  # 500 iterations of burn in
  for(i in 1:500){                         #  loop over iteration
    cpp_res <- update_blocks_cpp(b_c = b.c,
                                 X = X,
                                 Y = Y,
                                 id = id,
                                 sd_0 = sd.0,
                                 lambda = lambda,
                                 tau = tau,
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 slab_code = slab_code)
    b.c <- cpp_res$b_c

    eta_c = X%*%calc_beta(b.c,lambda,tau,p,slab_code)
    # update noise variance
    var_y = 1/rgamma(1, shape=0.01 + 0.5*length(Y), 0.01 + 0.5*sum((Y-eta_c)^2))
    sd_y = var_y^0.5

    # update scale parameter for w
    tau = rtruncnorm(1,a=0,b=Inf,
                     mean=1/sd_y^2*sum(Y*eta_c)*(1/2 + 1/sd_y^2 * sum(eta_c^2))^(-1),
                     sd=(1/2 + 1/sd_y^2 * sum(eta_c^2))^(-1))
  }

  B_hat      <- -1           # initial guess for E[log Phi(a_0)]

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    cpp_res <- update_blocks_cpp(b_c = b.c,
                                 X = X,
                                 Y = Y,
                                 id = id,
                                 sd_0 = sd.0,
                                 lambda = lambda,
                                 tau = tau,
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 slab_code = slab_code)
    b.c <- cpp_res$b_c

    eta_c = X%*%calc_beta(b.c,lambda,tau,p,slab_code)
    # update noise variance
    var_y = 1/rgamma(1, shape=0.01 + 0.5*length(Y), 0.01 + 0.5*sum((Y-eta_c)^2))
    sd_y = var_y^0.5

    # update scale parameter for w
    tau = rtruncnorm(1,a=0,b=Inf,
                     mean=1/sd_y^2*sum(Y*eta_c)*(1/2 + 1/sd_y^2 * sum(eta_c^2))^(-1),
                     sd=(1/2 + 1/sd_y^2 * sum(eta_c^2))^(-1))

    mc.b[i, ]    <- b.c             # Store the sample

    if (i%%K_block==0){
      z          <- mc.b[(i-K_block+1):i,2*p+1]                 # latent Z
      logPhi_z   <- mean(log(pnorm(z)))                # log Phi(Z)
      # stochastic approximation of B(lambda) = E[log Phi(Z)]
      t_block = i/K_block
      gamma_t <- t_block^(-gamma_power)     # Robbins step size
      B_hat   <- (1 - gamma_t)*B_hat + gamma_t*logPhi_z

      # keep B_hat away from 0 for stability
      B_hat <- max(min(B_hat, -0.3), -3)
      lambda_target <- -lambda / B_hat

      lr_t <- lr / (t_block^schedule)
      lambda <- max(1e-6, (1-lr_t)*lambda + lr_t*lambda_target)
      mc.lam[t_block]   <- lambda                        # store lambda
    }
    if (verbose==1) setTxtProgressBar(pb, i)
  }

  list(mc.lam       = mc.lam,
       lambda_final = lambda)
}
