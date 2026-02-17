#' ESS-within-Gibbs sampler for single source Bayesian HD AFT models
#'
#' Runs elliptical slice sampling updates for target parameters.
#'
#' @param X Design matrix.
#' @param Y Response.
#' @param C Observation indicator.
#' @param b.c Initial states for latent parameters.
#' @param xi shadow variable for slab scale parameter.
#' @param sd_y scale variable for noise.
#' @param lambda Threshold parameter.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param family Specification of outcome model, one of 'Weibull', 'Lognormal', 'Loglogistic'. Default is 'Weibull'.
#' @param slab Specification of slab distribution, one of 'exp', 'poly', 'nlp'. Default is 'nlp'.
#' @param verbose Verbosity flag.
#' @param debug Optional returning of MCMC runs other than the coefficient itself.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs_AFT <- function(X,Y,C, b.c=NULL, xi=NULL, sd_y=NULL, lambda=NULL,
                          N=5000, S.max=100, block_size=1,
                          family="Weibull", slab = "exp", verbose=1, debug=F) {

  fam_map <- c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
  fam_code <- fam_map[tolower(family)]
  if(is.na(fam_code)) stop("Family must be 'weibull', 'loglogistic', or 'lognormal'")

  slab_map <- c("exp" = 1, "poly" = 2, "nlp1" = 3, "nlp2" = 4)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'slab', or 'nlp1/nlp2'")

  # Type Safety
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  C <- as.numeric(C)

  p = ncol(X)
  sd.0  = sqrt(c(rep(1, p), rep(1,p), 1))
  if (is.null(b.c)) b.c = rnorm(2*p+1,0,sd.0)
  id <- lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx <- min(start_idx + block_size - 1, p)
    return(c(start_idx:end_idx, (start_idx:end_idx) + p))
  })
  id <- c(id, list(2*p + 1))
  if (is.null(lambda)) lambda = p^0.5

  # b.c:     Current state (vector)
  # LL.blg:  Function to compute log-likelihood
  # sd.0:    sd vector of the Gaussian prior
  # N:       Number of iterations
  # S.max    max nr of slice itr for each angle-sampling step

  d    <- length(b.c)                   # nr of parameters
  K    <- length(id)                    # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  MC.beta = matrix(NA, N, p)
  if (debug){
    mc.b <- matrix(NA, N, d)              # storage
    N.s  <- matrix(NA, N, K)              # nr of slice sampling itr at each MCMC-itr
    mc.tau = rep(NA,N)
    mc.sigma = rep(NA,N)
  }

  if (is.null(sd_y)) sd_y = 1
  if (is.null(xi)) xi = 1

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    cpp_res <- update_blocks_aft(b_c = b.c,
                                 X = X, Y = Y, C = C,
                                 id = id, sd_0 = sd.0,
                                 lambda = lambda, tau = abs(xi),
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 fam_code = fam_code,
                                 slab_code = slab_code)

    b.c <- cpp_res$b_c


    xi <- update_scale_aft(xi_curr = xi,
                           b_c = b.c, X = X, Y = Y, C = C,
                           lambda = lambda,
                           sd_y = sd_y,
                           sd_prior = 2.0, # Prior width for shadow var
                           fam_code = fam_code,
                           slab_code = slab_code)

    # update scale parameter
    sd_y <- update_sigma_to_aft(b_c = b.c,
                                X = X, Y = Y, C = C,
                                current_sigma = sd_y,
                                lambda = lambda, tau = abs(xi),
                                fam_code=fam_code,
                                slab_code=slab_code,
                                step_size = 0.1) # Tune step_size for ~30-40% acceptance

    if(debug){
      mc.b[i, ]    <- b.c                  # Store the sample
      N.s[i, ] <- cpp_res$N_s
      mc.tau[i]   <- abs(xi)
      mc.sigma[i] <- sd_y
    }
    MC.beta[i,] = calc_beta(b.c,lambda,abs(xi),p,slab_code)

    if (verbose==1) setTxtProgressBar(pb, i)
  }


  if (debug){
    return(list(mc.b=mc.b,
                MC_beta = MC.beta,
                n.s=N.s,
                mc_tau = mc.tau,
                mc_sigma = mc.sigma))
  }else{
    return(list(MC_beta = MC.beta))
  }

}

#' Empirical Bayes estimation of prior spike probabilities of the SpSL model
#'
#' Run a stochastic EM algorithm
#'
#' @inheritParams ESS_Gibbs_AFT
#' @param gamma_power Robbins–Monro step-size exponent for SAEM updates.
#' @param lr Learning rate for lambda updates.
#' @param K_block Block size (iterations) per SAEM update.
#' @param schedule Exponent controlling learning-rate decay (e.g., lr / t^schedule).
#' @return A list containing MCMC draws and lambda trajectories.
#' @export
EB_SAEM_AFT <- function(X,Y,C,b.c=NULL,
                        sd.0=NULL, lambda=NULL, N=5000,
                        S.max=100, block_size=1,
                        lambda_init = 9,
                        gamma_power = 0.9,
                        lr = 0.1,
                        K_block = 10,
                        schedule = 0.5, family = 'Weibull', slab = "exp",
                        verbose=1) {

  fam_map <- c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
  fam_code <- fam_map[tolower(family)]
  if(is.na(fam_code)) stop("Family must be 'weibull', 'loglogistic', or 'lognormal'")

  slab_map <- c("exp" = 1, "poly" = 2, "nlp1" = 3, "nlp2" = 4)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'slab', or 'nlp1/nlp2'")

  # Type Safety
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  C <- as.numeric(C)

  p = ncol(X)
  if (is.null(sd.0)) sd.0  = sqrt(c(rep(1, p), rep(1,p), 1))
  if (is.null(b.c)) b.c = rnorm(2*p+1,0,sd.0)
  id <- lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx <- min(start_idx + block_size - 1, p)
    return(c(start_idx:end_idx, (start_idx:end_idx) + p))
  })
  id <- c(id, list(2*p + 1))

  d    <- length(b.c)                   # nr of parameters
  K    <- length(id)                    # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  mc.b <- matrix(NA, N, d)              # storage
  sd_y = 1; xi = 1
  mc.lam = rep(NA,N/K_block); lambda=lambda_init

  # 500 iterations of burn in
  for(i in 1:500){                         #  loop over iteration
    cpp_res <- update_blocks_aft(b_c = b.c,
                                 X = X, Y = Y, C = C,
                                 id = id, sd_0 = sd.0,
                                 lambda = lambda, tau = abs(xi),
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 fam_code = fam_code,
                                 slab_code = slab_code)

    b.c <- cpp_res$b_c

    xi <- update_scale_aft(xi_curr = xi,
                           b_c = b.c, X = X, Y = Y, C = C,
                           lambda = lambda,
                           sd_y = sd_y,
                           sd_prior = 2.0, # Prior width for shadow var
                           fam_code = fam_code,
                           slab_code = slab_code)

    # update scale parameter
    sd_y <- update_sigma_to_aft(b_c = b.c,
                                X = X, Y = Y, C = C,
                                current_sigma = sd_y,
                                lambda = lambda, tau = abs(xi),
                                fam_code=fam_code,
                                slab_code=slab_code,
                                step_size = 0.1) # Tune step_size for ~30-40% acceptance
  }

  B_hat      <- -1           # initial guess for E[log Phi(a_0)]

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    cpp_res <- update_blocks_aft(b_c = b.c,
                                 X = X, Y = Y, C = C,
                                 id = id, sd_0 = sd.0,
                                 lambda = lambda, tau = abs(xi),
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 fam_code = fam_code,
                                 slab_code = slab_code)

    b.c <- cpp_res$b_c
    mc.b[i, ]    <- b.c                  # Store the sample

    xi <- update_scale_aft(xi_curr = xi,
                           b_c = b.c, X = X, Y = Y, C = C,
                           lambda = lambda,
                           sd_y = sd_y,
                           sd_prior = 2.0, # Prior width for shadow var
                           fam_code = fam_code,
                           slab_code = slab_code)

    # update scale parameter
    sd_y <- update_sigma_to_aft(b_c = b.c,
                                X = X, Y = Y, C = C,
                                current_sigma = sd_y,
                                lambda = lambda, tau = abs(xi),
                                fam_code=fam_code,
                                slab_code=slab_code,
                                step_size = 0.1) # Tune step_size for ~30-40% acceptance

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

      lr_t <- lr / (t_block^schedule)   # 0.3 is mild decay; tweak if needed
      lambda <- max(1e-6, (1-lr_t)*lambda + lr_t*lambda_target)
      mc.lam[t_block]   <- lambda                        # store lambda
    }
    if (verbose==1) setTxtProgressBar(pb, i)
  }

  list(mc.lam    = mc.lam,
       lambda_final = lambda,
       b_c = b.c, xi = xi, sd_y=sd_y)
}
