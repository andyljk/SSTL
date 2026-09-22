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
#' @param b0 Initial value for the intercept term, default is 0.
#' @param intercept Whether to estimate the intercept term in the MCMC algorithm, default is TRUE.
#' @param lambda Threshold parameter.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param family Specification of outcome model, one of 'Weibull', 'Lognormal', 'Loglogistic'. Default is 'Weibull'.
#' @param slab Specification of slab distribution, one of 'exp', 'poly', 'nlp', 'guassian'. Default is 'exp'.
#' @param verbose Verbosity flag.
#' @param debug Optional returning of MCMC runs other than the coefficient itself.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs_AFT <- function(X,Y,C,b.c=NULL,
                          sd.0=NULL, lambda=NULL,
                          xi=NULL, sd_y=NULL,
                          b0=NULL, intercept=TRUE,
                          N=5000, S.max=100, block_size=1,
                          family="Weibull", slab = "exp",
                          verbose=1, debug=F) {

  fam_map <- c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
  fam_code <- fam_map[tolower(family)]
  if(is.na(fam_code)) stop("Family must be 'weibull', 'loglogistic', or 'lognormal'")

  slab_map <- c("exp" = 1, "poly" = 2, "nlp" = 3, "nlp2" = 4, "guassian" = 5)
  slab_code <- slab_map[tolower(slab)]
  if(is.na(slab_code)) stop("Slab must be 'exp', 'poly', 'nlp', or 'guassian'")

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

  # b.c:     Current state (vector)
  # LL.blg:  Function to compute log-likelihood
  # sd.0:    sd vector of the Gaussian prior
  # N:       Number of iterations
  # S.max    max nr of slice itr for each angle-sampling step

  d    <- length(b.c)                   # nr of parameters
  K    <- length(id)                    # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  if (debug){
    N.s  <- matrix(NA, N, K)              # nr of slice sampling itr at each MCMC-itr
    mc.b <- matrix(NA, N, d)              # storage
    mc.tau = rep(NA,N)
  }
  MC.beta = matrix(NA, N, p)
  mc.sigma = rep(NA,N)
  if (is.null(sd_y)) sd_y = 1
  if (is.null(xi)) xi = 1
  if (is.null(b0)) b0 = 0
  if (intercept) MC.b0 = rep(NA, N)

  resid <- as.numeric(Y - b0 - X %*% calc_beta(b.c, lambda, exp(xi), p, slab_code))


  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    if (intercept) {
      int_res <- update_intercept_to_aft(b0_curr = b0,
                                         resid = resid, Y = Y,
                                         C = C,
                                         sd_y = sd_y,
                                         fam_code = fam_code,
                                         sd_prior = 2.0)
      b0 <- int_res$b0
      resid <- int_res$resid
    }

    cpp_res <- update_blocks_aft(b_c = b.c,
                                 resid = resid, Y = Y,
                                 X = X, C = C,
                                 id = id, sd_0 = sd.0,
                                 lambda = lambda, tau = exp(xi),
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 fam_code = fam_code,
                                 slab_code = slab_code)

    b.c <- cpp_res$b_c
    resid <- cpp_res$resid
    if (debug) N.s[i, ] <- cpp_res$N_s

    scale_res <- update_scale_aft(xi_curr = xi,
                                  Y_scale = Y - b0, resid = resid, Y = Y, C = C,
                                  sd_y = sd_y,
                                  sd_prior = 2.0, # Prior width for shadow var
                                  fam_code = fam_code)
    xi <- scale_res$xi
    resid <- scale_res$resid

    # update scale parameter
    sd_y <- update_sigma_to_aft(resid = resid, Y = Y,
                                C = C,
                                current_sigma = sd_y,
                                fam_code=fam_code,
                                step_size = 0.1) # Tune step_size for ~30-40% acceptance

    if (debug){
      mc.b[i, ]    <- b.c                  # Store the sample
      mc.tau[i]   <- exp(xi)
    }
    mc.sigma[i] <- sd_y
    if (intercept) MC.b0[i] <- b0
    MC.beta[i,] = calc_beta(b.c,lambda,exp(xi),p,slab_code)

    if (verbose==1) setTxtProgressBar(pb, i)
  }

  if (debug) {
    out <- list(mc_b=mc.b, MC_beta = MC.beta, N_s=N.s,
                mc_tau = mc.tau, mc_sigma = mc.sigma)
    if (intercept) out$mc_b0 <- MC.b0
    return(out)
  } else {
    out <- list(MC_beta = MC.beta, mc_sigma = mc.sigma)
    if (intercept) out$MC_b0 <- MC.b0
    return(out)
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
#' @param optimizer "legacy" is the plain doubly smoothed MCEM, "adagrad" for adaptive step sizes.
#' @param polyak_start Return average or not, starting from a percentage of the run.
#' @return A list containing the lambda trajectory, final threshold, and final
#'   state (b_c, xi, sd_y, and b0_final when an intercept is estimated).
#' @export
EB_SAEM_AFT <- function(X,Y,C,N=5000, burn=1000,
                        S.max=50, block_size=1,
                        lambda_init = 9,
                        gamma_power = 0.9,
                        lr = 0.1,
                        K_block = 10,
                        schedule = 0.5, family = 'Weibull', slab = "exp",
                        b0=NULL, intercept=FALSE,
                        optimizer = c("legacy", "adagrad"),
                        adagrad_eps = 1e-8,
                        max_log_step = 0.35,
                        lambda_min = 1e-6,
                        lambda_max = 1e4,
                        polyak_start = 0.9,
                        verbose=1) {

  optimizer = match.arg(optimizer)

  # Type Safety
  X <- as.matrix(X); Y <- as.numeric(Y); C <- as.numeric(C)
  p = ncol(X)
  n_blocks = floor(N / K_block)

  # burn in
  res = ESS_Gibbs_AFT(X=X, Y=Y, C=C,
                      lambda=lambda_init,
                      b0=b0, intercept=intercept,
                      N=burn, S.max=S.max,
                      block_size=block_size,
                      family=family, slab=slab,
                      verbose=0, debug=T)

  # Initialize the parameters
  b.c = res$mc_b[burn,]
  xi = log(res$mc_tau[burn])
  sd_y = res$mc_sigma[burn]
  if (intercept) b0 = res$mc_b0[burn]
  else b0 = 0

  # storages
  mc.lam = rep(NA,n_blocks); lambda=lambda_init
  B_hat_trace = rep(NA, n_blocks)

  # AdaGrad state in log-lambda space
  theta = log(lambda)
  G_acc = 0

  B_hat = mean(pnorm(res$mc_b[max(1, burn-1000):burn,2*p+1],log=T))  # initial guess for E[log Phi(a_0)]

  if (verbose==1) pb <- txtProgressBar(min = 0, max = n_blocks, style = 3)
  for(t_block in seq_len(n_blocks)){                         #  loop over iteration

    res <- ESS_Gibbs_AFT(X=X, Y=Y, C=C,
                         b.c=b.c, lambda=lambda,
                         xi=xi, sd_y=sd_y,
                         b0=b0, intercept=intercept,
                         N=K_block, S.max=S.max,
                         block_size=block_size,
                         family=family, slab=slab,
                         verbose=0, debug=T)

    b.c  <- res$mc_b[K_block, ]
    xi    <- log(res$mc_tau[K_block])
    sd_y <- res$mc_sigma[K_block]
    if (intercept) b0 <- res$mc_b0[K_block]

    # SAEM update
    logPhi_z   <- mean(pnorm(res$mc_b[,2*p+1],log.p=T))                # log Phi(Z)
    # stochastic approximation of B(lambda) = E[log Phi(Z)]
    gamma_t <- t_block^(-gamma_power)     # Robbins step size
    B_hat   <- (1 - gamma_t)*B_hat + gamma_t*logPhi_z

    if (optimizer == "legacy") {
      # Existing update rule (kept for compatibility)
      lambda_target <- -lambda / B_hat
      lr_t <- lr / (t_block^schedule)
      lambda <- max(lambda_min, min(lambda_max, (1 - lr_t) * lambda + lr_t * lambda_target))
      theta <- log(lambda)
    } else { # AdaGrad on theta = log(lambda)
      # EM signal in log-space: theta* - theta = log(-1/B_hat)
      d_t = log(-1 / B_hat)
      G_acc = G_acc + d_t^2
      lr_t = lr / ((t_block^schedule) * sqrt(G_acc + adagrad_eps))
      step_theta = max(-max_log_step, min(max_log_step, lr_t * d_t))
      theta <- theta + step_theta
      theta <- max(log(lambda_min), min(log(lambda_max), theta))
      lambda <- exp(theta)
    }

    mc.lam[t_block] = lambda
    B_hat_trace[t_block] = B_hat                        # store lambda
    if (verbose==1) setTxtProgressBar(pb, t_block)
  }

  # Polyak average in tail for a more stable final estimate
  if (polyak_start < 1) {
    i0 <- max(1, floor(polyak_start * n_blocks))
    lambda_final <- mean(mc.lam[i0:n_blocks])
  } else {
    lambda_final <- lambda
  }

  out <- list(mc.lam = mc.lam,
              lambda_final = lambda_final,
              b_c = b.c, xi = xi, sd_y = sd_y)
  if (intercept) out$b0_final <- b0
  out
}
