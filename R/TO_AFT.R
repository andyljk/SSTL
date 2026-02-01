#' ESS-within-Gibbs sampler for single source Bayesian HD AFT models
#'
#' Runs elliptical slice sampling updates for target parameters.
#'
#' @param X Design matrix.
#' @param Y Response.
#' @param C Observation indicator.
#' @param b.c Initial states for latent parameters.
#' @param sd.0 Prior SD vector for target latent parameters.
#' @param lambda Threshold parameter.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param family Specification of outcome model, one of 'Weibull', 'Lognormal', 'Loglogistic'. Default is 'Weibull'.
#' @param verbose Verbosity flag.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs_AFT <- function(X,Y,C,b.c=NULL,
                      sd.0=NULL, lambda=NULL, N=5000,
                      S.max=100,
                      block_size=1, family="Weibull", verbose=1) {

  fam_map <- c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
  fam_code <- fam_map[tolower(family)]
  if(is.na(fam_code)) stop("Family must be 'weibull', 'loglogistic', or 'lognormal'")

  # Type Safety
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  C <- as.numeric(C)

  p = ncol(X)
  if (is.null(sd.0)) sd.0  = sqrt(c(rep(.75, p), rep(1,p), 1))
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
  N.s  <- matrix(NA, N, K)              # nr of slice sampling itr at each MCMC-itr
  mc.b <- matrix(NA, N, d)              # storage
  mc.tau2_w = rep(NA,N)
  mc.sigma = rep(NA,N)
  sd_y = 1

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    cpp_res <- update_blocks_aft(b_c = b.c,
                                 X = X,
                                 Y = Y,
                                 C = C,   # Pass censoring vector
                                 id = id,
                                 sd_0 = sd.0,
                                 lambda = lambda,
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 fam_code=fam_code)

    b.c <- cpp_res$b_c
    N.s[i, ] <- cpp_res$N_s

    # update prior variance of w
    tau2_w = 1/rgamma(1, shape = 0.1 + p/2, 0.1 + sum(b.c[1:p]^2)/2)
    sd.0[1:p] = tau2_w^0.5; mc.tau2_w[i] = tau2_w

    # update scale parameter
    sd_y <- update_sigma_to_aft(b_c = b.c,
                                X = X, Y = Y, C = C,
                                current_sigma = sd_y,
                                lambda = lambda, fam_code=fam_code,
                                step_size = 0.1) # Tune step_size for ~30-40% acceptance
    mc.sigma[i] <- sd_y

    mc.b[i, ]    <- b.c                  # Store the sample

    if (verbose==1) setTxtProgressBar(pb, i)
  }



  return(list(mc.b=mc.b, n.s=N.s,
              mc.tau2_w = mc.tau2_w,
              mc.sigma = mc.sigma,
              lambda=lambda))
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
                           sd.0=NULL, N=5000,
                           S.max=100, block_size=3,
                           lambda_init = 9,
                           gamma_power = 0.9,
                           lr = 0.1,
                           K_block = 10,
                           schedule = 0.5, family = 'Weibull',
                           verbose=1) {

  fam_map <- c("weibull" = 1, "loglogistic" = 2, "lognormal" = 3)
  fam_code <- fam_map[tolower(family)]
  if(is.na(fam_code)) stop("Family must be 'weibull', 'loglogistic', or 'lognormal'")

  # N: total SAEM iterations
  # gamma_k = k^{-gamma_power}, with 0.5 < gamma_power <= 1
  # per-parameter adaptive lr state
  # Type Safety
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  C <- as.numeric(C) # Ensure C is numeric (0/1) for dot product in C++

  p = ncol(X)
  if (is.null(sd.0)) sd.0  = sqrt(c(rep(.75, p), rep(1,p), 1))
  if (is.null(b.c)) b.c = rnorm(2*p+1,0,sd.0)
  id <- lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx <- min(start_idx + block_size - 1, p)
    return(c(start_idx:end_idx, (start_idx:end_idx) + p))
  })
  id <- c(id, list(2*p + 1))

  d    <- length(b.c)                   # nr of parameters
  K    <- length(id)                    # nr of parameter blocks, i.e. b=(b.1, ..., b.K) with b.k in R^d.k
  N.s  <- matrix(NA, N, K)              # nr of slice sampling itr at each MCMC-itr
  mc.b <- matrix(NA, N, d)              # storage
  mc.tau2_w = rep(NA,N)
  mc.sigma = rep(NA,N); sd_y = 1
  mc.lam = rep(NA,N/K_block); lambda=lambda_init

  # 500 iterations of burn in
  for(i in 1:500){                         #  loop over iteration
    cpp_res <- update_blocks_aft(b_c = b.c,
                                 X = X,
                                 Y = Y,
                                 C = C,   # Pass censoring vector
                                 id = id,
                                 sd_0 = sd.0,
                                 lambda = lambda,
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 fam_code=fam_code)

    b.c <- cpp_res$b_c

    # update prior variance of w
    tau2_w = 1/rgamma(1, shape = 0.1 + p/2, 0.1 + sum(b.c[1:p]^2)/2)
    sd.0[1:p] = tau2_w^0.5

    # update scale parameter
    sd_y <- update_sigma_to_aft(b_c = b.c,
                                X = X, Y = Y, C = C,
                                current_sigma = sd_y,
                                lambda = lambda, fam_code=fam_code,
                                step_size = 0.1) # Tune step_size for ~30-40% acceptance
  }

  B_hat      <- -1           # initial guess for E[log Phi(a_0)]

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    cpp_res <- update_blocks_aft(b_c = b.c,
                                 X = X,
                                 Y = Y,
                                 C = C,   # Pass censoring vector
                                 id = id,
                                 sd_0 = sd.0,
                                 lambda = lambda,
                                 sd_y = sd_y,
                                 S_max = S.max,
                                 fam_code=fam_code)

    b.c <- cpp_res$b_c
    N.s[i, ] <- cpp_res$N_s

    # update prior variance of w
    tau2_w = 1/rgamma(1, shape = 0.1 + p/2, 0.1 + sum(b.c[1:p]^2)/2)
    sd.0[1:p] = tau2_w^0.5; mc.tau2_w[i] = tau2_w

    # update scale parameter
    sd_y <- update_sigma_to_aft(b_c = b.c,
                                X = X, Y = Y, C = C,
                                current_sigma = sd_y,
                                lambda = lambda, fam_code=fam_code,
                                step_size = 0.1) # Tune step_size for ~30-40% acceptance
    mc.sigma[i] <- sd_y

    mc.b[i, ]    <- b.c                  # Store the sample

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

  list(mc.b      = mc.b,
       mc.lam    = mc.lam,
       mc.tau2_w = mc.tau2_w,
       N.s       = N.s,
       lambda_final = lambda)
}


# T.n   = function(b) pmax(b, 0)
# beta  = function(b,p,lambda) b[1:p]*T.n(b[(p+1):(2*p)]-qnorm(pnorm(b[2*p+1])^(1/lambda)))
#
# # data
# set.seed(714)
# p     = 200; p_0 = p/10
# n     = 1000
# sd_y  = 1
# b0    = c(rep(0.5,p_0), rep(0, p-p_0))
# X     = matrix(rnorm(p*n), n,p)
# logT  = X%*%b0 - revd(n,0,sd_y) # log survival time
#
# logC  = rnorm(n, mean(logT), sd(logT))
#
# Y     = pmin(logT, logC)
# C     = as.integer(logT <= logC)
#
# b.c   = c(b0, rep(0,p), 0)
# sd.0  = sqrt(c(rep(.75, p), rep(1,p), 1))
#
# lam_eb = EB_SAEM_AFT(X,Y,C,b.c,sd.0,N=10000,gamma_power=0.9,K_block=20)
# plot(lam_eb$mc.lam)
# lam = lam_eb$lambda_final
#
# t0    = Sys.time()
# MC  = ESS_Gibbs_AFT(X,Y,C,b.c=b.c, sd.0=sd.0, lambda=lam, N=10000, family='Weibull')
# t1    = Sys.time()
# print(t1-t0, digits = 2)
#
# MC.b = t(apply(MC$mc.b, 1, function(b) b[1:p] * T.n(b[(p+1):(2*p)] - qnorm(pnorm(b[2*p+1])^(1/lam)) )  ))
# MC.alp = t(apply(MC$mc.b[5000:10000,], 1, function(b) T.n(b[(p+1):(2*p)] - qnorm(pnorm(b[2*p+1])^(1/lam)) )  ))
#
# alp_pm = colMeans(MC.alp==0)
# roc_obj <- roc(b0, 1-alp_pm)
# auc_value <- auc(roc_obj)
#
# boxplot(MC.b[5000:10000,1:p], outline=F, ylim=c(-.2, 1.3))
#
# b_est = colMeans(MC.b[5000:10000,])
#
# sum((b_est - b0)^2)/sum(b0^2)
#
#
#
# # compare with classical AFT model
#
# library(survival)
#
# fit_weibull <- survreg(Surv(Y, C) ~ ., data = mydata, dist = "weibull")
# summary(fit_weibull)


