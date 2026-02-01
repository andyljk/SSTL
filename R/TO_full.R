# rm(list=ls())
#
# library(posterior)
# library(pROC)
# library(optimx)
# library(Rcpp)
# library(RcppArmadillo)
#
# # Source the C++ function
# sourceCpp("~/Desktop/Steffen/Model_dev/ESS_fast_target_only.cpp")

# fix w for now, use the transformation for a_0

# elliptical slice sampling within gibbs function
#' ESS-within-Gibbs sampler for single source Bayesian HD linear regression models.
#'
#' Runs elliptical slice sampling updates for target parameters and source biases.
#'
#' @param X Design matrix.
#' @param Y Response vector.
#' @param b.c Initial states for latent parameters.
#' @param sd.0 Prior SD vector for target latent parameters.
#' @param lambda Threshold parameter.
#' @param N Number of MCMC iterations.
#' @param S.max Maximum slice iterations per update.
#' @param block_size Block size for updates.
#' @param verbose Verbosity flag.
#' @return A list containing MCMC draws and diagnostics.
#' @export
ESS_Gibbs <- function(X,Y,b.c=NULL, sd.0=NULL, lambda=NULL,
                      N=5000, block_size=3, S.max=500, verbose=1) {

  p = ncol(X)
  if (is.null(sd.0)) sd.0  = sqrt(c(rep(.75, p), rep(1,p), 1))
  if (is.null(b.c)) b.c = rnorm(2*p+1,0,sd.0)

  id = lapply(seq(1, p, by = block_size), function(start_idx) {
    end_idx = min(start_idx + block_size - 1, p)
    w_idx = start_idx:end_idx; a_idx = w_idx + p
    return(c(w_idx, a_idx))
  })
  id = c(id, list(2*p + 1))

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
  mc.tau2_w = mc.tau2_a0 = rep(NA,N)
  mc.var = rep(NA,N); sd_y = 1

  if (verbose==1) pb <- txtProgressBar(min = 0, max = N, style = 3)
  for(i in 1:N){                         #  loop over iteration
    cpp_res <- update_blocks_cpp(b_c = b.c,
                                        X = X,
                                        Y = Y,
                                        id = id,
                                        sd_0 = sd.0,
                                        lambda = lambda,
                                        sd_y = sd_y,
                                        S_max = S.max)
    b.c <- cpp_res$b_c
    N.s[i, ] <- cpp_res$N_s

    # update noise variance
    var_y = 1/rgamma(1, shape=0.01 + 0.5*length(Y), 0.01 + 0.5*sum((Y-X%*%calc_beta(b.c,lambda,p))^2))
    sd_y = var_y^0.5; mc.var[i] = var_y

    # update prior variance of w
    tau2_w = 1/rgamma(1, shape = 0.1 + p/2, 0.1 + sum(b.c[1:p]^2)/2)
    sd.0[1:p] = tau2_w^0.5; mc.tau2_w[i] = tau2_w

    mc.b[i, ]    <- b.c                  # Store the sample

    if (verbose==1) setTxtProgressBar(pb, i)
  }



  return(list(mc.b=mc.b, n.s=N.s,
              mc.tau2_w = mc.tau2_w,
              mc.tau2_a0 = mc.tau2_a0,
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
EB_Gibbs_SAEM <- function(X,Y,b.c=NULL,sd.0=NULL, lambda=NULL,
                           N=5000, block_size=3,S.max=100,
                           lambda_init = 9,
                           gamma_power = 0.9,
                           lr = 0.1,
                           K_block = 10,
                           schedule = 0.5,
                           verbose=1) {

  p = ncol(X)
  if (is.null(sd.0)) sd.0  = sqrt(c(rep(.75, p), rep(1,p), 1))
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
  N.s  <- matrix(NA, N, K)              # nr of slice sampling itr at each MCMC-itr
  mc.b <- matrix(NA, N, d)              # storage
  mc.tau2_w = rep(NA,N)
  mc.var = rep(NA,N); sd_y = 1
  mc.lam = rep(NA,N/K_block); lambda=lambda_init

  # 500 iterations of burn in
  for(i in 1:500){                         #  loop over iteration
    cpp_res <- update_blocks_cpp(b_c = b.c,
                                 X = X,
                                 Y = Y,
                                 id = id,
                                 sd_0 = sd.0,
                                 lambda = lambda,
                                 sd_y = sd_y,
                                 S_max = S.max)
    b.c <- cpp_res$b_c

    # update noise variance
    var_y = 1/rgamma(1, shape=0.01 + 0.5*length(Y), 0.01 + 0.5*sum((Y-X%*%calc_beta(b.c,lambda,p))^2))
    sd_y = var_y^0.5

    # update prior variance of w
    tau2_w = 1/rgamma(1, shape = 0.1 + p/2, 0.1 + sum(b.c[1:p]^2)/2)
    sd.0[1:p] = tau2_w^0.5
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
                                 sd_y = sd_y,
                                 S_max = S.max)
    b.c <- cpp_res$b_c
    N.s[i, ] <- cpp_res$N_s

    # update noise variance
    var_y = 1/rgamma(1, shape=0.01 + 0.5*length(Y), 0.01 + 0.5*sum((Y-X%*%calc_beta(b.c,lambda,p))^2))
    sd_y = var_y^0.5; mc.var[i] = var_y

    # update prior variance of w
    tau2_w = 1/rgamma(1, shape = 0.1 + p/2, 0.1 + sum(b.c[1:p]^2)/2)
    sd.0[1:p] = tau2_w^0.5; mc.tau2_w[i] = tau2_w

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

      lr_t <- lr / (t_block^schedule)
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
#
# T.n   = function(b) pmax(b, 0)
# calc_beta  = function(b,lambda,p) b[1:p]*T.n(b[(p+1):(2*p)]-qnorm(pnorm(b[2*p+1])^(1/lambda)))
#
# sim_data = function(n,p,sparse_level=0.1, effect_size=0.5){
#   p_0 = round(p*sparse_level)
#   sd_y  = 1
#   b0    = c(rep(effect_size,p_0), rep(0, p-p_0))
#   X     = matrix(rnorm(p*n), n,p) # rmvn(n,rep(0,p),AR1_cov(p,rho=0.7)) #
#   Y     = X%*%b0 + rnorm(n,0,sd_y)
#   return(X,Y,b0)
# }

# # data
# set.seed(456)
#
#
# b.c   = c(b0, rep(0,p), 0)
# sd.0  = sqrt(c(rep(.75, p), rep(1,p), 1))
# #
# #
# lam_eb = ESS.Gibbs.SAEM(X,Y,b.c,sd.0,N=10000,gamma_power=0.9,schedule = 0.5,K_block=10,block_size=1)
# plot(lam_eb$mc.lam)
# lam = lam_eb$lambda_final
#
# t0    = Sys.time()
# MC  = ESS.Gibbs(X,Y,b.c=b.c, sd.0=sd.0, lambda=lam, N=10000, S.max=100)
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
# par(mfrow=c(1,1))
# boxplot(MC.b[5000:10000,1:p], outline=F, ylim=c(-.2, 1.3))
#
# # par(mfrow=c(3,2),mar=c(3,3,1,1))
# # plot(MC.b[,1]);plot(MC.b[,3]);plot(MC.b[,10])
# # plot(MC.b[,15]);plot(MC.b[,20]);plot(MC.b[,50])
#
# b_est = colMeans(MC.b[5000:10000,])
#
# sum((b_est - b0)^2)/sum(b0^2)
#
#
#
# # compare with classical spike and slab
#
# library(spikeslab)
#
# data = data.frame(cbind(X,Y))
# names(data)[151] = "y"
#
# ss_fit = spikeslab(x=X, y=Y,
#                    big.p.value = 1,  # Use all variables initially
#                    # MCMC parameters
#                    n.iter1 = 500,    # Burn-in iterations
#                    n.iter2 = 1000,   # MCMC iterations after burn-in
#                    # Spike and slab parameters
#                    max.var = p,      # Maximum variables to consider
#                    verbose = F)
#
# posterior_inclusion <- ss_fit$phat
#
# # Calculate posterior probability of being zero
# posterior_zero <- 1 - posterior_inclusion
#
#
# sum((ss_fit$bma - b0)^2)/sum(b0^2)
# sum((ss_fit$gnet - b0)^2)/sum(b0^2)
#
#
# library(glmtrans)
#
# dat = list()
# dat$target$x = X; dat$target$y = as.numeric(Y)
#
# # fit glmtrans
# fit_trans = glmtrans(dat$target, dat$source, intercept=FALSE)
# beta_trans = fit_trans$beta[-1]
# sum((beta_trans - b0)^2)/sum(b0^2)
#
