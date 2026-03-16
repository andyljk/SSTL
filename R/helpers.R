#' Internal helper: positive-part operator

#' @keywords internal
#' @noRd
calc_bias <- function(mat,p,lam_s,tau_S,slab_code,approx=F,k_apx=10){
  S = ncol(mat)
  bias_mat <- sapply(1:S, function(s) {
    calc_beta(mat[, s], lam_s[s], tau_S[s], p, slab_code, approx, k_apx)
  })
  return(bias_mat)
}

#' @keywords internal
#' @noRd
a0_star <- function(a0_raw,lam) {
  # This calculates a_0* = Phi_inv( Phi(a0_raw)^(1/p) )
  return(qnorm(pnorm(a0_raw)^(1/lam)))
}

#' @keywords internal
#' @noRd
T_n1 = function(b) as.numeric(b > 0)
#' @keywords internal
#' @noRd
T_log <- function(b, k) 1 / (1 + exp(-k * b))
#' @keywords internal
#' @noRd
H_n1 = function(w, phi=2, d=2) phi * sign(w) * (exp(d*w^2)-1)^(0.5/d) # tau * sign(w) * sqrt(abs(w)) * exp(0.5 * w^2)

#' @keywords internal
#' @noRd
T_n2 = function(b) as.numeric(b > 0)
#' @keywords internal
#' @noRd
H_n2 = function(w, phi=2) phi * sign(w) * sqrt(abs(w)) * exp(0.5 * w^2)

#' @keywords internal
#' @noRd
T_c = function(b) pmax(b,0)
#' @keywords internal
#' @noRd
H_c = function(w) sign(w)*exp(0.5*w^2)

#' @keywords internal
#' @noRd
T_l = function(b) pmax(b,0)
#' @keywords internal
#' @noRd
H_l = function(w) w

#' @keywords internal
#' @noRd
calc_beta  = function(b,lambda,tau,p,slab_code,approx=F,k_apx=10){
  if (slab_code == 1){T_u = T_l; H_u = H_l}
  else if (slab_code == 2){T_u = T_c; H_u = H_c}
  else if (slab_code == 3){T_u = if (approx) function(x) T_log(x, k_apx) else T_n1; H_u = H_n1}
  else if (slab_code == 4){T_u = T_n2; H_u = H_n2}
  tau*H_u(b[1:p])*T_u(b[(p+1):(2*p)]-qnorm(pnorm(b[2*p+1])^(1/lambda)))
}

# function to generate correlated covariates
Gen_AR1 <- function(n,p,rho) {
  if (abs(rho) >= 1) stop("rho must be in (-1, 1).")
  Sigma  <- toeplitz(rho^(0:(p -1)))          # AR(1) covariance
  MASS::mvrnorm(n, mu=rep(0,p), Sigma=Sigma)
}
