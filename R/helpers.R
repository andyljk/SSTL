#' Internal helper: positive-part operator
#' @keywords internal
#' @noRd
T.n <- function(b) pmax(b, 0)

#' @keywords internal
#' @noRd
a0_star <- function(a0_raw, lam) stats::qnorm(pnorm(a0_raw)^(1/lam))

#' @keywords internal
#' @noRd
calc_beta = function(b, lambda, p) {
  w = b[1:p]; a = b[(p+1):(2*p)]; a0 = b[2*p+1]
  return(w * T.n(a - a0_star(a0, lambda)))
}

#' @keywords internal
#' @noRd
calc_bias <- function(mat,p,lam_s){
  w_mat = mat[1:p, , drop=FALSE]
  alp_mat = mat[(p+1):(2*p), , drop=FALSE]
  thres_vec = rep(a0_star(mat[2*p + 1, ], lam_s), each = p)
  thres_mat = matrix(thres_vec, nrow = p, ncol = ncol(mat))
  return(w_mat * pmax(alp_mat - thres_mat, 0))
}

# function to generate correlated covariates
Gen_AR1 <- function(n,p,rho) {
  if (abs(rho) >= 1) stop("rho must be in (-1, 1).")
  Sigma  <- toeplitz(rho^(0:(p -1)))          # AR(1) covariance
  MASS::mvrnorm(n, mu=rep(0,p), Sigma=Sigma)
}
