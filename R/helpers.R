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
a0_star <- function(a0_raw, lam) {
  if (!is.finite(a0_raw) || !is.finite(lam) || lam <= 0) {
    return(NaN)
  }
  qnorm(pnorm(a0_raw, log.p = TRUE) / lam, log.p = TRUE)
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
  thresh <- a0_star(b[2*p + 1], lambda)
  tau * H_u(b[1:p]) * T_u(b[(p+1):(2*p)] - thresh)
}

#' @keywords internal
#' @noRd
calc_bias_group <- function(mat, group_map, lam_s, tau_S, slab_code, approx=F, k_apx=10){
  S <- ncol(mat)
  bias_mat <- sapply(seq_len(S), function(s) {
    calc_beta_group(mat[, s], group_map, lam_s[s], tau_S[s], slab_code, approx, k_apx)
  })
  return(bias_mat)
}

#' @keywords internal
#' @noRd
calc_beta_group <- function(b, group_map, lambda, tau, slab_code, approx=F, k_apx=10){
  p <- length(group_map)
  G <- max(group_map)
  if (slab_code == 1){T_u = T_l; H_u = H_l}
  else if (slab_code == 2){T_u = T_c; H_u = H_c}
  else if (slab_code == 3){T_u = if (approx) function(x) T_log(x, k_apx) else T_n1; H_u = H_n1}
  else if (slab_code == 4){T_u = T_n2; H_u = H_n2}
  a_group <- b[(p + 1):(p + G)]
  thresh <- a0_star(b[p + G + 1], lambda)
  tau * H_u(b[1:p]) * T_u(a_group[group_map] - thresh)
}

# function to generate correlated covariates
Gen_AR1 <- function(n,p,rho) {
  if (abs(rho) >= 1) stop("rho must be in (-1, 1).")
  Sigma  <- toeplitz(rho^(0:(p -1)))          # AR(1) covariance
  MASS::mvrnorm(n, mu=rep(0,p), Sigma=Sigma)
}

generate_logC <- function(logT, target_pct) {
  if (target_pct == 0) return(rep(Inf, length(logT)))
  T_time <- as.numeric(exp(logT))
  obj_fn <- function(lambda) {
    expected_cens <- mean(1 - exp(-lambda * T_time))
    return(expected_cens - target_pct)
  }

  # Find the root lambda that makes the objective function 0
  opt <- uniroot(obj_fn, interval = c(1e-10, 1e5), extendInt = "yes")
  lambda_c <- opt$root

  # Generate independent censoring times
  C_time <- rexp(length(T_time), rate = lambda_c)
  return(log(C_time))
}

#' @keywords internal
#' @noRd
build_group_id <- function(group_map) {
  split(seq_along(group_map), factor(group_map, levels = seq_len(max(group_map))))
}

#' @keywords internal
#' @noRd
validate_group_map <- function(group_map, p) {
  if (length(group_map) != p) stop("group_map must have length p.")
  if (anyNA(group_map)) stop("group_map cannot contain NA values.")
  if (any(group_map < 1)) stop("group_map must use 1-based positive group labels.")
  labs <- sort(unique(group_map))
  if (!identical(labs, seq_len(max(group_map)))) {
    stop("group_map must use contiguous labels 1, ..., G.")
  }
}
