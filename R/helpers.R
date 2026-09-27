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
validate_group_map <- function(group_map, p) {
  if (length(group_map) != p) stop("group_map must have length p.")
  if (anyNA(group_map)) stop("group_map cannot contain NA values.")
  if (any(group_map < 1)) stop("group_map must use 1-based positive group labels.")
  labs <- sort(unique(group_map))
  if (!identical(labs, seq_len(max(group_map)))) {
    stop("group_map must use contiguous labels 1, ..., G.")
  }
}

#' @keywords internal
#' @noRd
as_double_matrix <- function(x) {
  x <- as.matrix(x)
  storage.mode(x) <- "double"
  x
}
