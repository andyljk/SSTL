# SSTL

SSTL is an R package for Bayesian sparse regression and transfer learning with discrete spike-and-slab priors. It supports target-only (TO), transfer learning (TL), and grouped transfer learning models. Supported model types include generalized linear models: linear, logistic, Poisson, Gamma, Beta; and survival models including accelerated failure time models with lognormal, Weibull, and loglogistic outcome errors.

## Installation

Install the package and its dependencies from [GitHub](https://github.com/andyljk/SSTL). A C++17 compiler is required.

```r
install.packages("remotes") # Once, if needed
remotes::install_github("andyljk/SSTL", upgrade = "never")
```


## 1. Target-only regression

Simulate a Gaussian outcome with 200 predictors, four nonzero coefficients, and 80 target observations.

```r
library(SSTL)
set.seed(123)

p <- 200
beta <- c(1, -1, 0.5, 0.5, rep(0, p - 4))
X_T <- matrix(rnorm(80 * p), ncol = p)
Y_T <- as.numeric(0.5 + X_T %*% beta + rnorm(80))

fit_to <- SSTL(
  X = X_T, Y = Y_T, family = "Gaussian",
  EB = TRUE, EB_control = list(N = 1000, burn = 500),
  N = 5000, burn = 1000, intercept = TRUE, verbose = 1
)
fit_to$post_mean[1:10]
fit_to$pip[1:10] # Posterior inclusion probabilities
```

## 2. Transfer learning

Add two related source studies with 120 and 160 observations, each with the same 200 predictors. TL estimates target coefficients and study-specific deviations. Source data are supplied as lists; predictor columns must have the same order across studies.

```r
X_s <- lapply(c(120, 160), function(n) matrix(rnorm(n * p), ncol = p))
beta_s <- list(
  beta + c(0.2, rep(0, p - 1)),
  beta + c(0, -0.2, rep(0, p - 2))
)
Y_s <- lapply(1:2, function(s) {
  as.numeric(0.2 + X_s[[s]] %*% beta_s[[s]] + rnorm(nrow(X_s[[s]])))
})

fit_tl <- SSTL(
  X = X_T, Y = Y_T, X_s = X_s, Y_s = Y_s,
  family = "Gaussian", EB = TRUE,
  EB_control = list(N = 1000, burn = 500, warm_start = FALSE),
  N = 5000, burn = 1000, intercept = TRUE, verbose = 1
)
fit_tl$post_mean[1:10] # Target coefficients
```

## 3. Grouped transfer learning

Reuse the target and two source studies. In this simulation, put the four active predictors in group 1 and the remaining 196 in group 2. Group labels must be consecutive integers starting at 1.

```r
fit_group <- SSTL(
  X = X_T, Y = Y_T, X_s = X_s, Y_s = Y_s,
  group_map = c(rep(1, 4), rep(2, p - 4)),
  family = "Gaussian", EB = TRUE,
  EB_control = list(N = 1000, burn = 500, warm_start = FALSE),
  N = 5000, burn = 1000, intercept = TRUE, verbose = 1
)
fit_group$post_mean[1:10] # Target coefficients
```

Run the examples in order. `SSTL()` selects TO from target data alone, TL when source lists are supplied, and grouped TL when `group_map` is supplied. `EB = TRUE` estimates thresholds before the main MCMC; `EB = FALSE` skips EB. With `verbose = 1`, stage labels and progress bars show EB followed by MCMC. Intercepts are estimated separately, so do not add an intercept column.

## More details on model type options

All three model types support `"Gaussian"`, `"Logistic"` (binary 0/1), `"Student-t"` (fixed `df`, default 4), and `"Poisson"` (counts, log link). TO and TL also support `"Negative-binomial"` (counts, log link), `"Gamma"` (positive values, log link), and `"Beta"` (values strictly between 0 and 1, logit link). Family names are case-insensitive.

For survival outcomes, choose `"Weibull"`, `"Lognormal"`, or `"Loglogistic"`. Supply observed times with `log = TRUE` (default), or already-logged times with `log = FALSE`. `C` and `C_s` indicate events (1) versus right-censoring (0); omitting them treats observations as uncensored.

Results include `post_mean`, `post_median`, `post_sd`, `post_interval` (95% intervals), and `pip` for target coefficients, calculated after `burn`. Raw draws remain in `MC_beta` (iterations by predictors); `lambda` records the fitted thresholds and `EB` contains the EB result. Use multiple chains and check mixing for inference. Additional sampler options such as `df`, `slab`, `S.max`, `block_size`, and `debug` remain available; see `?SSTL` for details.
