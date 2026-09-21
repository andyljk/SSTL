// ess_fast.cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "sstl_helpers.h"

using namespace Rcpp;
using namespace arma;

// --- MAIN FUNCTION ---

// [[Rcpp::export]]
List update_blocks_general(arma::vec b_c, arma::vec resid, const arma::mat& X, const arma::vec& Y,
                       const Rcpp::List& id, const arma::vec& sd_0,
                       double lambda, double tau, double sd_y, int S_max,
                       int fam_code, int slab_code, double df = 4.0) {

  int p = X.n_cols;
  int K = id.size();

  // Parse current state
  vec w = b_c.subvec(0, p-1);
  vec a = b_c.subvec(p, 2*p-1);
  double a0 = b_c(2*p);

  // Reconstruct the current slope effects once; residual is passed in
  vec beta = sstl::get_beta(w, a, tau, a0, lambda, slab_code);
  double current_ll = sstl::log_lik_general(resid, Y, sd_y, fam_code, df);

  vec N_s = zeros(K);

  for(int k = 0; k < K; k++) {
    // Get parameter indices for this block (0-based)
    IntegerVector idx_r = id[k];
    uvec idx = as<uvec>(idx_r) - 1;

    // --- A. SETUP ESS ---
    vec f_curr = b_c.elem(idx);

    vec nu(idx.n_elem);
    for(int j=0; j<idx.n_elem; j++) nu(j) = R::rnorm(0, sd_0(idx(j)));

    double u_uni = R::runif(0, 1);
    double log_y_threshold = current_ll + log(u_uni);

    double theta = R::runif(0, 2 * M_PI);
    double theta_min = theta - 2 * M_PI;
    double theta_max = theta;

    // --- B. SLICE LOOP ---
    int n_s = 0;

    // PRE-ALLOCATE OUTSIDE THE LOOP
    vec resid_prop = resid;
    vec beta_prop;
    uword start_col = 0, end_col = 0;

    while(n_s < S_max) {
      n_s++;

      // 1. Propose new state
      vec f_prop = f_curr * cos(theta) + nu * sin(theta);
      resid_prop = resid;

      if (k == K - 1) {
        // CASE 1: Last block (a0) -> Global Update
        vec b_prop = b_c;
        b_prop.elem(idx) = f_prop;

        vec w_prop = b_prop.subvec(0, p-1);
        vec a_prop = b_prop.subvec(p, 2*p-1);
        double a0_prop = b_prop(2*p);

        beta_prop = sstl::get_beta(w_prop, a_prop, tau, a0_prop, lambda, slab_code);
        vec delta_beta = beta_prop - beta;

        resid_prop -= X * delta_beta;

      } else {
        // CASE 2: Standard Block (w + a) -> Local Update
        int half_size = idx.n_elem / 2;
        start_col = idx(0) % p;
        end_col = idx(half_size - 1) % p;

        // Extract the proposed w and a ONLY for this specific block
        vec w_sub_prop = f_prop.subvec(0, half_size - 1);
        vec a_sub_prop = f_prop.subvec(half_size, idx.n_elem - 1);
        beta_prop = sstl::get_beta(w_sub_prop, a_sub_prop, tau, a0, lambda, slab_code);

        // Extract the current beta for these specific columns
        vec beta_sub_curr = beta.subvec(start_col, end_col);
        vec d_sub = beta_prop - beta_sub_curr;
        resid_prop -= X.cols(start_col, end_col) * d_sub;
      }

      double ll_prop = sstl::log_lik_general(resid_prop, Y, sd_y, fam_code, df);

      if(ll_prop > log_y_threshold) {
        // ACCEPT
        b_c.elem(idx) = f_prop;
        if (k == K - 1) {
          beta = beta_prop;
        } else {
          beta.subvec(start_col, end_col) = beta_prop;
        }
        resid = resid_prop;
        current_ll = ll_prop;
        break;
      } else {
        // REJECT
        if(theta < 0) theta_min = theta;
        else theta_max = theta;
        theta = R::runif(theta_min, theta_max);
      }
    }
    N_s(k) = n_s;
  }

  return List::create(Named("b_c") = b_c,
                      Named("resid") = resid,
                      Named("N_s") = N_s);
}


// [[Rcpp::export]]
double update_sigma_to_general(const arma::vec& resid, const arma::vec& Y,
                           double current_sigma, int fam_code,
                           double step_size=0.1, double df = 4.0) {
  if (fam_code == 2 || fam_code == 4) return current_sigma; // Logistic and Poisson regression have no outcome scale.

  // Metropolis-Hastings Step
  double log_sigma_curr = log(current_sigma);
  double log_sigma_prop = R::rnorm(log_sigma_curr, step_size);
  double sigma_prop     = exp(log_sigma_prop);

  double ll_curr = sstl::log_lik_general(resid, Y, current_sigma, fam_code, df);
  double ll_prop = sstl::log_lik_general(resid, Y, sigma_prop, fam_code, df);

  // Prior ratio including the log-proposal Jacobian; Jeffreys prior cancels for Gaussian/Student-t.
  double log_prior_ratio = 0.0;
  if (fam_code == 5) { // Inverse-Gamma(0.4, 0.3) for negative-binomial shape
    log_prior_ratio = -0.4 * (log_sigma_prop - log_sigma_curr) - 0.3 * (1.0 / sigma_prop - 1.0 / current_sigma);
  } else if (fam_code == 6 || fam_code == 7) { // Gamma(0.01, 0.01) for gamma shape/beta precision
    log_prior_ratio = 0.01 * (log_sigma_prop - log_sigma_curr) - 0.01 * (sigma_prop - current_sigma);
  }

  if (log(R::runif(0, 1)) < (ll_prop - ll_curr + log_prior_ratio)) {
    return sigma_prop;
  } else {
    return current_sigma;
  }
}



// [[Rcpp::export]]
List update_scale_general(double xi_curr, // Current Shadow Variable for Tau
                      const arma::vec& Y_scale, arma::vec resid, const arma::vec& Y,
                      double sd_y, double sd_prior = 10.0,
                      int fam_code = 1, double df = 4.0) {

  // 1. Setup ESS for the shadow variable
  double nu = R::rnorm(0, sd_prior);

  // Initial Likelihood
  double current_scale = std::exp(xi_curr);
  double current_ll = sstl::log_lik_general(resid, Y, sd_y, fam_code, df);

  // Threshold
  double u = R::runif(0, 1);
  double log_y_thresh = current_ll + log(u);

  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  double xi_prop = xi_curr;

  vec resid_prop(resid.n_elem);

  // 3. ESS Loop
  int iter = 0;
  while(true) {
    // Propose new Shadow Variable on the ellipse
    xi_prop = xi_curr * cos(theta) + nu * sin(theta);
    double scale_prop = std::exp(xi_prop);

    // Y_scale excludes the intercept; Y is the original response for the likelihood.
    // Since resid = Y_scale - current_scale * Z, we have (Y_scale - resid) = current_scale * Z
    resid_prop = Y_scale - (Y_scale - resid) * (scale_prop / current_scale);
    double prop_ll = sstl::log_lik_general(resid_prop, Y, sd_y, fam_code, df);

    if(prop_ll > log_y_thresh) {
      resid = resid_prop;
      break;
    } else {
      iter++;
      if (iter >= 20) {
        xi_prop = xi_curr; // Revert to current state (Reject)
        break;
      }
      // Shrink the bracket
      if(theta < 0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
  }

  return List::create(Named("xi") = xi_prop,
                      Named("resid") = resid);
}


// [[Rcpp::export]]
List update_intercept_to_general(double b0_curr, arma::vec resid, const arma::vec& Y,
                             double sd_y, int fam_code,
                             double sd_prior = 10.0, double df = 4.0) {

  // Current residual is assumed to be Y - b0_curr - X * beta
  double nu = R::rnorm(0, sd_prior);

  double current_ll = sstl::log_lik_general(resid, Y, sd_y, fam_code, df);

  double u = R::runif(0, 1);
  double log_y_thresh = current_ll + log(u);

  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  double b0_prop = b0_curr;
  vec resid_prop(resid.n_elem);

  // 3. ESS Loop
  int iter = 0;
  while(true) {
    b0_prop = b0_curr * cos(theta) + nu * sin(theta);

    // Intercept updates only shift the residual by the proposed intercept change
    resid_prop = resid - (b0_prop - b0_curr);
    double prop_ll = sstl::log_lik_general(resid_prop, Y, sd_y, fam_code, df);

    if(prop_ll > log_y_thresh) {
      resid = resid_prop;
      break;
    } else {
      iter++;
      if (iter >= 20) {
        b0_prop = b0_curr; // Revert to current state (Reject)
        break;
      }
      if(theta < 0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
  }

  return List::create(Named("b0") = b0_prop,
                      Named("resid") = resid);
}
