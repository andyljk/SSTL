// ess_fast.cpp
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "ntl_helpers.h"

using namespace Rcpp;
using namespace arma;

// --- MAIN FUNCTION ---

// [[Rcpp::export]]
List update_blocks_aft(arma::vec b_c, const arma::mat& X, const arma::vec& Y, const arma::vec& C,
                       const Rcpp::List& id, const arma::vec& sd_0,
                       double lambda, double tau, double sd_y, int S_max,
                       int fam_code, int slab_code) {

  int p = X.n_cols;
  int K = id.size();

  // Parse current state
  vec w = b_c.subvec(0, p-1);
  vec a = b_c.subvec(p, 2*p-1);
  double a0 = b_c(2*p);

  // Initialize Beta and Residuals ONCE
  vec beta = ntl::get_beta(w, a, tau, a0, lambda, slab_code);
  vec resid = Y - X * beta;
  double current_ll = ntl::log_lik_aft(resid, C, sd_y, fam_code);

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

    while(n_s < S_max) {
      n_s++;

      // 1. Propose new state
      vec f_prop = f_curr * cos(theta) + nu * sin(theta);

      // 2. Construct tentative beta
      // We update the full b_c vector temporarily
      vec b_prop = b_c;
      b_prop.elem(idx) = f_prop;

      vec w_prop = b_prop.subvec(0, p-1);
      vec a_prop = b_prop.subvec(p, 2*p-1);
      double a0_prop = b_prop(2*p);

      vec beta_prop = ntl::get_beta(w_prop, a_prop, tau, a0_prop, lambda, slab_code);

      // 3. Simplified Likelihood Update
      vec delta_beta = beta_prop - beta;
      vec resid_prop = resid;

      if (k == K - 1) {
        // CASE 1: Last block (a0) -> Global Update
        resid_prop = resid - X * delta_beta;

      } else {
        // CASE 2: Standard Block (w + a) -> Local Update

        int half_size = idx.n_elem / 2;
        uword start_col = idx(0);              // First w index
        uword end_col = idx(half_size - 1);    // Last w index

        vec d_sub = delta_beta.subvec(start_col, end_col);
        resid_prop -= X.cols(start_col, end_col) * d_sub;
      }

      double ll_prop = ntl::log_lik_aft(resid_prop, C, sd_y, fam_code);

      if(ll_prop > log_y_threshold) {
        // ACCEPT
        b_c = b_prop;
        beta = beta_prop;
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

  return List::create(Named("b_c") = b_c, Named("N_s") = N_s);
}


// [[Rcpp::export]]
double update_sigma_to_aft(arma::vec b_c, const arma::mat& X, const arma::vec& Y, const arma::vec& C,
                           double current_sigma, double lambda, double tau,
                           int fam_code, int slab_code, double step_size=0.1) {

  // Reconstruct Beta & Calculate Residuals
  int p = X.n_cols;
  vec w = b_c.subvec(0, p-1);
  vec a = b_c.subvec(p, 2*p-1);
  double a0 = b_c(2*p);

  vec beta = ntl::get_beta(w, a, tau, a0, lambda, slab_code);
  vec resid = Y - X * beta;

  // Metropolis-Hastings Step
  double log_sigma_curr = log(current_sigma);
  double log_sigma_prop = R::rnorm(log_sigma_curr, step_size);
  double sigma_prop     = exp(log_sigma_prop);

  double ll_curr = ntl::log_lik_aft(resid, C, current_sigma, fam_code);
  double ll_prop = ntl::log_lik_aft(resid, C, sigma_prop, fam_code);

  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return sigma_prop;
  } else {
    return current_sigma;
  }
}



// [[Rcpp::export]]
double update_scale_aft(double xi_curr, // Current Shadow Variable for Tau
                        arma::vec b_c, const arma::mat& X, const arma::vec& Y, const arma::vec& C,
                        double lambda, double sd_y, double sd_prior = 10.0,
                        int fam_code = 1, int slab_code = 1) {

  // 1. Pre-calculate the Unscaled Linear Predictor (Z)
  int p = X.n_cols;
  vec w = b_c.subvec(0, p-1);
  vec a = b_c.subvec(p, 2*p-1);
  double a0 = b_c(2*p);

  // Pass 1.0 for tau to get the raw direction
  vec beta_unscaled = ntl::get_beta(w, a, 1.0, a0, lambda, slab_code);
  vec Z = X * beta_unscaled;

  // 2. Setup ESS for the Shadow Variable (tau)
  double nu = R::rnorm(0, sd_prior);

  // Initial Likelihood
  double current_scale = std::abs(xi_curr);
  vec resid = Y - current_scale * Z;
  double current_ll = ntl::log_lik_aft(resid, C, sd_y, fam_code);

  // Threshold
  double u = R::runif(0, 1);
  double log_y_thresh = current_ll + log(u);

  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  double xi_prop = xi_curr;

  // 3. ESS Loop
  int iter = 0;
  while(true) {
    // Propose new Shadow Variable on the ellipse
    xi_prop = xi_curr * cos(theta) + nu * sin(theta);
    double scale_prop = std::abs(xi_prop);

    // Fast Residual Update (Vector Subtraction only)
    vec resid_prop = Y - scale_prop * Z;
    double prop_ll = ntl::log_lik_aft(resid_prop, C, sd_y, fam_code);

    if(prop_ll > log_y_thresh) {
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

  return xi_prop; // Return the new shadow variable
}
