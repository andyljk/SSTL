#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "ntl_helpers.h"

using namespace Rcpp;
using namespace arma;

// --- MAIN FUNCTION ---

// [[Rcpp::export]]
List update_blocks_cpp(arma::vec b_c, const arma::mat& X, const arma::vec& Y,
                       const Rcpp::List& id, const arma::vec& sd_0,
                       double lambda, double tau, double sd_y, int S_max,
                       int slab_code) {

  int p = X.n_cols;
  int K = id.size();

  // Parse current state
  vec w = b_c.subvec(0, p-1);
  vec a = b_c.subvec(p, 2*p-1);
  double a0 = b_c(2*p);

  // Initialize Beta and Residuals ONCE
  vec beta = ntl::get_beta(w, a, tau, a0, lambda, slab_code);
  vec resid = Y - X * beta;
  double current_ll = ntl::log_lik_resid(resid, sd_y);

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

        beta_prop = ntl::get_beta(w_prop, a_prop, tau, a0_prop, lambda, slab_code);
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
        beta_prop = ntl::get_beta(w_sub_prop, a_sub_prop, tau, a0, lambda, slab_code);

        // Extract the current beta for these specific columns
        vec beta_sub_curr = beta.subvec(start_col, end_col);
        vec d_sub = beta_prop - beta_sub_curr;
        resid_prop -= X.cols(start_col, end_col) * d_sub;
      }

      double ll_prop = ntl::log_lik_resid(resid_prop, sd_y);

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

  return List::create(Named("b_c") = b_c, Named("N_s") = N_s);
}
