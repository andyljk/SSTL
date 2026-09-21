#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "sstl_helpers.h"

using namespace Rcpp;
using namespace arma;

// ============================================================================
// FUNCTION 1: Update Target Parameters (beta_T)
// Impact: Changes Y_T likelihood AND ALL Y_S likelihoods
// ============================================================================

// [[Rcpp::export]]
List update_target_cpp(arma::vec bt_c, const arma::mat& X_T, const arma::vec& Y_T,
                       const Rcpp::List& X_S_list, const Rcpp::List& Y_S_list,
                       const arma::mat& bs_c, // Needed to calculate source residuals
                       const Rcpp::List& id, const arma::vec& sd_T,
                       double lambda_T, const arma::vec& lambda_S,
                       double tau, const arma::vec& tau_S,
                       double sd_y_T, const arma::vec& sd_y_S,
                       int S_max, int slab_code) {

  int p = X_T.n_cols;
  int S = X_S_list.size();
  int K = id.size();

  // Preallocate matrices to save speed
  std::vector<arma::mat> X_s_cpp(S);
  std::vector<arma::vec> Y_s_cpp(S);
  for(int s=0; s<S; s++) {
    X_s_cpp[s] = as<arma::mat>(X_S_list[s]);
    Y_s_cpp[s] = as<arma::vec>(Y_S_list[s]);
  }

  // Current Beta_T
  vec w_T = bt_c.subvec(0, p-1);
  vec a_T = bt_c.subvec(p, 2*p-1);
  double a0_T = bt_c(2*p);
  vec beta_T = sstl::get_beta(w_T, a_T, tau, a0_T, lambda_T, slab_code);

  // Initialize Target Residuals
  vec resid_T = Y_T - X_T * beta_T;
  double ll_T = sstl::log_lik_resid(resid_T, Y_T, sd_y_T);

  // Initialize Source Residuals (List of vectors)
  std::vector<vec> resid_S(S);
  double ll_S_total = 0;

  for(int s=0; s<S; s++) {
    // Construct Bias for Source s
    vec bs_col = bs_c.col(s);
    vec w_s = bs_col.subvec(0, p-1);
    vec a_s = bs_col.subvec(p, 2*p-1);
    double a0_s = bs_col(2*p);
    vec bias_s = sstl::get_beta(w_s, a_s, tau_S(s), a0_s, lambda_S(s), slab_code);

    // Residual = Y - X * (beta_T + bias)
    resid_S[s] = Y_s_cpp[s] - X_s_cpp[s] * (beta_T + bias_s);
    ll_S_total += sstl::log_lik_resid(resid_S[s], Y_s_cpp[s], sd_y_S(s));
  }

  double current_ll_global = ll_T + ll_S_total;
  vec N_s_out = zeros(K);

  // --- LOOP OVER BLOCKS ---
  for(int k=0; k<K; k++) {
    IntegerVector idx_r = id[k];
    uvec idx = as<uvec>(idx_r) - 1;

    // A. Setup ESS
    vec f_curr = bt_c.elem(idx);
    vec nu(idx.n_elem);
    for(int j=0; j<idx.n_elem; j++) nu(j) = R::rnorm(0, sd_T(idx(j)));

    double u_uni = R::runif(0, 1);
    double log_y_threshold = current_ll_global + log(u_uni);

    double theta = R::runif(0, 2 * M_PI);
    double theta_min = theta - 2 * M_PI;
    double theta_max = theta;

    int n_s = 0;

    // B. Slice Loop
    vec resid_T_prop = resid_T;
    std::vector<vec> resid_S_prop(S);
    for(int s = 0; s < S; s++) {
      resid_S_prop[s] = resid_S[s];
    }

    while(n_s < S_max) {
      n_s++;
      vec f_prop = f_curr * cos(theta) + nu * sin(theta);

      // Construct tentative parameters
      vec bt_prop = bt_c;
      bt_prop.elem(idx) = f_prop;

      vec w_prop = bt_prop.subvec(0, p-1);
      vec a_prop = bt_prop.subvec(p, 2*p-1);
      double a0_prop = bt_prop(2*p);

      vec beta_T_prop = sstl::get_beta(w_prop, a_prop, tau, a0_prop, lambda_T, slab_code);

      // --- C. GLOBAL RESIDUAL UPDATE ---
      vec delta_beta = beta_T_prop - beta_T;

      // 1. Update Target Residual
      resid_T_prop = resid_T;
      if (k == K - 1) { // Global a0 update
        resid_T_prop -= X_T * delta_beta;
      } else { // Sparse update
        // Assuming standard block structure: w_idx (first half) matches cols
        int half = idx.n_elem / 2;
        uword start = idx(0);
        uword end = idx(half-1);
        vec d_sub = delta_beta.subvec(start, end);
        resid_T_prop -= X_T.cols(start, end) * d_sub;
      }
      double ll_T_prop = sstl::log_lik_resid(resid_T_prop, Y_T, sd_y_T);

      // 2. Update Source Residuals
      double ll_S_prop_total = 0;


      for(int s=0; s<S; s++) {
        vec r_s_curr = resid_S[s]; // copy current resid

        // Apply same delta_beta to source
        if (k == K - 1) {
          resid_S_prop[s] -= X_s_cpp[s] * delta_beta;
        } else {
          int half = idx.n_elem / 2;
          uword start = idx(0);
          uword end = idx(half-1);
          vec d_sub = delta_beta.subvec(start, end);
          resid_S_prop[s] -= X_s_cpp[s].cols(start, end) * d_sub;
        }

        ll_S_prop_total += sstl::log_lik_resid(resid_S_prop[s], Y_s_cpp[s], sd_y_S(s));
      }

      double prop_ll_global = ll_T_prop + ll_S_prop_total;

      if(prop_ll_global > log_y_threshold) {
        // ACCEPT
        bt_c = bt_prop;
        beta_T = beta_T_prop;
        resid_T = resid_T_prop;
        resid_S = resid_S_prop; // update all source resids
        current_ll_global = prop_ll_global;
        break;
      } else {
        if(theta < 0) theta_min = theta;
        else theta_max = theta;
        theta = R::runif(theta_min, theta_max);
      }
    }
    N_s_out(k) = n_s;
  }

  return List::create(Named("bt_c") = bt_c, Named("N_t") = N_s_out);
}

// ============================================================================
// FUNCTION: Update Source Biases (Jointly across sources)
// Respects cov_W by updating rows of bs_c simultaneously
// ============================================================================

// [[Rcpp::export]]
List update_source_joint_cpp(arma::mat bs_c, // (2p+1) x S matrix
                             const Rcpp::List& X_s_list, const Rcpp::List& Y_s_list,
                             const arma::vec& beta_T, // Fixed Target Beta
                             const arma::vec& lambda_S, const arma::vec& tau_S,
                             const arma::vec& sd_y_S,
                             int S_max, int slab_code) {

  int p = (bs_c.n_rows - 1) / 2;
  int S = bs_c.n_cols;
  int n_params = bs_c.n_rows; // 2p + 1

  std::vector<arma::mat> X_s_cpp(S);
  std::vector<arma::vec> Y_s_cpp(S);
  for(int s=0; s<S; s++) {
    X_s_cpp[s] = as<arma::mat>(X_s_list[s]);
    Y_s_cpp[s] = as<arma::vec>(Y_s_list[s]);
  }

  // Pre-Calculate Current Residuals for ALL Sources
  std::vector<vec> resid_S(S);
  std::vector<double> ll_S(S);
  double current_ll_total = 0;

  for(int s=0; s<S; s++) {
    vec bias = sstl::calc_bias_vec(bs_c.col(s), tau_S(s), p, lambda_S(s), slab_code);

    // Residual = Y - X(beta_T + bias)
    resid_S[s] = Y_s_cpp[s] - X_s_cpp[s] * (beta_T + bias);
    ll_S[s] = sstl::log_lik_resid(resid_S[s], Y_s_cpp[s], sd_y_S(s));
    current_ll_total += ll_S[s];
  }

  vec N_s_out = zeros(n_params);

  // Loop over rows of bs_c
  for(int j=0; j<n_params; j++) {

    // Setup Ellipse
    vec nu = randn(S);
    // B. Slice Sampling Setup
    rowvec f_curr = bs_c.row(j); // The current row across all S

    double u_uni = R::runif(0, 1);
    double log_y_threshold = current_ll_total + log(u_uni);

    double theta = R::runif(0, 2 * M_PI);
    double theta_min = theta - 2 * M_PI;
    double theta_max = theta;

    int n_s = 0;

    std::vector<arma::vec> resid_S_prop(S);
    std::vector<double> precomputed_thresh(S, 0.0);
    if (j < 2 * p) {
      for (int s = 0; s < S; s++) {
        double a0_fixed = bs_c(2 * p, s);
        double lam = lambda_S(s);
        double thresh_prob = std::pow(sstl::pnorm_custom(a0_fixed), 1.0 / lam);
        precomputed_thresh[s] = sstl::qnorm_custom(thresh_prob);
      }
    }

    // C. Slice Loop
    while(n_s < S_max) {
      n_s++;

      // Propose New Row
      rowvec f_prop_row = f_curr * cos(theta) + trans(nu) * sin(theta);

      // Calculate Likelihood Delta
      double prop_ll_total = 0;

      for(int s=0; s<S; s++) {
        double val_new = f_prop_row(s);

        resid_S_prop[s] = resid_S[s];
        double lam = lambda_S(s);

        if (j == 2*p) {
          // CASE 1: Global a0 update (recompute full vector)
          vec bs_col = bs_c.col(s);
          vec bias_old = sstl::get_beta(bs_col.subvec(0, p-1),
                                  bs_col.subvec(p, 2*p-1), tau_S(s),
                                  bs_col(2*p), lam, slab_code);

          // Construct New Bias (Vector)
          vec bias_new = sstl::get_beta(bs_col.subvec(0, p-1),
                                  bs_col.subvec(p, 2*p-1), tau_S(s),
                                  val_new, lam, slab_code); // Use val_new for a0

          vec diff = bias_new - bias_old;
          resid_S_prop[s] -= X_s_cpp[s] * diff;

        } else {
          // --- CASE 2: Local w_k or a_k (Scalar Math) ---
          int k = j % p;
          double w_fixed = bs_c(k, s);
          double a_fixed = bs_c(k+p, s);

          // Determine Threshold
          double thresh = precomputed_thresh[s];

          // 1. Beta Old
          double beta_old_k = sstl::calc_scalar_beta(w_fixed, a_fixed, thresh, tau_S(s), slab_code);

          // 2. Beta New (Swap parameter)
          double w_temp = (j < p) ? val_new : w_fixed;
          double a_temp = (j < p) ? a_fixed : val_new;
          double beta_new_k = sstl::calc_scalar_beta(w_temp, a_temp, thresh, tau_S(s), slab_code);

          // 3. Update Residual
          double d_val = beta_new_k - beta_old_k;
          resid_S_prop[s] -= X_s_cpp[s].col(k) * d_val;
        }

        prop_ll_total += sstl::log_lik_resid(resid_S_prop[s], Y_s_cpp[s], sd_y_S(s));
      }

      if(prop_ll_total > log_y_threshold) {
        // ACCEPT: Update global state
        bs_c.row(j) = f_prop_row;
        resid_S = resid_S_prop;
        current_ll_total = prop_ll_total;
        break;
      } else {
        if(theta < 0) theta_min = theta;
        else theta_max = theta;
        theta = R::runif(theta_min, theta_max);
      }
    }
    N_s_out(j) = n_s;
  }

  return List::create(Named("bs_c") = bs_c, Named("N_s") = N_s_out);
}

// [[Rcpp::export]]
double update_target_scale_cpp(double xi_t_curr, // Scalar shadow variable
                               const double sd_0,
                               arma::vec bt_c,
                               const arma::mat& bs_c,
                               const arma::mat& X_T, const arma::vec& Y_T,
                               const Rcpp::List& X_s_list, const Rcpp::List& Y_s_list,
                               double lambda_T, const arma::vec& lambda_S,
                               double sd_y_T, const arma::vec& sd_y_S,
                               const arma::vec& tau_S, // Fixed source scales
                               int slab_code) {

  int p = (bt_c.n_elem - 1) / 2;
  int S = X_s_list.size();

  // Reference original numeric responses without copying their data.
  std::vector<arma::vec> Y_s_cpp;
  Y_s_cpp.reserve(S);
  for (int s = 0; s < S; ++s) {
    Rcpp::NumericVector Y_s = Y_s_list[s];
    Y_s_cpp.emplace_back(Y_s.begin(), Y_s.size(), false, true);
  }

  // 1. Setup Independent Gaussian Prior (Scalar)
  double nu = R::rnorm(0, sd_0);

  // 2. Pre-calculate Fixed Residuals and Direction Vectors

  // A. Target Direction (Unscaled Beta, scale=1.0)
  vec w_T = bt_c.subvec(0, p-1);
  vec a_T = bt_c.subvec(p, 2*p-1);
  double a0_T = bt_c(2*p);
  vec beta_T_raw = sstl::get_beta(w_T, a_T, 1.0, a0_T, lambda_T, slab_code);

  vec Z_T = X_T * beta_T_raw;       // Target Direction
  vec resid_fixed_T = Y_T;          // Target Fixed Residual (Y - 0)

  // B. Source Directions and Fixed Residuals
  std::vector<vec> Z_s_list(S);
  std::vector<vec> resid_fixed_s_list(S);

  double current_ll = 0;
  double tau_curr = std::abs(xi_t_curr);

  // Calculate Initial Target LL
  // Resid = Y - tau * Z
  vec resid = resid_fixed_T - tau_curr * Z_T;
  current_ll += sstl::log_lik_resid(resid, Y_T, sd_y_T);

  // Calculate Initial Source LLs
  for(int s=0; s<S; s++) {
    mat X_s = X_s_list[s];
    vec Y_s = Y_s_list[s];
    vec bs_col = bs_c.col(s);

    // Calculate Source Bias (Fixed during target scale update)
    vec bias_s = sstl::get_beta(bs_col.subvec(0, p-1), bs_col.subvec(p, 2*p-1),
                          tau_S(s), bs_col(2*p), lambda_S(s), slab_code);
    resid_fixed_s_list[s] = Y_s - X_s * bias_s; // Fixed part of residual: Y_s - X_s * bias_s
    Z_s_list[s] = X_s * beta_T_raw; // Variable direction: X_s * beta_T_raw
    resid = resid_fixed_s_list[s] - tau_curr * Z_s_list[s];
    current_ll += sstl::log_lik_resid(resid, Y_s_cpp[s], sd_y_S(s)); // Current Source LL
  }

  // 3. ESS Loop
  double u = R::runif(0, 1);
  double log_y_thresh = current_ll + log(u);

  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  double xi_prop = xi_t_curr;

  int iter = 0;
  while(true) {
    // Propose new shadow variable
    xi_prop = xi_t_curr * cos(theta) + nu * sin(theta);
    double tau_prop = std::abs(xi_prop);

    double prop_ll = 0;

    // Target LL
    resid = resid_fixed_T - tau_prop * Z_T;
    prop_ll += sstl::log_lik_resid(resid, Y_T, sd_y_T);

    // Source LLs
    for(int s=0; s<S; s++) {
      resid = resid_fixed_s_list[s] - tau_prop * Z_s_list[s];
      prop_ll += sstl::log_lik_resid(resid, Y_s_cpp[s], sd_y_S(s));
    }

    if(prop_ll > log_y_thresh) {
      break;
    } else {
      iter++;
      if (iter >= 20) {
        xi_prop = xi_t_curr; // Revert to current state (Reject)
        break;
      }
      if(theta < 0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
  }

  return xi_prop;
}

// [[Rcpp::export]]
arma::vec update_source_scales_cpp(arma::vec xi_s_curr, // Size S shadow variables
                             const double sd_0, // prior variance of scales
                             const arma::mat& bs_c,
                             const Rcpp::List& X_s_list, const Rcpp::List& Y_s_list,
                             const arma::vec& beta_T,
                             const arma::vec& lambda_S,
                             const arma::vec& sd_y_S,
                             int slab_code) {

  int S = xi_s_curr.n_elem;

  // Reference original numeric responses without copying their data.
  std::vector<arma::vec> Y_s_cpp;
  Y_s_cpp.reserve(S);
  for (int s = 0; s < S; ++s) {
    Rcpp::NumericVector Y_s = Y_s_list[s];
    Y_s_cpp.emplace_back(Y_s.begin(), Y_s.size(), false, true);
  }
  int p = (bs_c.n_rows - 1) / 2;

  // 1. Setup Independent Spherical Gaussian Prior
  vec nu = sd_0 * randn(S);

  // 2. Pre-calculate "Fixed" Residuals and "Direction" Vectors

  std::vector<vec> Z_list(S);          // The scalable direction vector
  std::vector<vec> resid_fixed_list(S); // The static part of residual
  double current_ll = 0;

  vec tau_curr = abs(xi_s_curr);

  // Calculate current likelihood & linear predictor bias
  for(int s=0; s<S; s++) {
    mat X = X_s_list[s];
    vec Y = Y_s_list[s];
    vec bs_col = bs_c.col(s);

    // Calculate unscaled bias
    vec bias_init = sstl::get_beta(bs_col.subvec(0, p-1), bs_col.subvec(p, 2*p-1),
                             1.0, bs_col(2*p), lambda_S(s), slab_code);
    Z_list[s] = X * bias_init; // Calculate linear predictor bias
    resid_fixed_list[s] = Y - X * beta_T; // Calculate fixed residual from target pars
    vec resid = resid_fixed_list[s] - tau_curr(s) * Z_list[s];
    current_ll += sstl::log_lik_resid(resid, Y_s_cpp[s], sd_y_S(s)); // Current Likelihood
  }

  // 3. ESS Loop
  double u = R::runif(0, 1);
  double log_y_thresh = current_ll + log(u);

  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  vec xi_prop = xi_s_curr;
  vec resid; // preallocate memory

  int iter = 0;
  while(true) {
    // Propose new shadow variables
    xi_prop = xi_s_curr * cos(theta) + nu * sin(theta);
    vec tau_prop = abs(xi_prop);

    double prop_ll = 0;
    for(int s=0; s<S; s++) {
      vec resid = resid_fixed_list[s] - tau_prop(s) * Z_list[s];
      prop_ll += sstl::log_lik_resid(resid, Y_s_cpp[s], sd_y_S(s));
    }

    if(prop_ll > log_y_thresh) {
      break;
    } else {
      iter++;
      if (iter >= 20) {
        xi_prop = xi_s_curr; // Revert to current state (Reject)
        break;
      }
      if(theta < 0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
  }

  return xi_prop;
}
