#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "ntl_helpers.h"

using namespace Rcpp;
using namespace arma;

// ============================================================================
// FUNCTION 1: Update Target Parameters (beta_T)
// Impact: Changes Y_T likelihood AND ALL Y_S likelihoods
// ============================================================================

// [[Rcpp::export]]
List update_target_aft(arma::vec bt_c, arma::vec resid_T, const Rcpp::List& resid_S_list,
                       const arma::mat& X_T, const arma::vec& C_T,
                       const Rcpp::List& X_S_list, const Rcpp::List& C_S_list,
                       const Rcpp::List& id, const arma::vec& sd_T,
                       double lambda_T, double tau,
                       double sd_y_T, const arma::vec& sd_y_S,
                       int S_max, int fam_code, int slab_code,
                       bool approx = false, double k_apx = 10.0) {

  int p = X_T.n_cols;
  int S = X_S_list.size();
  int K = id.size();

  // Preallocate matrices to save speed
  std::vector<arma::mat> X_s_cpp(S);
  std::vector<arma::vec> C_s_cpp(S);
  std::vector<arma::vec> resid_S(S);
  for(int s=0; s<S; s++) {
    X_s_cpp[s] = as<arma::mat>(X_S_list[s]);
    C_s_cpp[s] = as<arma::vec>(C_S_list[s]);
    resid_S[s] = as<arma::vec>(resid_S_list[s]);
  }

  // Current Beta_T
  vec w_T = bt_c.subvec(0, p-1);
  vec a_T = bt_c.subvec(p, 2*p-1);
  double a0_T = bt_c(2*p);
  vec beta_T = ntl::get_beta(w_T, a_T, tau, a0_T, lambda_T, slab_code, approx, k_apx);

  double ll_T = ntl::log_lik_aft(resid_T, C_T, sd_y_T, fam_code);

  double ll_S_total = 0;
  for(int s=0; s<S; s++) {
    ll_S_total += ntl::log_lik_aft(resid_S[s], C_s_cpp[s], sd_y_S(s), fam_code);
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

      vec beta_T_prop = ntl::get_beta(w_prop, a_prop, tau, a0_prop, lambda_T, slab_code, approx, k_apx);

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
      double ll_T_prop = ntl::log_lik_aft(resid_T_prop, C_T, sd_y_T, fam_code);

      // 2. Update Source Residuals
      double ll_S_prop_total = 0;


      for(int s=0; s<S; s++) {
        resid_S_prop[s] = resid_S[s]; // copy current resid

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

        ll_S_prop_total += ntl::log_lik_aft(resid_S_prop[s], C_s_cpp[s], sd_y_S(s), fam_code);
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

  Rcpp::List resid_S_out(S);
  for(int s = 0; s < S; s++) resid_S_out[s] = resid_S[s];

  return List::create(Named("bt_c") = bt_c,
                      Named("resid_T") = resid_T,
                      Named("resid_S") = resid_S_out,
                      Named("N_t") = N_s_out);
}

// ============================================================================
// FUNCTION: Update Source Biases (Jointly across sources)
// Respects cov_W by updating rows of bs_c simultaneously
// ============================================================================

// [[Rcpp::export]]
List update_source_joint_aft(arma::mat bs_c, // (2p+1) x S matrix
                             const Rcpp::List& resid_S_list,
                             const Rcpp::List& X_s_list, const Rcpp::List& C_s_list,
                             const arma::vec& lambda_S, const arma::vec& tau_S,
                             const arma::vec& sd_y_S,
                             int S_max, int fam_code, int slab_code,
                             bool approx = false, double k_apx = 10.0) {

  int p = (bs_c.n_rows - 1) / 2;
  int S = bs_c.n_cols;
  int n_params = bs_c.n_rows; // 2p + 1

  std::vector<arma::mat> X_s_cpp(S);
  std::vector<arma::vec> C_s_cpp(S);
  std::vector<arma::vec> resid_S(S);
  for(int s=0; s<S; s++) {
    X_s_cpp[s] = as<arma::mat>(X_s_list[s]);
    C_s_cpp[s] = as<arma::vec>(C_s_list[s]);
    resid_S[s] = as<arma::vec>(resid_S_list[s]);
  }

  std::vector<double> ll_S(S);
  double current_ll_total = 0;

  for(int s=0; s<S; s++) {
    ll_S[s] = ntl::log_lik_aft(resid_S[s], C_s_cpp[s], sd_y_S(s), fam_code);
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
        double thresh_prob = std::pow(ntl::pnorm_custom(a0_fixed), 1.0 / lam);
        precomputed_thresh[s] = ntl::qnorm_custom(thresh_prob);
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
          vec bias_old = ntl::get_beta(bs_col.subvec(0, p-1),
                                       bs_col.subvec(p, 2*p-1), tau_S(s),
                                       bs_col(2*p), lam, slab_code,
                                       approx, k_apx);

          // Construct New Bias (Vector)
          vec bias_new = ntl::get_beta(bs_col.subvec(0, p-1),
                                       bs_col.subvec(p, 2*p-1), tau_S(s),
                                       val_new, lam, slab_code,
                                       approx, k_apx); // Use val_new for a0

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
          double beta_old_k = ntl::calc_scalar_beta(w_fixed, a_fixed, thresh, tau_S(s), slab_code, approx, k_apx);

          // 2. Beta New (Swap parameter)
          double w_temp = (j < p) ? val_new : w_fixed;
          double a_temp = (j < p) ? a_fixed : val_new;
          double beta_new_k = ntl::calc_scalar_beta(w_temp, a_temp, thresh, tau_S(s), slab_code, approx, k_apx);

          // 3. Update Residual
          double d_val = beta_new_k - beta_old_k;
          resid_S_prop[s] -= X_s_cpp[s].col(k) * d_val;
        }

        prop_ll_total += ntl::log_lik_aft(resid_S_prop[s], C_s_cpp[s], sd_y_S(s), fam_code);
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

  Rcpp::List resid_S_out(S);
  for(int s = 0; s < S; s++) resid_S_out[s] = resid_S[s];

  return List::create(Named("bs_c") = bs_c,
                      Named("resid_S") = resid_S_out,
                      Named("N_s") = N_s_out);
}

// [[Rcpp::export]]
List update_target_scale_aft(double xi_t_curr, // Scalar shadow variable
                             const double sd_0,
                             const arma::vec& Y_T, arma::vec resid_T,
                             const Rcpp::List& Y_s_list, const Rcpp::List& resid_S_list,
                             arma::vec bt_c, const arma::mat& X_T, const arma::vec& C_T,
                             double lambda_T,
                             const Rcpp::List& X_s_list, const Rcpp::List& C_s_list,
                             double sd_y_T, const arma::vec& sd_y_S,
                             int fam_code, int slab_code,
                             bool approx = false, double k_apx = 10.0) {

  int S = X_s_list.size();

  // 1. Setup Independent Gaussian Prior (Scalar)
  double nu = R::rnorm(0, sd_0);

  double tau_curr = std::abs(xi_t_curr);
  std::vector<arma::vec> resid_S(S);
  std::vector<arma::vec> Y_s_cpp(S);
  std::vector<arma::vec> C_s_cpp(S);

  for(int s=0; s<S; s++) {
    Y_s_cpp[s] = as<arma::vec>(Y_s_list[s]);
    C_s_cpp[s] = as<arma::vec>(C_s_list[s]);
    resid_S[s] = as<arma::vec>(resid_S_list[s]);
  }

  double current_ll = ntl::log_lik_aft(resid_T, C_T, sd_y_T, fam_code);
  for(int s=0; s<S; s++) {
    current_ll += ntl::log_lik_aft(resid_S[s], C_s_cpp[s], sd_y_S(s), fam_code);
  }

  // 3. ESS Loop
  double u = R::runif(0, 1);
  double log_y_thresh = current_ll + log(u);

  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  double xi_prop = xi_t_curr;
  vec resid_T_prop = resid_T;
  std::vector<vec> resid_S_prop(S);

  int iter = 0;
  while(true) {
    // Propose new shadow variable
    xi_prop = xi_t_curr * cos(theta) + nu * sin(theta);
    double tau_prop = std::abs(xi_prop);

    double prop_ll = 0;

    resid_T_prop = Y_T - (Y_T - resid_T) * (tau_prop / tau_curr);
    prop_ll += ntl::log_lik_aft(resid_T_prop, C_T, sd_y_T, fam_code);

    for(int s=0; s<S; s++) {
      resid_S_prop[s] = Y_s_cpp[s] - (Y_s_cpp[s] - resid_S[s]) * (tau_prop / tau_curr);
      prop_ll += ntl::log_lik_aft(resid_S_prop[s], C_s_cpp[s], sd_y_S(s), fam_code);
    }

    if(prop_ll > log_y_thresh) {
      resid_T = resid_T_prop;
      resid_S = resid_S_prop;
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

  Rcpp::List resid_S_out(S);
  for(int s = 0; s < S; s++) resid_S_out[s] = resid_S[s];

  return List::create(Named("xi_t") = xi_prop,
                      Named("resid_T") = resid_T,
                      Named("resid_S") = resid_S_out);
}

// [[Rcpp::export]]
List update_source_scales_aft(arma::vec xi_s_curr, // Size S shadow variables
                              const double sd_0, // prior variance of scales
                              const Rcpp::List& Y_s_list,
                              const Rcpp::List& resid_S_list,
                              const arma::mat& bs_c,
                              const Rcpp::List& X_s_list, const Rcpp::List& C_s_list,
                              const arma::vec& lambda_S,
                              const arma::vec& sd_y_S,
                              int fam_code, int slab_code,
                              bool approx = false, double k_apx = 10.0) {

  int S = xi_s_curr.n_elem;

  // 1. Setup Independent Spherical Gaussian Prior
  vec nu = sd_0 * randn(S);

  std::vector<vec> resid_S(S);
  std::vector<vec> Y_s_cpp(S);
  std::vector<arma::vec> C_s_cpp(S);

  vec tau_curr = abs(xi_s_curr);
  double current_ll = 0;

  for(int s=0; s<S; s++) {
    Y_s_cpp[s] = as<arma::vec>(Y_s_list[s]);
    C_s_cpp[s] = as<arma::vec>(C_s_list[s]);
    resid_S[s] = as<arma::vec>(resid_S_list[s]);
    current_ll += ntl::log_lik_aft(resid_S[s], C_s_cpp[s], sd_y_S(s), fam_code);
  }

  // 3. ESS Loop
  double u = R::runif(0, 1);
  double log_y_thresh = current_ll + log(u);

  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  vec xi_prop = xi_s_curr;
  std::vector<vec> resid_S_prop(S);

  int iter = 0;
  while(true) {
    // Propose new shadow variables
    xi_prop = xi_s_curr * cos(theta) + nu * sin(theta);
    vec tau_prop = abs(xi_prop);

    double prop_ll = 0;
    for(int s=0; s<S; s++) {
      resid_S_prop[s] = Y_s_cpp[s] - (Y_s_cpp[s] - resid_S[s]) * (tau_prop(s) / tau_curr(s));
      prop_ll += ntl::log_lik_aft(resid_S_prop[s], C_s_cpp[s], sd_y_S(s), fam_code);
    }

    if(prop_ll > log_y_thresh) {
      resid_S = resid_S_prop;
      break;
    } else {
      iter++;
      if (iter > 20) {
        xi_prop = xi_s_curr; // Revert to current state (Reject)
        break;
      }
      if(theta < 0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
  }

  Rcpp::List resid_S_out(S);
  for(int s = 0; s < S; s++) resid_S_out[s] = resid_S[s];

  return List::create(Named("xi_s") = xi_prop,
                      Named("resid_S") = resid_S_out);
}

// [[Rcpp::export]]
List update_target_intercept_tl_aft(double b0_T_curr, arma::vec resid_T, const arma::vec& C_T,
                                    double sd_y_T, int fam_code,
                                    double sd_prior = 10.0) {

  double nu = R::rnorm(0, sd_prior);
  double current_ll = ntl::log_lik_aft(resid_T, C_T, sd_y_T, fam_code);

  double u = R::runif(0, 1);
  double log_y_thresh = current_ll + log(u);

  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  double b0_T_prop = b0_T_curr;
  vec resid_T_prop(resid_T.n_elem);

  int iter = 0;
  while(true) {
    b0_T_prop = b0_T_curr * cos(theta) + nu * sin(theta);
    resid_T_prop = resid_T - (b0_T_prop - b0_T_curr);
    double prop_ll = ntl::log_lik_aft(resid_T_prop, C_T, sd_y_T, fam_code);

    if(prop_ll > log_y_thresh) {
      resid_T = resid_T_prop;
      break;
    } else {
      iter++;
      if (iter >= 20) {
        b0_T_prop = b0_T_curr;
        break;
      }
      if(theta < 0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
  }

  return List::create(Named("b0_T") = b0_T_prop,
                      Named("resid_T") = resid_T);
}

// [[Rcpp::export]]
List update_source_intercepts_tl_aft(arma::vec b0_s_curr,
                                     const Rcpp::List& resid_S_list, const Rcpp::List& C_s_list,
                                     const arma::vec& sd_y_S, int fam_code,
                                     double sd_prior = 10.0) {

  int S = b0_s_curr.n_elem;
  std::vector<arma::vec> resid_S(S);
  std::vector<arma::vec> C_s_cpp(S);
  for(int s = 0; s < S; s++) {
    resid_S[s] = as<arma::vec>(resid_S_list[s]);
    C_s_cpp[s] = as<arma::vec>(C_s_list[s]);
  }

  for(int s = 0; s < S; s++) {
    double nu = R::rnorm(0, sd_prior);
    double current_ll = ntl::log_lik_aft(resid_S[s], C_s_cpp[s], sd_y_S(s), fam_code);

    double u = R::runif(0, 1);
    double log_y_thresh = current_ll + log(u);

    double theta = R::runif(0, 2 * M_PI);
    double theta_min = theta - 2 * M_PI;
    double theta_max = theta;

    double b0_prop = b0_s_curr(s);
    vec resid_prop(resid_S[s].n_elem);

    int iter = 0;
    while(true) {
      b0_prop = b0_s_curr(s) * cos(theta) + nu * sin(theta);
      resid_prop = resid_S[s] - (b0_prop - b0_s_curr(s));
      double prop_ll = ntl::log_lik_aft(resid_prop, C_s_cpp[s], sd_y_S(s), fam_code);

      if(prop_ll > log_y_thresh) {
        b0_s_curr(s) = b0_prop;
        resid_S[s] = resid_prop;
        break;
      } else {
        iter++;
        if (iter >= 20) {
          break;
        }
        if(theta < 0) theta_min = theta;
        else theta_max = theta;
        theta = R::runif(theta_min, theta_max);
      }
    }
  }

  Rcpp::List resid_S_out(S);
  for(int s = 0; s < S; s++) resid_S_out[s] = resid_S[s];

  return List::create(Named("b0_s") = b0_s_curr,
                      Named("resid_S") = resid_S_out);
}

// Internal Helper for MH Step
double update_sigma_jeffreys_tl_aft(const arma::vec& resid, const arma::vec& C, double current_sigma, double step_size, int fam_code) {
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

// Update Target Sigma
// [[Rcpp::export]]
double update_sigma_target_tl_aft(const arma::vec& resid, const arma::vec& C,
                                  double current_sigma, int fam_code,
                                  double step_size=0.1) {
  return update_sigma_jeffreys_tl_aft(resid, C, current_sigma, step_size, fam_code);
}

// Update Source Sigma (Per Source)
// [[Rcpp::export]]
double update_sigma_source_tl_aft(const arma::vec& resid, const arma::vec& C,
                                  double current_sigma, int fam_code,
                                  double step_size=0.1) {
  return update_sigma_jeffreys_tl_aft(resid, C, current_sigma, step_size, fam_code);
}
