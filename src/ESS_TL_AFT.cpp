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
List update_target_aft(arma::vec bt_c, arma::vec resid_T, const Rcpp::List& resid_S_list,
                       const arma::mat& X_T, const arma::vec& Y_T, const arma::vec& C_T,
                       const Rcpp::List& X_S_list, const Rcpp::List& Y_s_list, const Rcpp::List& C_S_list,
                       const Rcpp::List& id, const arma::vec& sd_T,
                       double lambda_T, double tau,
                       double sd_y_T, const arma::vec& sd_y_S,
                       int S_max, int fam_code, int slab_code) {

  int p = X_T.n_cols;
  int S = X_S_list.size();
  int K = id.size();

  std::vector<arma::mat> X_s_cpp = sstl::mat_views(X_S_list);
  std::vector<arma::vec> Y_s_cpp = sstl::vec_views(Y_s_list);
  std::vector<arma::vec> C_s_cpp = sstl::vec_views(C_S_list);
  std::vector<arma::vec> resid_S(S), resid_S_prop(S);
  for (int s = 0; s < S; s++) resid_S[s] = as<arma::vec>(resid_S_list[s]);

  // Current Beta_T; the threshold only changes in the final (a0) block
  double a0_T = bt_c(2*p);
  double threshold = sstl::threshold_from_a0(a0_T, lambda_T);
  vec beta_T = sstl::get_beta(bt_c.subvec(0, p-1), bt_c.subvec(p, 2*p-1), tau, a0_T, lambda_T, slab_code);

  double ll_T = sstl::log_lik_aft(resid_T, Y_T, C_T, sd_y_T, fam_code);
  double ll_S_total = 0;
  for (int s = 0; s < S; s++) {
    ll_S_total += sstl::log_lik_aft(resid_S[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
  }

  double current_ll_global = ll_T + ll_S_total;
  vec N_s_out = zeros(K);
  vec resid_T_prop;

  // --- LOOP OVER BLOCKS ---
  for (int k = 0; k < K; k++) {
    IntegerVector idx_r = id[k];
    uvec idx = as<uvec>(idx_r) - 1;
    bool is_a0 = (k == K - 1);

    // A. Setup ESS
    vec f_curr = bt_c.elem(idx);
    vec nu(idx.n_elem);
    for (uword j = 0; j < idx.n_elem; j++) nu(j) = R::rnorm(0, sd_T(idx(j)));

    double u_uni = R::runif(0, 1);
    double log_y_threshold = current_ll_global + log(u_uni);

    double theta = R::runif(0, 2 * M_PI);
    double theta_min = theta - 2 * M_PI;
    double theta_max = theta;

    int n_s = 0;

    // Standard blocks hold w (first half) and a (second half) for columns start..start+half-1
    int half = idx.n_elem / 2;
    uword start = is_a0 ? 0 : idx(0);
    vec beta_prop, delta_beta, tau_h;
    uvec nz;
    if (is_a0) tau_h = tau * sstl::slab_weight_vec(bt_c.subvec(0, p-1), slab_code);

    // B. Slice Loop
    while (n_s < S_max) {
      n_s++;
      vec f_prop = f_curr * cos(theta) + nu * sin(theta);

      // Coefficient change implied by the proposal (only the block's columns, or all for a0)
      bool changed = false;
      if (!is_a0) {
        beta_prop.set_size(half);
        delta_beta.set_size(half);
        for (int j = 0; j < half; j++) {
          beta_prop(j) = sstl::calc_scalar_beta(f_prop(j), f_prop(half + j), threshold, tau, slab_code);
          delta_beta(j) = beta_prop(j) - beta_T(start + j);
          if (delta_beta(j) != 0.0) changed = true;
        }
      } else {
        double threshold_prop = sstl::threshold_from_a0(f_prop(0), lambda_T);
        const double* a_ptr = bt_c.memptr() + p;
        beta_prop.set_size(p);
        for (int j = 0; j < p; j++) {
          beta_prop(j) = tau_h(j) * sstl::activation_scalar(a_ptr[j], threshold_prop, slab_code);
        }
        delta_beta = beta_prop - beta_T;
        nz = find(delta_beta != 0.0);
        changed = nz.n_elem > 0;
      }

      // An unchanged beta leaves every likelihood unchanged, so ESS accepts without evaluating it.
      double prop_ll_global = current_ll_global;
      if (changed) {
        // C. Residual updates for the target and every source
        if (is_a0) resid_T_prop = resid_T - X_T.cols(nz) * delta_beta.elem(nz);
        else       resid_T_prop = resid_T - X_T.cols(start, start + half - 1) * delta_beta;
        double ll_T_prop = sstl::log_lik_aft(resid_T_prop, Y_T, C_T, sd_y_T, fam_code);

        double ll_S_prop_total = 0;
        for (int s = 0; s < S; s++) {
          if (is_a0) resid_S_prop[s] = resid_S[s] - X_s_cpp[s].cols(nz) * delta_beta.elem(nz);
          else       resid_S_prop[s] = resid_S[s] - X_s_cpp[s].cols(start, start + half - 1) * delta_beta;
          ll_S_prop_total += sstl::log_lik_aft(resid_S_prop[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
        }
        prop_ll_global = ll_T_prop + ll_S_prop_total;
      }

      if (!changed || prop_ll_global > log_y_threshold) {
        // ACCEPT
        bt_c.elem(idx) = f_prop;
        if (changed) {
          if (is_a0) {
            beta_T = beta_prop;
            threshold = sstl::threshold_from_a0(f_prop(0), lambda_T);
          } else {
            beta_T.subvec(start, start + half - 1) = beta_prop;
          }
          resid_T.swap(resid_T_prop);
          for (int s = 0; s < S; s++) resid_S[s].swap(resid_S_prop[s]);
          current_ll_global = prop_ll_global;
        }
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
// FUNCTION: Update Source Biases
// Sources are conditionally independent given the target, so each source's
// latent parameters are updated in turn by one-dimensional ESS steps that
// involve only that source's likelihood.
// ============================================================================

// [[Rcpp::export]]
List update_source_joint_aft(arma::mat bs_c, // (2p+1) x S matrix
                             const Rcpp::List& resid_S_list,
                             const Rcpp::List& X_s_list, const Rcpp::List& Y_s_list, const Rcpp::List& C_s_list,
                             const arma::vec& lambda_S, const arma::vec& tau_S,
                             const arma::vec& sd_y_S,
                             int S_max, int fam_code, int slab_code) {

  int p = (bs_c.n_rows - 1) / 2;
  int S = bs_c.n_cols;
  int n_params = bs_c.n_rows; // 2p + 1

  std::vector<arma::mat> X_s_cpp = sstl::mat_views(X_s_list);
  std::vector<arma::vec> Y_s_cpp = sstl::vec_views(Y_s_list);
  std::vector<arma::vec> C_s_cpp = sstl::vec_views(C_s_list);
  std::vector<arma::vec> resid_S(S);
  for (int s = 0; s < S; s++) resid_S[s] = as<arma::vec>(resid_S_list[s]);

  vec N_s_out = zeros(n_params); // slice iterations per row, summed over sources
  vec resid_prop;

  for (int s = 0; s < S; s++) {
    vec& resid = resid_S[s];
    const mat& X = X_s_cpp[s];
    double tau_s = tau_S(s);
    double lambda_s = lambda_S(s);
    double ll_curr = sstl::log_lik_aft(resid, Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
    // a0 is the last row, so the threshold is fixed while w and a are updated
    double threshold = sstl::threshold_from_a0(bs_c(2*p, s), lambda_s);

    for (int j = 0; j < n_params; j++) {
      bool is_a0 = (j == 2*p);
      int k = j % p;

      double f_curr = bs_c(j, s);
      double nu = R::rnorm(0, 1);
      double log_y_threshold = ll_curr + log(R::runif(0, 1));
      double theta = R::runif(0, 2 * M_PI);
      double theta_min = theta - 2 * M_PI;
      double theta_max = theta;
      int n_s = 0;

      double beta_old = 0.0;
      vec w_s, a_s, bias_old, bias_prop, delta;
      uvec nz;
      if (is_a0) {
        w_s = bs_c.col(s).subvec(0, p-1);
        a_s = bs_c.col(s).subvec(p, 2*p-1);
        bias_old = sstl::get_beta(w_s, a_s, tau_s, f_curr, lambda_s, slab_code);
      } else {
        beta_old = sstl::calc_scalar_beta(bs_c(k, s), bs_c(k+p, s), threshold, tau_s, slab_code);
      }

      while (n_s < S_max) {
        n_s++;
        double f_prop = f_curr * cos(theta) + nu * sin(theta);

        bool changed;
        if (is_a0) {
          bias_prop = sstl::get_beta(w_s, a_s, tau_s, f_prop, lambda_s, slab_code);
          delta = bias_prop - bias_old;
          nz = find(delta != 0.0);
          changed = nz.n_elem > 0;
          if (changed) resid_prop = resid - X.cols(nz) * delta.elem(nz);
        } else {
          double beta_new = (j < p)
            ? sstl::calc_scalar_beta(f_prop, bs_c(k+p, s), threshold, tau_s, slab_code)
            : sstl::calc_scalar_beta(bs_c(k, s), f_prop, threshold, tau_s, slab_code);
          double d_val = beta_new - beta_old;
          changed = (d_val != 0.0);
          if (changed) resid_prop = resid - X.col(k) * d_val;
        }

        // An unchanged bias leaves the likelihood unchanged, so ESS accepts without evaluating it.
        double ll_prop = changed ? sstl::log_lik_aft(resid_prop, Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code) : ll_curr;
        if (!changed || ll_prop > log_y_threshold) {
          bs_c(j, s) = f_prop;
          if (changed) {
            resid.swap(resid_prop);
            ll_curr = ll_prop;
          }
          break;
        } else {
          if(theta < 0) theta_min = theta;
          else theta_max = theta;
          theta = R::runif(theta_min, theta_max);
        }
      }
      N_s_out(j) += n_s;
    }
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
                             const arma::vec& Y_T_scale, arma::vec resid_T,
                             const Rcpp::List& Y_s_scale_list, const Rcpp::List& resid_S_list,
                             arma::vec bt_c, const arma::mat& X_T, const arma::vec& Y_T, const arma::vec& C_T,
                             double lambda_T,
                             const Rcpp::List& X_s_list, const Rcpp::List& Y_s_list, const Rcpp::List& C_s_list,
                             double sd_y_T, const arma::vec& sd_y_S,
                             int fam_code, int slab_code) {

  // Scale-adjusted responses construct proposals; original responses enter the likelihood.
  int S = X_s_list.size();

  // 1. Setup Independent Gaussian Prior (Scalar)
  double nu = R::rnorm(0, sd_0);

  double tau_curr = std::abs(xi_t_curr);
  std::vector<arma::vec> resid_S(S);
  std::vector<arma::vec> Y_s_scale_cpp(S);
  // Reference original numeric responses without copying their data.
  std::vector<arma::vec> Y_s_cpp;
  Y_s_cpp.reserve(S);
  for (int s = 0; s < S; ++s) {
    Rcpp::NumericVector Y_s = Y_s_list[s];
    Y_s_cpp.emplace_back(Y_s.begin(), Y_s.size(), false, true);
  }
  std::vector<arma::vec> C_s_cpp(S);

  for(int s=0; s<S; s++) {
    Y_s_scale_cpp[s] = as<arma::vec>(Y_s_scale_list[s]);
    C_s_cpp[s] = as<arma::vec>(C_s_list[s]);
    resid_S[s] = as<arma::vec>(resid_S_list[s]);
  }

  double current_ll = sstl::log_lik_aft(resid_T, Y_T, C_T, sd_y_T, fam_code);
  for(int s=0; s<S; s++) {
    current_ll += sstl::log_lik_aft(resid_S[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
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

    resid_T_prop = Y_T_scale - (Y_T_scale - resid_T) * (tau_prop / tau_curr);
    prop_ll += sstl::log_lik_aft(resid_T_prop, Y_T, C_T, sd_y_T, fam_code);

    for(int s=0; s<S; s++) {
      resid_S_prop[s] = Y_s_scale_cpp[s] - (Y_s_scale_cpp[s] - resid_S[s]) * (tau_prop / tau_curr);
      prop_ll += sstl::log_lik_aft(resid_S_prop[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
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
                              const Rcpp::List& Y_s_scale_list,
                              const Rcpp::List& resid_S_list,
                              const arma::mat& bs_c,
                              const Rcpp::List& X_s_list, const Rcpp::List& Y_s_list, const Rcpp::List& C_s_list,
                              const arma::vec& lambda_S,
                              const arma::vec& sd_y_S,
                              int fam_code, int slab_code) {

  // Scale-adjusted responses construct proposals; original responses enter the likelihood.
  int S = xi_s_curr.n_elem;

  // 1. Setup Independent Spherical Gaussian Prior
  vec nu = sd_0 * randn(S);

  std::vector<vec> resid_S(S);
  std::vector<vec> Y_s_scale_cpp(S);
  // Reference original numeric responses without copying their data.
  std::vector<arma::vec> Y_s_cpp;
  Y_s_cpp.reserve(S);
  for (int s = 0; s < S; ++s) {
    Rcpp::NumericVector Y_s = Y_s_list[s];
    Y_s_cpp.emplace_back(Y_s.begin(), Y_s.size(), false, true);
  }
  std::vector<arma::vec> C_s_cpp(S);

  vec tau_curr = abs(xi_s_curr);
  double current_ll = 0;

  for(int s=0; s<S; s++) {
    Y_s_scale_cpp[s] = as<arma::vec>(Y_s_scale_list[s]);
    C_s_cpp[s] = as<arma::vec>(C_s_list[s]);
    resid_S[s] = as<arma::vec>(resid_S_list[s]);
    current_ll += sstl::log_lik_aft(resid_S[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
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
      resid_S_prop[s] = Y_s_scale_cpp[s] - (Y_s_scale_cpp[s] - resid_S[s]) * (tau_prop(s) / tau_curr(s));
      prop_ll += sstl::log_lik_aft(resid_S_prop[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
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
List update_target_intercept_tl_aft(double b0_T_curr, arma::vec resid_T, const arma::vec& Y_T, const arma::vec& C_T,
                                    double sd_y_T, int fam_code,
                                    double sd_prior = 10.0) {

  double nu = R::rnorm(0, sd_prior);
  double current_ll = sstl::log_lik_aft(resid_T, Y_T, C_T, sd_y_T, fam_code);

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
    double prop_ll = sstl::log_lik_aft(resid_T_prop, Y_T, C_T, sd_y_T, fam_code);

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
                                     const Rcpp::List& resid_S_list, const Rcpp::List& Y_s_list, const Rcpp::List& C_s_list,
                                     const arma::vec& sd_y_S, int fam_code,
                                     double sd_prior = 10.0) {

  int S = b0_s_curr.n_elem;
  std::vector<arma::vec> resid_S(S);
  // Reference original numeric responses without copying their data.
  std::vector<arma::vec> Y_s_cpp;
  Y_s_cpp.reserve(S);
  for (int s = 0; s < S; ++s) {
    Rcpp::NumericVector Y_s = Y_s_list[s];
    Y_s_cpp.emplace_back(Y_s.begin(), Y_s.size(), false, true);
  }
  std::vector<arma::vec> C_s_cpp(S);
  for(int s = 0; s < S; s++) {
    resid_S[s] = as<arma::vec>(resid_S_list[s]);
    C_s_cpp[s] = as<arma::vec>(C_s_list[s]);
  }

  for(int s = 0; s < S; s++) {
    double nu = R::rnorm(0, sd_prior);
    double current_ll = sstl::log_lik_aft(resid_S[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);

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
      double prop_ll = sstl::log_lik_aft(resid_prop, Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);

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
double update_sigma_jeffreys_tl_aft(const arma::vec& resid, const arma::vec& Y, const arma::vec& C, double current_sigma, double step_size, int fam_code) {
  double log_sigma_curr = log(current_sigma);
  double log_sigma_prop = R::rnorm(log_sigma_curr, step_size);
  double sigma_prop     = exp(log_sigma_prop);

  double ll_curr = sstl::log_lik_aft(resid, Y, C, current_sigma, fam_code);
  double ll_prop = sstl::log_lik_aft(resid, Y, C, sigma_prop, fam_code);

  if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
    return sigma_prop;
  } else {
    return current_sigma;
  }
}

// Update Target Sigma
// [[Rcpp::export]]
double update_sigma_target_tl_aft(const arma::vec& resid, const arma::vec& Y, const arma::vec& C,
                                  double current_sigma, int fam_code,
                                  double step_size=0.1) {
  return update_sigma_jeffreys_tl_aft(resid, Y, C, current_sigma, step_size, fam_code);
}

// Update Source Sigma (Per Source)
// [[Rcpp::export]]
double update_sigma_source_tl_aft(const arma::vec& resid, const arma::vec& Y, const arma::vec& C,
                                  double current_sigma, int fam_code,
                                  double step_size=0.1) {
  return update_sigma_jeffreys_tl_aft(resid, Y, C, current_sigma, step_size, fam_code);
}
