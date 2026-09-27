#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "sstl_helpers.h"

using namespace Rcpp;
using namespace arma;

namespace {

std::vector<arma::vec> as_vec_vec(const Rcpp::List& x_list) {
  int S = x_list.size();
  std::vector<arma::vec> out(S);
  for (int s = 0; s < S; ++s) out[s] = as<arma::vec>(x_list[s]);
  return out;
}

std::vector<arma::uvec> as_uvec_groups(const Rcpp::List& id) {
  int G = id.size();
  std::vector<arma::uvec> groups(G);
  for (int g = 0; g < G; ++g) {
    groups[g] = as<arma::uvec>(id[g]) - 1;
  }
  return groups;
}

} // namespace

// [[Rcpp::export]]
List update_target_group_aft(arma::vec bt_c,
                             arma::vec resid_T,
                             const Rcpp::List& resid_S_list,
                             const arma::mat& X_T,
                             const arma::vec& Y_T, const arma::vec& C_T,
                             const Rcpp::List& X_S_list,
                             const Rcpp::List& Y_s_list, const Rcpp::List& C_S_list,
                             const Rcpp::List& id,
                             const Rcpp::IntegerVector& group_map,
                             const arma::vec& sd_T,
                             double lambda_T,
                             double tau,
                             double sd_y_T,
                             const arma::vec& sd_y_S,
                             int S_max,
                             int fam_code,
                             int slab_code) {
  int p = X_T.n_cols;
  int S = X_S_list.size();
  int G = id.size();

  std::vector<arma::mat> X_s_cpp = sstl::mat_views(X_S_list);
  // Reference original numeric responses without copying their data.
  std::vector<arma::vec> Y_s_cpp;
  Y_s_cpp.reserve(S);
  for (int s = 0; s < S; ++s) {
    Rcpp::NumericVector Y_s = Y_s_list[s];
    Y_s_cpp.emplace_back(Y_s.begin(), Y_s.size(), false, true);
  }
  std::vector<arma::vec> C_s_cpp = as_vec_vec(C_S_list);
  std::vector<arma::vec> resid_S = as_vec_vec(resid_S_list);
  std::vector<arma::uvec> group_cols = as_uvec_groups(id);

  vec w_T = bt_c.subvec(0, p - 1);
  vec a_T = bt_c.subvec(p, p + G - 1);
  double a0_T = bt_c(p + G);
  double threshold_T = sstl::threshold_from_a0(a0_T, lambda_T);
  vec beta_T = sstl::get_beta_group(w_T, a_T, group_map, tau, a0_T, lambda_T,
                                   slab_code);

  double current_ll_global = sstl::log_lik_aft(resid_T, Y_T, C_T, sd_y_T, fam_code);
  for (int s = 0; s < S; ++s) {
    current_ll_global += sstl::log_lik_aft(resid_S[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
  }

  vec N_s_out = zeros(p + G + 1);

  for (int g = 0; g < G; ++g) {
    const uvec& affected_cols = group_cols[g];
    int idx_a = p + g;

    // Group gate update first.
    double f_curr = bt_c(idx_a);
    double nu = R::rnorm(0.0, sd_T(idx_a));
    double log_y_threshold = current_ll_global + std::log(R::runif(0.0, 1.0));
    double theta = R::runif(0.0, 2.0 * M_PI);
    double theta_min = theta - 2.0 * M_PI;
    double theta_max = theta;
    int n_s = 0;

    double act_old = sstl::activation_scalar(f_curr, threshold_T, slab_code);
    vec h_group = sstl::slab_weight_vec(w_T.elem(affected_cols), slab_code);

    while (n_s < S_max) {
      ++n_s;
      double f_prop = f_curr * std::cos(theta) + nu * std::sin(theta);
      double act_new = sstl::activation_scalar(f_prop, threshold_T, slab_code);
      if (act_new == act_old) { // group stays inactive (or unchanged): ESS accepts
        bt_c(idx_a) = f_prop;
        a_T(g) = f_prop;
        break;
      }
      vec delta_beta = tau * h_group * (act_new - act_old);

      vec resid_T_prop = resid_T - X_T.cols(affected_cols) * delta_beta;
      double ll_T_prop = sstl::log_lik_aft(resid_T_prop, Y_T, C_T, sd_y_T, fam_code);

      double ll_S_prop_total = 0.0;
      std::vector<vec> resid_S_prop(S);
      for (int s = 0; s < S; ++s) {
        resid_S_prop[s] = resid_S[s] - X_s_cpp[s].cols(affected_cols) * delta_beta;
        ll_S_prop_total += sstl::log_lik_aft(resid_S_prop[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
      }

      double prop_ll_global = ll_T_prop + ll_S_prop_total;
      if (prop_ll_global > log_y_threshold) {
        bt_c(idx_a) = f_prop;
        a_T(g) = f_prop;
        beta_T.elem(affected_cols) += delta_beta;
        resid_T = resid_T_prop;
        resid_S = resid_S_prop;
        current_ll_global = prop_ll_global;
        break;
      }

      if (theta < 0.0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
    N_s_out(idx_a) = n_s;

    // Then update the group-specific weights one by one.
    for (uword pos = 0; pos < affected_cols.n_elem; ++pos) {
      uword j = affected_cols(pos);

      double w_curr = bt_c(j);
      double nu_w = R::rnorm(0.0, sd_T(j));
      double log_y_threshold_w = current_ll_global + std::log(R::runif(0.0, 1.0));
      double theta_w = R::runif(0.0, 2.0 * M_PI);
      double theta_min_w = theta_w - 2.0 * M_PI;
      double theta_max_w = theta_w;
      int n_w = 0;

      while (n_w < S_max) {
        ++n_w;
        double w_prop = w_curr * std::cos(theta_w) + nu_w * std::sin(theta_w);

        double beta_old = sstl::calc_scalar_beta(w_curr, a_T(g), threshold_T, tau,
                                                slab_code);
        double beta_new = sstl::calc_scalar_beta(w_prop, a_T(g), threshold_T, tau,
                                                slab_code);
        double delta_beta = beta_new - beta_old;
        if (delta_beta == 0.0) { // unchanged likelihood: ESS accepts
          bt_c(j) = w_prop;
          w_T(j) = w_prop;
          beta_T(j) = beta_new;
          break;
        }

        vec resid_T_prop = resid_T - X_T.col(j) * delta_beta;
        double ll_T_prop = sstl::log_lik_aft(resid_T_prop, Y_T, C_T, sd_y_T, fam_code);

        double ll_S_prop_total = 0.0;
        std::vector<vec> resid_S_prop(S);
        for (int s = 0; s < S; ++s) {
          resid_S_prop[s] = resid_S[s] - X_s_cpp[s].col(j) * delta_beta;
          ll_S_prop_total += sstl::log_lik_aft(resid_S_prop[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
        }

        double prop_ll_global = ll_T_prop + ll_S_prop_total;
        if (prop_ll_global > log_y_threshold_w) {
          bt_c(j) = w_prop;
          w_T(j) = w_prop;
          beta_T(j) = beta_new;
          resid_T = resid_T_prop;
          resid_S = resid_S_prop;
          current_ll_global = prop_ll_global;
          break;
        }

        if (theta_w < 0.0) theta_min_w = theta_w;
        else theta_max_w = theta_w;
        theta_w = R::runif(theta_min_w, theta_max_w);
      }
      N_s_out(j) = n_w;
    }
  }

  // Global threshold update last.
  int idx_a0 = p + G;
  double f_curr = bt_c(idx_a0);
  double nu = R::rnorm(0.0, sd_T(idx_a0));
  double log_y_threshold = current_ll_global + std::log(R::runif(0.0, 1.0));
  double theta = R::runif(0.0, 2.0 * M_PI);
  double theta_min = theta - 2.0 * M_PI;
  double theta_max = theta;
  int n_s = 0;

  while (n_s < S_max) {
    ++n_s;
    double f_prop = f_curr * std::cos(theta) + nu * std::sin(theta);
    vec beta_T_prop = sstl::get_beta_group(w_T, a_T, group_map, tau, f_prop, lambda_T,
                                          slab_code);
    vec delta_beta = beta_T_prop - beta_T;
    uvec nz = find(delta_beta != 0.0);
    if (nz.n_elem == 0) { // unchanged likelihood: ESS accepts
      bt_c(idx_a0) = f_prop;
      a0_T = f_prop;
      threshold_T = sstl::threshold_from_a0(a0_T, lambda_T);
      beta_T = beta_T_prop;
      break;
    }

    vec resid_T_prop = resid_T - X_T.cols(nz) * delta_beta.elem(nz);
    double ll_T_prop = sstl::log_lik_aft(resid_T_prop, Y_T, C_T, sd_y_T, fam_code);

    double ll_S_prop_total = 0.0;
    std::vector<vec> resid_S_prop(S);
    for (int s = 0; s < S; ++s) {
      resid_S_prop[s] = resid_S[s] - X_s_cpp[s].cols(nz) * delta_beta.elem(nz);
      ll_S_prop_total += sstl::log_lik_aft(resid_S_prop[s], Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
    }

    double prop_ll_global = ll_T_prop + ll_S_prop_total;
    if (prop_ll_global > log_y_threshold) {
      bt_c(idx_a0) = f_prop;
      a0_T = f_prop;
      threshold_T = sstl::threshold_from_a0(a0_T, lambda_T);
      beta_T = beta_T_prop;
      resid_T = resid_T_prop;
      resid_S = resid_S_prop;
      current_ll_global = prop_ll_global;
      break;
    }

    if (theta < 0.0) theta_min = theta;
    else theta_max = theta;
    theta = R::runif(theta_min, theta_max);
  }
  N_s_out(idx_a0) = n_s;

  Rcpp::List resid_S_out(S);
  for (int s = 0; s < S; ++s) resid_S_out[s] = resid_S[s];

  return List::create(Named("bt_c") = bt_c,
                      Named("resid_T") = resid_T,
                      Named("resid_S") = resid_S_out,
                      Named("N_t") = N_s_out);
}

// [[Rcpp::export]]
List update_source_joint_group_aft(arma::mat bs_c,
                                   const Rcpp::List& resid_S_list,
                                   const Rcpp::List& X_s_list,
                                   const Rcpp::List& Y_s_list, const Rcpp::List& C_s_list,
                                   const Rcpp::List& id,
                                   const Rcpp::IntegerVector& group_map,
                                   const arma::vec& lambda_S,
                                   const arma::vec& tau_S,
                                   const arma::vec& sd_y_S,
                                   int S_max,
                                   int fam_code,
                                   int slab_code) {
  int p = group_map.size();
  int S = bs_c.n_cols;
  int G = id.size();
  int idx_a0 = p + G;

  std::vector<arma::mat> X_s_cpp = sstl::mat_views(X_s_list);
  std::vector<arma::vec> Y_s_cpp = sstl::vec_views(Y_s_list);
  std::vector<arma::vec> C_s_cpp = sstl::vec_views(C_s_list);
  std::vector<arma::vec> resid_S = as_vec_vec(resid_S_list);
  std::vector<arma::uvec> group_cols = as_uvec_groups(id);

  vec N_s_out = zeros(p + G + 1); // slice iterations per row, summed over sources
  vec resid_prop;

  // Sources are conditionally independent given the target: update each source
  // in turn with one-dimensional ESS steps on its own likelihood.
  for (int s = 0; s < S; ++s) {
    vec& resid = resid_S[s];
    const mat& X = X_s_cpp[s];
    double tau_s = tau_S(s);
    double ll_curr = sstl::log_lik_aft(resid, Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
    // a0 is updated last, so the threshold is fixed for the gates and weights
    double thresh_s = sstl::threshold_from_a0(bs_c(idx_a0, s), lambda_S(s));

    for (int g = 0; g < G; ++g) {
      const uvec& affected_cols = group_cols[g];
      int idx_a = p + g;

      // Group gate first.
      double f_curr = bs_c(idx_a, s);
      double nu = R::rnorm(0.0, 1.0);
      double log_y_threshold = ll_curr + std::log(R::runif(0.0, 1.0));
      double theta = R::runif(0.0, 2.0 * M_PI);
      double theta_min = theta - 2.0 * M_PI;
      double theta_max = theta;
      int n_s = 0;

      double act_old = sstl::activation_scalar(f_curr, thresh_s, slab_code);
      vec w_group(affected_cols.n_elem);
      for (uword pos = 0; pos < affected_cols.n_elem; ++pos) w_group(pos) = bs_c(affected_cols(pos), s);
      vec h_group = sstl::slab_weight_vec(w_group, slab_code);

      while (n_s < S_max) {
        ++n_s;
        double f_prop = f_curr * std::cos(theta) + nu * std::sin(theta);
        double act_new = sstl::activation_scalar(f_prop, thresh_s, slab_code);
        if (act_new == act_old) { // unchanged likelihood: ESS accepts
          bs_c(idx_a, s) = f_prop;
          break;
        }
        vec delta_beta = tau_s * h_group * (act_new - act_old);
        resid_prop = resid - X.cols(affected_cols) * delta_beta;
        double ll_prop = sstl::log_lik_aft(resid_prop, Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
        if (ll_prop > log_y_threshold) {
          bs_c(idx_a, s) = f_prop;
          resid.swap(resid_prop);
          ll_curr = ll_prop;
          break;
        }
        if (theta < 0.0) theta_min = theta;
        else theta_max = theta;
        theta = R::runif(theta_min, theta_max);
      }
      N_s_out(idx_a) += n_s;

      // Then the group's feature-level weights.
      double a_val = bs_c(idx_a, s);
      for (uword pos = 0; pos < affected_cols.n_elem; ++pos) {
        uword j = affected_cols(pos);

        double w_curr = bs_c(j, s);
        double nu_w = R::rnorm(0.0, 1.0);
        double log_y_threshold_w = ll_curr + std::log(R::runif(0.0, 1.0));
        double theta_w = R::runif(0.0, 2.0 * M_PI);
        double theta_min_w = theta_w - 2.0 * M_PI;
        double theta_max_w = theta_w;
        int n_w = 0;
        double beta_old = sstl::calc_scalar_beta(w_curr, a_val, thresh_s, tau_s, slab_code);

        while (n_w < S_max) {
          ++n_w;
          double w_prop = w_curr * std::cos(theta_w) + nu_w * std::sin(theta_w);
          double beta_new = sstl::calc_scalar_beta(w_prop, a_val, thresh_s, tau_s, slab_code);
          double delta_beta = beta_new - beta_old;
          if (delta_beta == 0.0) { // inactive group: ESS accepts
            bs_c(j, s) = w_prop;
            break;
          }
          resid_prop = resid - X.col(j) * delta_beta;
          double ll_prop = sstl::log_lik_aft(resid_prop, Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
          if (ll_prop > log_y_threshold_w) {
            bs_c(j, s) = w_prop;
            resid.swap(resid_prop);
            ll_curr = ll_prop;
            break;
          }
          if (theta_w < 0.0) theta_min_w = theta_w;
          else theta_max_w = theta_w;
          theta_w = R::runif(theta_min_w, theta_max_w);
        }
        N_s_out(j) += n_w;
      }
    }

    // Global threshold for this source last.
    double f_curr = bs_c(idx_a0, s);
    double nu = R::rnorm(0.0, 1.0);
    double log_y_threshold = ll_curr + std::log(R::runif(0.0, 1.0));
    double theta = R::runif(0.0, 2.0 * M_PI);
    double theta_min = theta - 2.0 * M_PI;
    double theta_max = theta;
    int n_s = 0;

    vec bs_col = bs_c.col(s);
    vec bias_old = sstl::calc_bias_vec_group(bs_col, tau_s, group_map, lambda_S(s), slab_code);

    while (n_s < S_max) {
      ++n_s;
      double f_prop = f_curr * std::cos(theta) + nu * std::sin(theta);
      bs_col(idx_a0) = f_prop;
      vec delta = sstl::calc_bias_vec_group(bs_col, tau_s, group_map, lambda_S(s), slab_code) - bias_old;
      uvec nz = find(delta != 0.0);
      if (nz.n_elem == 0) { // unchanged likelihood: ESS accepts
        bs_c(idx_a0, s) = f_prop;
        break;
      }
      resid_prop = resid - X.cols(nz) * delta.elem(nz);
      double ll_prop = sstl::log_lik_aft(resid_prop, Y_s_cpp[s], C_s_cpp[s], sd_y_S(s), fam_code);
      if (ll_prop > log_y_threshold) {
        bs_c(idx_a0, s) = f_prop;
        resid.swap(resid_prop);
        ll_curr = ll_prop;
        break;
      }
      if (theta < 0.0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
    N_s_out(idx_a0) += n_s;
  }

  Rcpp::List resid_S_out(S);
  for (int s = 0; s < S; ++s) resid_S_out[s] = resid_S[s];

  return List::create(Named("bs_c") = bs_c,
                      Named("resid_S") = resid_S_out,
                      Named("N_s") = N_s_out);
}
