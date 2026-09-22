#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "sstl_helpers.h"

using namespace Rcpp;
using namespace arma;

namespace {

std::vector<arma::mat> as_mat_vec(const Rcpp::List& x_list) {
  int S = x_list.size();
  std::vector<arma::mat> out(S);
  for (int s = 0; s < S; ++s) out[s] = as<arma::mat>(x_list[s]);
  return out;
}

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
List update_target_group_general(arma::vec bt_c,
                             arma::vec resid_T,
                             const Rcpp::List& resid_S_list,
                             const arma::mat& X_T,
                             const arma::vec& Y_T,
                             const Rcpp::List& X_S_list,
                             const Rcpp::List& Y_s_list,
                             const Rcpp::List& id,
                             const Rcpp::IntegerVector& group_map,
                             const arma::vec& sd_T,
                             double lambda_T,
                             double tau,
                             double sd_y_T,
                             const arma::vec& sd_y_S,
                             int S_max,
                             int fam_code,
                             int slab_code, double df = 4.0) {
  int p = X_T.n_cols;
  int S = X_S_list.size();
  int G = id.size();

  std::vector<arma::mat> X_s_cpp = as_mat_vec(X_S_list);
  // Reference original numeric responses without copying their data.
  std::vector<arma::vec> Y_s_cpp;
  Y_s_cpp.reserve(S);
  for (int s = 0; s < S; ++s) {
    Rcpp::NumericVector Y_s = Y_s_list[s];
    Y_s_cpp.emplace_back(Y_s.begin(), Y_s.size(), false, true);
  }
  std::vector<arma::vec> resid_S = as_vec_vec(resid_S_list);
  std::vector<arma::uvec> group_cols = as_uvec_groups(id);

  vec w_T = bt_c.subvec(0, p - 1);
  vec a_T = bt_c.subvec(p, p + G - 1);
  double a0_T = bt_c(p + G);
  double threshold_T = sstl::threshold_from_a0(a0_T, lambda_T);
  vec beta_T = sstl::get_beta_group(w_T, a_T, group_map, tau, a0_T, lambda_T,
                                   slab_code);

  double current_ll_global = sstl::log_lik_general(resid_T, Y_T, sd_y_T, fam_code, df);
  for (int s = 0; s < S; ++s) {
    current_ll_global += sstl::log_lik_general(resid_S[s], Y_s_cpp[s], sd_y_S(s), fam_code, df);
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
      vec delta_beta = tau * h_group * (act_new - act_old);

      vec resid_T_prop = resid_T - X_T.cols(affected_cols) * delta_beta;
      double ll_T_prop = sstl::log_lik_general(resid_T_prop, Y_T, sd_y_T, fam_code, df);

      double ll_S_prop_total = 0.0;
      std::vector<vec> resid_S_prop(S);
      for (int s = 0; s < S; ++s) {
        resid_S_prop[s] = resid_S[s] - X_s_cpp[s].cols(affected_cols) * delta_beta;
        ll_S_prop_total += sstl::log_lik_general(resid_S_prop[s], Y_s_cpp[s], sd_y_S(s), fam_code, df);
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

        vec resid_T_prop = resid_T - X_T.col(j) * delta_beta;
        double ll_T_prop = sstl::log_lik_general(resid_T_prop, Y_T, sd_y_T, fam_code, df);

        double ll_S_prop_total = 0.0;
        std::vector<vec> resid_S_prop(S);
        for (int s = 0; s < S; ++s) {
          resid_S_prop[s] = resid_S[s] - X_s_cpp[s].col(j) * delta_beta;
          ll_S_prop_total += sstl::log_lik_general(resid_S_prop[s], Y_s_cpp[s], sd_y_S(s), fam_code, df);
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

    vec resid_T_prop = resid_T - X_T * delta_beta;
    double ll_T_prop = sstl::log_lik_general(resid_T_prop, Y_T, sd_y_T, fam_code, df);

    double ll_S_prop_total = 0.0;
    std::vector<vec> resid_S_prop(S);
    for (int s = 0; s < S; ++s) {
      resid_S_prop[s] = resid_S[s] - X_s_cpp[s] * delta_beta;
      ll_S_prop_total += sstl::log_lik_general(resid_S_prop[s], Y_s_cpp[s], sd_y_S(s), fam_code, df);
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
List update_source_joint_group_general(arma::mat bs_c,
                                   const Rcpp::List& resid_S_list,
                                   const Rcpp::List& X_s_list,
                                   const Rcpp::List& Y_s_list,
                                   const Rcpp::List& id,
                                   const Rcpp::IntegerVector& group_map,
                                   const arma::vec& lambda_S,
                                   const arma::vec& tau_S,
                                   const arma::vec& sd_y_S,
                                   int S_max,
                                   int fam_code,
                                   int slab_code, double df = 4.0) {
  int p = group_map.size();
  int S = bs_c.n_cols;
  int G = id.size();
  int idx_a0 = p + G;

  std::vector<arma::mat> X_s_cpp = as_mat_vec(X_s_list);
  // Reference original numeric responses without copying their data.
  std::vector<arma::vec> Y_s_cpp;
  Y_s_cpp.reserve(S);
  for (int s = 0; s < S; ++s) {
    Rcpp::NumericVector Y_s = Y_s_list[s];
    Y_s_cpp.emplace_back(Y_s.begin(), Y_s.size(), false, true);
  }
  std::vector<arma::vec> resid_S = as_vec_vec(resid_S_list);
  std::vector<arma::uvec> group_cols = as_uvec_groups(id);

  double current_ll_total = 0.0;
  for (int s = 0; s < S; ++s) {
    current_ll_total += sstl::log_lik_general(resid_S[s], Y_s_cpp[s], sd_y_S(s), fam_code, df);
  }

  vec N_s_out = zeros(p + G + 1);

  for (int g = 0; g < G; ++g) {
    const uvec& affected_cols = group_cols[g];
    int idx_a = p + g;

    // Shared group gate across sources, updated row-wise.
    rowvec f_curr = bs_c.row(idx_a);
    vec nu = randn(S);
    double log_y_threshold = current_ll_total + std::log(R::runif(0.0, 1.0));
    double theta = R::runif(0.0, 2.0 * M_PI);
    double theta_min = theta - 2.0 * M_PI;
    double theta_max = theta;
    int n_s = 0;

    while (n_s < S_max) {
      ++n_s;
      rowvec f_prop_row = f_curr * std::cos(theta) + trans(nu) * std::sin(theta);

      double prop_ll_total = 0.0;
      std::vector<vec> resid_S_prop(S);
      for (int s = 0; s < S; ++s) {
        double thresh_s = sstl::threshold_from_a0(bs_c(idx_a0, s), lambda_S(s));
        double act_old = sstl::activation_scalar(f_curr(s), thresh_s, slab_code);
        double act_new = sstl::activation_scalar(f_prop_row(s), thresh_s, slab_code);
        vec w_vec = bs_c.col(s).subvec(0, p - 1);
        vec h_group = sstl::slab_weight_vec(w_vec.elem(affected_cols), slab_code);
        vec delta_beta = tau_S(s) * h_group * (act_new - act_old);

        resid_S_prop[s] = resid_S[s] - X_s_cpp[s].cols(affected_cols) * delta_beta;
        prop_ll_total += sstl::log_lik_general(resid_S_prop[s], Y_s_cpp[s], sd_y_S(s), fam_code, df);
      }

      if (prop_ll_total > log_y_threshold) {
        bs_c.row(idx_a) = f_prop_row;
        resid_S = resid_S_prop;
        current_ll_total = prop_ll_total;
        break;
      }

      if (theta < 0.0) theta_min = theta;
      else theta_max = theta;
      theta = R::runif(theta_min, theta_max);
    }
    N_s_out(idx_a) = n_s;

    // Then feature-level weights for that group, still updated row-wise across sources.
    for (uword pos = 0; pos < affected_cols.n_elem; ++pos) {
      uword j = affected_cols(pos);

      rowvec w_curr = bs_c.row(j);
      vec nu_w = randn(S);
      double log_y_threshold_w = current_ll_total + std::log(R::runif(0.0, 1.0));
      double theta_w = R::runif(0.0, 2.0 * M_PI);
      double theta_min_w = theta_w - 2.0 * M_PI;
      double theta_max_w = theta_w;
      int n_w = 0;

      while (n_w < S_max) {
        ++n_w;
        rowvec w_prop_row = w_curr * std::cos(theta_w) + trans(nu_w) * std::sin(theta_w);

        double prop_ll_total = 0.0;
        std::vector<vec> resid_S_prop(S);
        for (int s = 0; s < S; ++s) {
          double thresh_s = sstl::threshold_from_a0(bs_c(idx_a0, s), lambda_S(s));
          double a_val = bs_c(idx_a, s);
          double beta_old = sstl::calc_scalar_beta(w_curr(s), a_val, thresh_s, tau_S(s),
                                                  slab_code);
          double beta_new = sstl::calc_scalar_beta(w_prop_row(s), a_val, thresh_s, tau_S(s),
                                                  slab_code);
          double delta_beta = beta_new - beta_old;

          resid_S_prop[s] = resid_S[s] - X_s_cpp[s].col(j) * delta_beta;
          prop_ll_total += sstl::log_lik_general(resid_S_prop[s], Y_s_cpp[s], sd_y_S(s), fam_code, df);
        }

        if (prop_ll_total > log_y_threshold_w) {
          bs_c.row(j) = w_prop_row;
          resid_S = resid_S_prop;
          current_ll_total = prop_ll_total;
          break;
        }

        if (theta_w < 0.0) theta_min_w = theta_w;
        else theta_max_w = theta_w;
        theta_w = R::runif(theta_min_w, theta_max_w);
      }
      N_s_out(j) = n_w;
    }
  }

  // Global threshold row across sources.
  rowvec f_curr = bs_c.row(idx_a0);
  vec nu = randn(S);
  double log_y_threshold = current_ll_total + std::log(R::runif(0.0, 1.0));
  double theta = R::runif(0.0, 2.0 * M_PI);
  double theta_min = theta - 2.0 * M_PI;
  double theta_max = theta;
  int n_s = 0;

  while (n_s < S_max) {
    ++n_s;
    rowvec f_prop_row = f_curr * std::cos(theta) + trans(nu) * std::sin(theta);

    double prop_ll_total = 0.0;
    std::vector<vec> resid_S_prop(S);
    for (int s = 0; s < S; ++s) {
      vec bs_col = bs_c.col(s);
      vec bias_old = sstl::calc_bias_vec_group(bs_col, tau_S(s), group_map, lambda_S(s),
                                              slab_code);
      bs_col(idx_a0) = f_prop_row(s);
      vec bias_new = sstl::calc_bias_vec_group(bs_col, tau_S(s), group_map, lambda_S(s),
                                              slab_code);
      resid_S_prop[s] = resid_S[s] - X_s_cpp[s] * (bias_new - bias_old);
      prop_ll_total += sstl::log_lik_general(resid_S_prop[s], Y_s_cpp[s], sd_y_S(s), fam_code, df);
    }

    if (prop_ll_total > log_y_threshold) {
      bs_c.row(idx_a0) = f_prop_row;
      resid_S = resid_S_prop;
      current_ll_total = prop_ll_total;
      break;
    }

    if (theta < 0.0) theta_min = theta;
    else theta_max = theta;
    theta = R::runif(theta_min, theta_max);
  }
  N_s_out(idx_a0) = n_s;

  Rcpp::List resid_S_out(S);
  for (int s = 0; s < S; ++s) resid_S_out[s] = resid_S[s];

  return List::create(Named("bs_c") = bs_c,
                      Named("resid_S") = resid_S_out,
                      Named("N_s") = N_s_out);
}
