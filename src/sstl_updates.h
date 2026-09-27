#pragma once
// Internal ESS-within-Gibbs updates. Each one updates the chain state in place;
// the chains in SSTL_fit.cpp call them once per iteration.
#include "sstl_helpers.h"

namespace sstl {

// ESS_TO_AFT.cpp
void update_blocks_aft(arma::vec& b_c, arma::vec& resid, const arma::mat& X, const arma::vec& Y, const arma::vec& C, const std::vector<arma::uvec>& id, const arma::vec& sd_0, double lambda, double tau, double sd_y, int S_max, int fam_code, int slab_code, arma::vec& N_s);
double update_sigma_to_aft(const arma::vec& resid, const arma::vec& Y, const arma::vec& C, double current_sigma, int fam_code, double step_size);
double update_scale_aft(double xi_curr, const arma::vec& Y_scale, arma::vec& resid, const arma::vec& Y, const arma::vec& C, double sd_y, double sd_prior, int fam_code);
double update_intercept_to_aft(double b0_curr, arma::vec& resid, const arma::vec& Y, const arma::vec& C, double sd_y, int fam_code, double sd_prior);

// ESS_TO_general.cpp
void update_blocks_general(arma::vec& b_c, arma::vec& resid, const arma::mat& X, const arma::vec& Y, const std::vector<arma::uvec>& id, const arma::vec& sd_0, double lambda, double tau, double sd_y, int S_max, int fam_code, int slab_code, double df, arma::vec& N_s);
double update_sigma_to_general(const arma::vec& resid, const arma::vec& Y, double current_sigma, int fam_code, double step_size, double df);
double update_scale_general(double xi_curr, const arma::vec& Y_scale, arma::vec& resid, const arma::vec& Y, double sd_y, double sd_prior, int fam_code, double df);
double update_intercept_to_general(double b0_curr, arma::vec& resid, const arma::vec& Y, double sd_y, int fam_code, double sd_prior, double df);

// ESS_TL_AFT.cpp
void update_target_aft(arma::vec& bt_c, arma::vec& resid_T, std::vector<arma::vec>& resid_S, const arma::mat& X_T, const arma::vec& Y_T, const arma::vec& C_T, const std::vector<arma::mat>& X_s_cpp, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::vec>& C_s_cpp, const std::vector<arma::uvec>& id, const arma::vec& sd_T, double lambda_T, double tau, double sd_y_T, const arma::vec& sd_y_S, int S_max, int fam_code, int slab_code, arma::vec& N_t);
void update_source_joint_aft(arma::mat& bs_c, std::vector<arma::vec>& resid_S, const std::vector<arma::mat>& X_s_cpp, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::vec>& C_s_cpp, const arma::vec& lambda_S, const arma::vec& tau_S, const arma::vec& sd_y_S, int S_max, int fam_code, int slab_code, arma::vec& N_s);
double update_target_scale_aft(double xi_t_curr, const double sd_0, const arma::vec& Y_T_scale, arma::vec& resid_T, const std::vector<arma::vec>& Y_s_scale_cpp, std::vector<arma::vec>& resid_S, const arma::vec& Y_T, const arma::vec& C_T, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::vec>& C_s_cpp, double sd_y_T, const arma::vec& sd_y_S, int fam_code);
arma::vec update_source_scales_aft(const arma::vec& xi_s_curr, const double sd_0, const std::vector<arma::vec>& Y_s_scale_cpp, std::vector<arma::vec>& resid_S, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::vec>& C_s_cpp, const arma::vec& sd_y_S, int fam_code);
double update_target_intercept_tl_aft(double b0_T_curr, arma::vec& resid_T, const arma::vec& Y_T, const arma::vec& C_T, double sd_y_T, int fam_code, double sd_prior);
void update_source_intercepts_tl_aft(arma::vec& b0_s_curr, std::vector<arma::vec>& resid_S, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::vec>& C_s_cpp, const arma::vec& sd_y_S, int fam_code, double sd_prior);
double update_sigma_jeffreys_tl_aft(const arma::vec& resid, const arma::vec& Y, const arma::vec& C, double current_sigma, double step_size, int fam_code);

// ESS_TL_general.cpp
void update_target_general(arma::vec& bt_c, arma::vec& resid_T, std::vector<arma::vec>& resid_S, const arma::mat& X_T, const arma::vec& Y_T, const std::vector<arma::mat>& X_s_cpp, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::uvec>& id, const arma::vec& sd_T, double lambda_T, double tau, double sd_y_T, const arma::vec& sd_y_S, int S_max, int fam_code, int slab_code, double df, arma::vec& N_t);
void update_source_joint_general(arma::mat& bs_c, std::vector<arma::vec>& resid_S, const std::vector<arma::mat>& X_s_cpp, const std::vector<arma::vec>& Y_s_cpp, const arma::vec& lambda_S, const arma::vec& tau_S, const arma::vec& sd_y_S, int S_max, int fam_code, int slab_code, double df, arma::vec& N_s);
double update_target_scale_general(double xi_t_curr, const double sd_0, const arma::vec& Y_T_scale, arma::vec& resid_T, const std::vector<arma::vec>& Y_s_scale_cpp, std::vector<arma::vec>& resid_S, const arma::vec& Y_T, const std::vector<arma::vec>& Y_s_cpp, double sd_y_T, const arma::vec& sd_y_S, int fam_code, double df);
arma::vec update_source_scales_general(const arma::vec& xi_s_curr, const double sd_0, const std::vector<arma::vec>& Y_s_scale_cpp, std::vector<arma::vec>& resid_S, const std::vector<arma::vec>& Y_s_cpp, const arma::vec& sd_y_S, int fam_code, double df);
double update_target_intercept_tl_general(double b0_T_curr, arma::vec& resid_T, const arma::vec& Y_T, double sd_y_T, int fam_code, double sd_prior, double df);
void update_source_intercepts_tl_general(arma::vec& b0_s_curr, std::vector<arma::vec>& resid_S, const std::vector<arma::vec>& Y_s_cpp, const arma::vec& sd_y_S, int fam_code, double sd_prior, double df);
double update_sigma_mh_tl_general(const arma::vec& resid, const arma::vec& Y, double current_sigma, double step_size, int fam_code, double df);

// ESS_TL_group_AFT.cpp
void update_target_group_aft(arma::vec& bt_c, arma::vec& resid_T, std::vector<arma::vec>& resid_S, const arma::mat& X_T, const arma::vec& Y_T, const arma::vec& C_T, const std::vector<arma::mat>& X_s_cpp, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::vec>& C_s_cpp, const std::vector<arma::uvec>& group_cols, const Rcpp::IntegerVector& group_map, const arma::vec& sd_T, double lambda_T, double tau, double sd_y_T, const arma::vec& sd_y_S, int S_max, int fam_code, int slab_code, arma::vec& N_t);
void update_source_joint_group_aft(arma::mat& bs_c, std::vector<arma::vec>& resid_S, const std::vector<arma::mat>& X_s_cpp, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::vec>& C_s_cpp, const std::vector<arma::uvec>& group_cols, const Rcpp::IntegerVector& group_map, const arma::vec& lambda_S, const arma::vec& tau_S, const arma::vec& sd_y_S, int S_max, int fam_code, int slab_code, arma::vec& N_s);

// ESS_TL_group_general.cpp
void update_target_group_general(arma::vec& bt_c, arma::vec& resid_T, std::vector<arma::vec>& resid_S, const arma::mat& X_T, const arma::vec& Y_T, const std::vector<arma::mat>& X_s_cpp, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::uvec>& group_cols, const Rcpp::IntegerVector& group_map, const arma::vec& sd_T, double lambda_T, double tau, double sd_y_T, const arma::vec& sd_y_S, int S_max, int fam_code, int slab_code, double df, arma::vec& N_t);
void update_source_joint_group_general(arma::mat& bs_c, std::vector<arma::vec>& resid_S, const std::vector<arma::mat>& X_s_cpp, const std::vector<arma::vec>& Y_s_cpp, const std::vector<arma::uvec>& group_cols, const Rcpp::IntegerVector& group_map, const arma::vec& lambda_S, const arma::vec& tau_S, const arma::vec& sd_y_S, int S_max, int fam_code, int slab_code, double df, arma::vec& N_s);

} // namespace sstl
