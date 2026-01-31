  #include <RcppArmadillo.h>
  // [[Rcpp::depends(RcppArmadillo)]]
  // [[Rcpp::plugins(cpp17)]]

  #include "ntl_helpers.h"

  using namespace Rcpp;
  using namespace arma;

  // --- Helper Functions (Same as before) ---

  // --- Survival AFT Log-Likelihood ---
  // z = (Y - X*beta) / sigma
  // LL = sum_{obs} (z - log(sigma)) - sum_{all} exp(z)
  double log_lik_aft(const vec& resid, const vec& C, double sd_y, int fam_code) {
    vec z = resid / sd_y;
    double ll = 0.0;

    if (fam_code == 1) { // --- Weibull ---
      double term1 = dot(C, z) - accu(C) * log(sd_y);
      double term2 = sum(exp(z));
      ll = term1 - term2;

    } else if (fam_code == 2) { // --- Log-Logistic ---
      vec log_denom = log(1.0 + exp(z));
      ll = dot(C, z - log(sd_y)) - dot(1.0 + C, log_denom);

    } else if (fam_code == 3) { // --- Log-Normal (Gaussian Errors) ---
      int n = resid.n_elem;
      for(int i=0; i<n; i++) {
        if(C(i) == 1.0) {
          ll += R::dnorm(z(i), 0.0, 1.0, 1) - log(sd_y);
        } else {
          ll += R::pnorm(-z(i), 0.0, 1.0, 1, 1);
        }
      }
    }

    return ll;
  }


  // ============================================================================
  // FUNCTION 1: Update Target Parameters (beta_T)
  // Impact: Changes Y_T likelihood AND ALL Y_S likelihoods
  // ============================================================================

  // [[Rcpp::export]]
  Rcpp::List update_target_aft(arma::vec bt_c, const arma::mat& X_T, const arma::vec& Y_T, const arma::vec& C_T,
                             const Rcpp::List& X_S_list, const Rcpp::List& Y_S_list, const Rcpp::List& C_S_list,
                             const arma::mat& bs_c, const Rcpp::List& id, const arma::vec& sd_T,
                             double lambda_T, const arma::vec& lambda_S,
                             double sd_y_T, const arma::vec& sd_y_S, int S_max, int fam_code) {

    int p = X_T.n_cols;
    int S = X_S_list.size();
    int K = id.size();

    // Current Beta_T
    vec w_T = bt_c.subvec(0, p-1);
    vec a_T = bt_c.subvec(p, 2*p-1);
    double a0_T = bt_c(2*p);
    vec beta_T = ntl::get_beta(w_T, a_T, a0_T, lambda_T);

    // Initialize Target Residuals
    vec resid_T = Y_T - X_T * beta_T;
    double ll_T = log_lik_aft(resid_T, C_T, sd_y_T, fam_code);

    // Initialize Source Residuals (List of vectors)
    std::vector<vec> resid_S(S);
    double ll_S_total = 0;

    for(int s=0; s<S; s++) {
      mat X_s = X_S_list[s];
      vec Y_s = Y_S_list[s];
      vec C_s = C_S_list[s];

      // Construct Bias for Source s
      vec bs_col = bs_c.col(s);
      vec bias_s = ntl::calc_bias_vec(bs_col, p, lambda_S(s));

      // Residual = Y - X * (beta_T + bias)
      resid_S[s] = Y_s - X_s * (beta_T + bias_s);
      ll_S_total += log_lik_aft(resid_S[s], C_s, sd_y_S(s), fam_code);
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
      while(n_s < S_max) {
        n_s++;
        vec f_prop = f_curr * cos(theta) + nu * sin(theta);

        // Construct tentative parameters
        vec bt_prop = bt_c;
        bt_prop.elem(idx) = f_prop;

        vec w_prop = bt_prop.subvec(0, p-1);
        vec a_prop = bt_prop.subvec(p, 2*p-1);
        double a0_prop = bt_prop(2*p);
        vec beta_T_prop = ntl::get_beta(w_prop, a_prop, a0_prop, lambda_T);

        // --- C. GLOBAL RESIDUAL UPDATE ---
        vec delta_beta = beta_T_prop - beta_T;

        // 1. Update Target Residual
        vec resid_T_prop = resid_T;
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
        double ll_T_prop = log_lik_aft(resid_T_prop, C_T, sd_y_T, fam_code);

        // 2. Update Source Residuals
        double ll_S_prop_total = 0;
        std::vector<vec> resid_S_prop(S);

        for(int s=0; s<S; s++) {
          mat X_s = X_S_list[s];
          vec C_s = C_S_list[s];
          vec r_s_curr = resid_S[s]; // copy current resid

          // Apply same delta_beta to source
          if (k == K - 1) {
            r_s_curr -= X_s * delta_beta;
          } else {
            int half = idx.n_elem / 2;
            uword start = idx(0);
            uword end = idx(half-1);
            vec d_sub = delta_beta.subvec(start, end);
            r_s_curr -= X_s.cols(start, end) * d_sub;
          }

          resid_S_prop[s] = r_s_curr;
          ll_S_prop_total += log_lik_aft(r_s_curr, C_s, sd_y_S(s), fam_code);
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
  Rcpp::List update_source_joint_aft(arma::mat bs_c, const Rcpp::List& X_s_list, const Rcpp::List& Y_s_list, const Rcpp::List& C_s_list,
                                     const arma::vec& beta_T, const arma::mat& chol_cov_W,
                                     const arma::vec& lambda_S, const arma::vec& sd_y_S, int S_max, int fam_code) {

    int p = (bs_c.n_rows - 1) / 2;
    int S = bs_c.n_cols;
    int n_params = bs_c.n_rows; // 2p + 1

    // Pre-Calculate Current Residuals for ALL Sources
    std::vector<vec> resid_S(S);
    std::vector<double> ll_S(S);
    double current_ll_total = 0;

    for(int s=0; s<S; s++) {
      mat X = X_s_list[s];
      vec Y = Y_s_list[s];
      vec C = C_s_list[s];
      vec bias = ntl::calc_bias_vec(bs_c.col(s), p, lambda_S(s));

      // Residual = Y - X(beta_T + bias)
      resid_S[s] = Y - X * (beta_T + bias);
      ll_S[s] = log_lik_aft(resid_S[s], C, sd_y_S(s), fam_code);
      current_ll_total += ll_S[s];
    }

    vec N_s_out = zeros(n_params);

    // Loop over rows of bs_c
    for(int j=0; j<n_params; j++) {

      // Setup Ellipse
      vec nu(S);

      if (j < p) {
        // Case 1: Weights 'w' (Correlated across sources)
        vec Z = randn(S);
        nu = chol_cov_W * Z;
      } else {
        // Case 2: Alphas 'a' or 'a0' (Independent across sources)
        vec nu = randn(S);
      }

      // B. Slice Sampling Setup
      rowvec f_curr = bs_c.row(j); // The current row across all S

      double u_uni = R::runif(0, 1);
      double log_y_threshold = current_ll_total + log(u_uni);

      double theta = R::runif(0, 2 * M_PI);
      double theta_min = theta - 2 * M_PI;
      double theta_max = theta;

      int n_s = 0;

      // C. Slice Loop
      while(n_s < S_max) {
        n_s++;

        // Propose New Row
        rowvec f_prop_row = f_curr * cos(theta) + trans(nu) * sin(theta);

        // Calculate Likelihood Delta
        double prop_ll_total = 0;
        std::vector<vec> resid_S_prop(S); // Store potential new residuals

        for(int s=0; s<S; s++) {
          double val_new = f_prop_row(s);

          resid_S_prop[s] = resid_S[s];
          mat X = X_s_list[s];
          vec C = C_s_list[s];
          double lam = lambda_S(s);

          if (j == 2*p) {
            // CASE 1: Global a0 update (recompute full vector)
            vec bs_col = bs_c.col(s);
            vec bias_old = ntl::get_beta(bs_col.subvec(0, p-1),
                                    bs_col.subvec(p, 2*p-1),
                                    bs_col(2*p), lam);

            // Construct New Bias (Vector)
            vec bias_new = ntl::get_beta(bs_col.subvec(0, p-1),
                                    bs_col.subvec(p, 2*p-1),
                                    val_new, lam); // Use val_new for a0

            vec diff = bias_new - bias_old;
            resid_S_prop[s] -= X * diff;

          } else {
            // CASE 2: Local w_k or a_k update
            int k = j % p; // Feature index

            // Get the other fixed parameters for this feature
            double w_fixed = bs_c(k, s);
            double a_fixed = bs_c(k+p, s);
            double a0_fixed = bs_c(2*p, s);

            // Calculate Threshold (Scalar)
            double thresh_prob = std::pow(ntl::pnorm_custom(a0_fixed), 1.0/lam);
            double threshold = ntl::qnorm_custom(thresh_prob);

            // Calculate Old Beta_k (Scalar)
            double act_old = a_fixed - threshold;
            double beta_old_k = w_fixed * (act_old > 0 ? act_old : 0.0);

            // Calculate New Beta_k (Scalar)
            // We swap in 'val_new' for the parameter that changed
            double w_temp = (j < p) ? val_new : w_fixed;
            double a_temp = (j < p) ? a_fixed : val_new;

            double act_new = a_temp - threshold;
            double beta_new_k = w_temp * (act_new > 0 ? act_new : 0.0);

            // Delta
            double d_val = beta_new_k - beta_old_k;
            resid_S_prop[s] -= X.col(k) * d_val;
          }

          prop_ll_total += log_lik_aft(resid_S_prop[s], C, sd_y_S(s), fam_code);
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


  // Internal Helper for MH Step
  double update_sigma_jeffreys_tl(const vec& resid, const vec& C, double current_sigma, double step_size, int fam_code) {
    double log_sigma_curr = log(current_sigma);
    double log_sigma_prop = R::rnorm(log_sigma_curr, step_size);
    double sigma_prop     = exp(log_sigma_prop);

    double ll_curr = log_lik_aft(resid, C, current_sigma, fam_code);
    double ll_prop = log_lik_aft(resid, C, sigma_prop, fam_code);

    if (log(R::runif(0, 1)) < (ll_prop - ll_curr)) {
      return sigma_prop;
    } else {
      return current_sigma;
    }
  }

  // Update Target Sigma
  // [[Rcpp::export]]
  double update_sigma_target_tl_cpp(arma::vec bt_c, const arma::mat& X, const arma::vec& Y, const arma::vec& C,
                                    double current_sigma, double lambda, int fam_code, double step_size = 0.1) {
    int p = X.n_cols;

    // Reconstruct Beta_T
    vec w = bt_c.subvec(0, p-1);
    vec a = bt_c.subvec(p, 2*p-1);
    double a0 = bt_c(2*p);
    vec beta_T = ntl::get_beta(w, a, a0, lambda);

    vec resid = Y - X * beta_T;
    return update_sigma_jeffreys_tl(resid, C, current_sigma, step_size, fam_code);
  }

  // Update Source Sigma (Per Source)
  // [[Rcpp::export]]
  double update_sigma_source_tl_cpp(arma::vec bs_col, const arma::vec& beta_T, const arma::mat& X, const arma::vec& Y, const arma::vec& C,
                                    double current_sigma, double lambda, int fam_code, double step_size = 0.1) {
    int p = X.n_cols;

    // Reconstruct Bias and Beta_S
    vec bias = ntl::calc_bias_vec(bs_col, p, lambda);
    vec beta_s = beta_T + bias;

    vec resid = Y - X * beta_s;
    return update_sigma_jeffreys_tl(resid, C, current_sigma, step_size, fam_code);
  }
