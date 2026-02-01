#pragma once
#include <RcppArmadillo.h>

namespace ntl {

// keep these inline to avoid duplicate symbol linker issues
inline double qnorm_custom(double p) { return R::qnorm(p, 0.0, 1.0, 1, 0); }
inline double pnorm_custom(double x) { return R::pnorm(x, 0.0, 1.0, 1, 0); }

inline arma::vec T_n_cpp(const arma::vec& x) {
  return arma::max(x, arma::zeros(arma::size(x)));
}

inline arma::vec get_beta(const arma::vec& w, const arma::vec& a, double a0, double lambda) {
  double thresh_prob = std::pow(pnorm_custom(a0), 1.0 / lambda);
  double threshold = qnorm_custom(thresh_prob);
  return w % T_n_cpp(a - threshold);
}

inline arma::vec calc_bias_vec(const arma::vec& bs_col, int p, double lambda) {
  arma::vec w  = bs_col.subvec(0, p - 1);
  arma::vec a  = bs_col.subvec(p, 2 * p - 1);
  double a0    = bs_col(2 * p);
  return get_beta(w, a, a0, lambda);
}

inline double log_lik_resid(const arma::vec& resid, double sd_y) {
  double n = resid.n_elem;
  double rss = sum(square(resid));
  return -0.5 * n * log(2 * M_PI) - n * log(sd_y) - 0.5 * rss / (sd_y * sd_y);
}

// --- Survival AFT Log-Likelihood ---
inline double log_lik_aft(const arma::vec& resid, const arma::vec& C, double sd_y, int fam_code) {
  arma::vec z = resid / sd_y;
  double ll = 0.0;

  if (fam_code == 1) { // --- Weibull ---
    double term1 = dot(C, z) - accu(C) * log(sd_y);
    double term2 = sum(exp(z));
    ll = term1 - term2;

  } else if (fam_code == 2) { // --- Log-Logistic ---
    arma::vec log_denom = log(1.0 + exp(z));
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

} // namespace ntl
