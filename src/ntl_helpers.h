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

} // namespace ntl
