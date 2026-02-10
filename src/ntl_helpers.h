#pragma once
#include <RcppArmadillo.h>

namespace ntl {

// keep these inline to avoid duplicate symbol linker issues
inline double qnorm_custom(double p) { return R::qnorm(p, 0.0, 1.0, 1, 0); }
inline double pnorm_custom(double x) { return R::pnorm(x, 0.0, 1.0, 1, 0); }

inline arma::vec T_n1_cpp(arma::vec x) {
  arma::uword n = x.n_elem;
  arma::vec out(n);
  for(arma::uword i = 0; i < n; ++i) {out[i] = (x[i] > 0.0);}
  return out; // 1.0 / (1.0 + arma::exp(-20.0 * x));
}

// 2.1 Helper: The Neuronized Transformation H(x) = sign(x)*exp(x^2)
inline arma::vec H_n1_cpp(arma::vec w, double phi=2.0, double d=2.0) {
  return phi * arma::sign(w) % arma::pow(arma::expm1(d * arma::square(w)), 0.5/d);
}

inline arma::vec T_n2_cpp(arma::vec x) {
  arma::uword n = x.n_elem;
  arma::vec out(n);
  for(arma::uword i = 0; i < n; ++i) {out[i] = (x[i] > 0.0);}
  return out; // 1.0 / (1.0 + arma::exp(-20.0 * x));
}

// 2.1 Helper: The Neuronized Transformation H(x) = sign(x)*exp(x^2)
inline arma::vec H_n2_cpp(arma::vec w, double phi=2.0) {
  return phi * arma::sign(w) % arma::sqrt(arma::abs(w)) % arma::exp(0.5 * arma::square(w));
}

inline arma::vec T_c_cpp(arma::vec x) {
  return arma::max(x, zeros(size(x)));
}

// 2.1 Helper: The Neuronized Transformation H(x) = sign(x)*exp(x^2)
inline arma::vec H_c_cpp(arma::vec w) {
  return arma::sign(w) % arma::exp(0.5 * arma::square(w));
}

inline arma::vec T_l_cpp(arma::vec x) {
  return arma::max(x, zeros(size(x)));
}

// 2.1 Helper: The Neuronized Transformation H(x) = sign(x)*exp(x^2)
inline arma::vec H_l_cpp(arma::vec w) {
  return w;
}

// 3. Helper: Calculate Beta from latent vectors w, a, a0
inline arma::vec get_beta(const arma::vec& w, const arma::vec& a, double tau,
             double a0, double lambda, int slab_code) {
  // Threshold calculation: qnorm(pnorm(a0)^(1/lambda))
  double thresh_prob = std::pow(pnorm_custom(a0), 1.0/lambda);
  double threshold = qnorm_custom(thresh_prob);

  // beta = w * T(a - threshold)
  if (slab_code==1){
    return tau * H_l_cpp(w) % T_l_cpp(a - threshold);
  }else if (slab_code==2){
    return tau * H_c_cpp(w) % T_c_cpp(a - threshold);
  }else if (slab_code == 3){
    return tau * H_n1_cpp(w) % T_n1_cpp(a - threshold);
  }else {
    return tau * H_n2_cpp(w) % T_n2_cpp(a - threshold);
  }
}

inline double log_lik_resid(const arma::vec& resid, double sd_y) {
  double n = resid.n_elem;
  double rss = sum(square(resid));
  return -0.5 * n * log(2 * M_PI) - n * log(sd_y) - 0.5 * rss / (sd_y * sd_y);
}

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

inline arma::vec calc_bias_vec(const arma::vec& bs_col, double tau_s, int p, double lambda, int slab_code) {
  arma::vec w = bs_col.subvec(0, p-1);
  arma::vec a = bs_col.subvec(p, 2*p-1);
  double a0 = bs_col(2*p);
  return get_beta(w, a, tau_s, a0, lambda, slab_code);
}

} // namespace ntl
