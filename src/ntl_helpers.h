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
  return out;
}

inline arma::vec T_log_cpp(arma::vec x, double k=10.0) {
  arma::uword n = x.n_elem;
  arma::vec out(n);
  double* out_ptr = out.memptr();
  const double* x_ptr = x.memptr();
  for(arma::uword i = 0; i < n; ++i) {out_ptr[i] = 1.0 / (1.0 + std::exp(-k * x_ptr[i]));}
  return out;
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

inline double threshold_from_a0(double a0, double lambda) {
  if (!std::isfinite(a0) || !std::isfinite(lambda) || lambda <= 0.0) {
    return R_NaN;
  }
  double log_thresh_prob = R::pnorm(a0, 0.0, 1.0, 1, 1) / lambda;
  return R::qnorm(log_thresh_prob, 0.0, 1.0, 1, 1);
}


inline double activation_scalar(double a_val, double thresh,
                                int slab_code,
                                bool approx = false,
                                double k_apx = 10.0) {
  double centered = a_val - thresh;
  if (slab_code == 1 || slab_code == 2) {
    return centered > 0.0 ? centered : 0.0;
  }
  if (slab_code == 3) {
    return approx ? 1.0 / (1.0 + std::exp(-k_apx * centered))
      : (centered > 0.0 ? 1.0 : 0.0);
  }
  return centered > 0.0 ? 1.0 : 0.0;
}

inline arma::vec slab_weight_vec(const arma::vec& w, int slab_code) {
  if (slab_code == 1) {
    return H_l_cpp(w);
  }
  if (slab_code == 2) {
    return H_c_cpp(w);
  }
  if (slab_code == 3) {
    return H_n1_cpp(w);
  }
  if (slab_code == 5) {
    return H_l_cpp(w);
  }
  return H_n2_cpp(w);
}

// 3. Helper: Calculate Beta from latent vectors w, a, a0
inline arma::vec get_beta(const arma::vec& w, const arma::vec& a, double tau,
                          double a0, double lambda, int slab_code, bool approx=false, double k_apx=10.0) {
  double threshold = threshold_from_a0(a0, lambda);

  // beta = w * T(a - threshold)
  if (slab_code==1){
    return tau * H_l_cpp(w) % T_l_cpp(a - threshold);
  }else if (slab_code==2){
    return tau * H_c_cpp(w) % T_c_cpp(a - threshold);
  }else if (slab_code == 3){
    arma::vec act = approx ? T_log_cpp(a - threshold, k_apx) : T_n1_cpp(a - threshold);
    return tau * H_n1_cpp(w) % act;
  }else if (slab_code == 5){
    return tau * H_l_cpp(w) % T_n1_cpp(a - threshold);
  }else {
    return tau * H_n2_cpp(w) % T_n2_cpp(a - threshold);
  }
}

inline arma::vec get_beta_group(const arma::vec& w, const arma::vec& a_group,
                                const Rcpp::IntegerVector& group_map,
                                double tau, double a0, double lambda,
                                int slab_code, bool approx = false,
                                double k_apx = 10.0) {
  int p = w.n_elem;
  arma::vec a_expanded(p);
  for (int j = 0; j < p; ++j) {
    a_expanded(j) = a_group(group_map[j] - 1);
  }
  return get_beta(w, a_expanded, tau, a0, lambda, slab_code, approx, k_apx);
}

// --- Define Scalar Beta Calculation Lambda ---
inline double calc_scalar_beta(double w_val, double a_val, double thresh,
                               double tau_s, int slab_code,
                               bool approx = false, double k_apx = 10.0) {
  double act, h_w;
  if (slab_code == 1){
    act = (a_val > thresh) ? (a_val - thresh) : 0.0;
    h_w = w_val;
  } else if (slab_code == 2) {
    act = (a_val > thresh) ? (a_val - thresh) : 0.0;
    h_w = (w_val > 0 ? 1.0 : -1.0) * std::exp(0.5 * w_val * w_val);
  } else if (slab_code == 3){
    act = approx ? 1.0 / (1.0 + std::exp(-k_apx * (a_val-thresh))) : (a_val > thresh) ? 1.0 : 0.0;
    h_w = 2.0 * (w_val > 0 ? 1.0 : -1.0) * std::pow(std::expm1(2.0 * w_val * w_val), 0.25);
  } else if (slab_code == 5) {
    act = (a_val > thresh) ? 1.0 : 0.0;
    h_w = w_val;
  } else {
    act = (a_val > thresh) ? 1.0 : 0.0;
    h_w = 2.0 * (w_val > 0 ? 1.0 : -1.0) * std::sqrt(std::abs(w_val)) * std::exp(0.5 * w_val * w_val);
  }
  return tau_s * h_w * act;
};

inline double log_lik_resid(const arma::vec& resid, double sd_y) {
  double n = resid.n_elem;
  double rss = sum(square(resid));
  return -0.5 * n * log(2 * M_PI) - n * log(sd_y) - 0.5 * rss / (sd_y * sd_y);
}

inline double log_lik_aft(const arma::vec& resid, const arma::vec& C, double sd_y, int fam_code) {
  double ll = 0.0;
  double inv_sd = 1.0 / sd_y;
  double log_sd = std::log(sd_y);
  int n = resid.n_elem;
  const double* r_ptr = resid.memptr();
  const double* c_ptr = C.memptr();

  if (fam_code == 1) { // --- Weibull ---
    for(int i = 0; i < n; i++) {
      double zi = r_ptr[i] * inv_sd;
      double ci = c_ptr[i];
      ll += ci * zi - std::exp(zi) - ci * log_sd;
    }

  } else if (fam_code == 2) { // --- Log-Logistic ---
    for(int i = 0; i < n; ++i) {
      double zi = r_ptr[i] * inv_sd;
      double ci = c_ptr[i];
      ll += ci * (zi - log_sd) - (1.0 + ci) * std::log1p(std::exp(zi));
    }

  } else if (fam_code == 3) { // --- Log-Normal (Gaussian Errors) ---
    for(int i = 0; i < n; ++i) {
      double zi = r_ptr[i] * inv_sd;
      if(c_ptr[i] == 1.0) {
        ll += R::dnorm(zi, 0.0, 1.0, 1) - log_sd;
      } else {
        ll += R::pnorm(-zi, 0.0, 1.0, 1, 1);
      }
    }
  }

  return ll;
}

inline arma::vec calc_bias_vec(const arma::vec& bs_col, double tau_s, int p,
                               double lambda, int slab_code,
                               bool approx = false, double k_apx = 10.0) {
  arma::vec w = bs_col.subvec(0, p-1);
  arma::vec a = bs_col.subvec(p, 2*p-1);
  double a0 = bs_col(2*p);
  return get_beta(w, a, tau_s, a0, lambda, slab_code, approx, k_apx);
}

inline arma::vec calc_bias_vec_group(const arma::vec& bs_col, double tau_s,
                                     const Rcpp::IntegerVector& group_map,
                                     double lambda, int slab_code,
                                     bool approx = false,
                                     double k_apx = 10.0) {
  int p = group_map.size();
  int G = Rcpp::max(group_map);
  arma::vec w = bs_col.subvec(0, p - 1);
  arma::vec a = bs_col.subvec(p, p + G - 1);
  double a0 = bs_col(p + G);
  return get_beta_group(w, a, group_map, tau_s, a0, lambda, slab_code, approx, k_apx);
}

} // namespace ntl
