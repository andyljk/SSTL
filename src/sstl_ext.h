#pragma once
// Extension of the ESS sweep for non-grouped transfer models with hard-indicator
// slabs ('nlp', 'nlp2', 'guassian'). Added to the original sweep:
//   * joint target/source factor support proposals on blocks of 1, 2 or 4
//     coordinates, and source-only pair proposals (collapsed MH moves);
//   * interweaving of the slab scales with the active coefficients held fixed;
//   * with probability ghf_prob, the GHF kernel: gate refresh given the supports (G),
//     one joint ESS update of all thresholds (H), and a prior refresh of the full
//     coefficient state when every support is empty (F).
// The source sweep can also draw inactive source weights directly from their N(0, 1)
// conditional (update_sources_direct). Every move leaves the SSTL posterior invariant.
#include "sstl_helpers.h"
#include <algorithm>

namespace sstl {
namespace ext {

struct Settings {
  bool on = false, outer_ess = true, direct_inactive = false, asis = false, mask_correction = true;
  int factor_reps = 0, source_pair_reps = 0, irls_iter = 3;
  double pair_prob = 0.8, quad_prob = 0.0, ghf_prob = 0.0, t_weight = 0.2, working_sd = 3.0;
  arma::vec single_cw, pair_cw, quad_cw;  // cumulative selector weights
  arma::umat pairs, quads;                // 0-based coordinates, one block per row
  arma::mat source_pair_cw;               // cumulative pair weights, one column per source
  // factor proposed/accepted/support changed, source pair proposed/accepted/support changed,
  // GHF runs, GHF threshold moves that changed a coefficient, empty-model refreshes, ASIS updates
  arma::vec counts = arma::zeros(10);
};

// One study: the target (index 0) or a source. b points at its latent vector (w, a, a0).
struct Study {
  const arma::mat* X;
  const arma::vec* Y;
  const arma::vec* C;  // AFT observation indicators, nullptr otherwise
  double sig, lambda, s_tau;
  double* b;
  double* xi;
  arma::vec* resid;
};

struct Model {
  int fam, slab;
  bool aft;
  double df;
  int p, S_max;
};

// ---- slab transform H, its inverse and log |H'| (matching calc_scalar_beta)
inline double sgn(double x) { return x > 0 ? 1.0 : -1.0; }

inline double H(double w, int slab) {
  if (slab == 3) return 2.0 * sgn(w) * std::pow(std::expm1(2.0 * w * w), 0.25);
  if (slab == 4) return 2.0 * sgn(w) * std::sqrt(std::abs(w)) * std::exp(0.5 * w * w);
  return w;
}

// Principal branch of Lambert W for z >= 0 (Halley iterations)
inline double lambert_w(double z) {
  double y = std::log1p(z);
  for (int i = 0; i < 100; ++i) {
    double ey = std::exp(y), f = y * ey - z;
    double step = f / (ey * (y + 1.0) - (y + 2.0) * f / (2.0 * (y + 1.0)));
    y -= step;
    if (std::abs(step) <= 1e-15 * (1.0 + std::abs(y))) break;
  }
  return y;
}

inline double H_inv(double h, int slab) {
  double q = 0.5 * h;
  if (slab == 3) return sgn(h) * std::sqrt(0.5 * std::log1p(q * q * q * q));
  // nlp2: (h / 2)^2 = |w| exp(w^2), so 2 w^2 = W(2 (h / 2)^4)
  if (slab == 4) return sgn(h) * std::sqrt(0.5 * lambert_w(2.0 * q * q * q * q));
  return h;
}

inline double log_dH(double w, int slab) {
  double a = std::abs(w);
  if (slab == 3) return M_LN2 + std::log(a) + 2.0 * a * a - 0.75 * std::log(std::expm1(2.0 * a * a));
  if (slab == 4) return 0.5 * a * a - 0.5 * std::log(a) + std::log1p(2.0 * a * a);
  return 0.0;
}

// Log density of an active coefficient beta = tau H(w), w ~ N(0, 1)
inline double log_slab(double beta, double tau, int slab) {
  double w = H_inv(beta / tau, slab);
  return -0.5 * w * w - M_LN_SQRT_2PI - std::log(tau) - log_dH(w, slab);
}

// Prior probability of one study's support with s of p coordinates active, after
// integrating the threshold and gates: lambda B(s + 1, lambda + p - s)
inline double log_supp(int s, int p, double lambda) {
  return std::log(lambda) + R::lbeta(s + 1.0, lambda + p - s);
}

inline double loglik(const Model& M, const Study& st, const arma::vec& r) {
  return M.aft ? log_lik_aft(r, *st.Y, *st.C, st.sig, M.fam)
               : log_lik_general_kernel(r, *st.Y, st.sig, M.fam, M.df);
}

// Score g and curvature c = -(second derivative) of each observation's log likelihood
// with respect to its linear predictor (eta = Y - resid; AFT: resid = log time - eta).
// Used only to build proposals, so the exponential terms (Weibull, Poisson) are evaluated
// at a capped argument: far from the data a few observations would otherwise get weights
// like exp(40) and make the small working systems singular.
inline void eta_derivs(const Model& M, const Study& st, const arma::vec& r, arma::vec& g, arma::vec& c) {
  int n = r.n_elem;
  const arma::vec& Y = *st.Y;
  double s = st.sig;
  g.set_size(n);
  c.set_size(n);
  for (int i = 0; i < n; ++i) {
    if (!M.aft) {
      double eta = Y(i) - r(i);
      if (M.fam == 1) {
        g(i) = r(i) / (s * s);
        c(i) = 1.0 / (s * s);
      } else if (M.fam == 2) {
        double mu = R::plogis(eta, 0.0, 1.0, 1, 0);
        g(i) = Y(i) - mu;
        c(i) = mu * (1.0 - mu);
      } else { // Poisson
        double mu = std::exp(std::min(eta, 20.0));
        g(i) = Y(i) - mu;
        c(i) = mu;
      }
    } else {
      double z = r(i) / s, ci = (*st.C)(i), lz, lzz;
      if (M.fam == 1) { // Weibull
        double ez = std::exp(std::min(z, 10.0));
        lz = ci - ez;
        lzz = -ez;
      } else if (M.fam == 2) { // log-logistic
        double F = R::plogis(z, 0.0, 1.0, 1, 0);
        lz = ci - (1.0 + ci) * F;
        lzz = -(1.0 + ci) * F * (1.0 - F);
      } else if (ci == 1.0) { // lognormal event
        lz = -z;
        lzz = -1.0;
      } else { // lognormal censored: log Phi(-z)
        double h = std::exp(R::dnorm(z, 0.0, 1.0, 1) - R::pnorm(z, 0.0, 1.0, 0, 1));
        lz = -h;
        lzz = -h * (h - z);
      }
      g(i) = -lz / s;
      c(i) = -lzz / (s * s);
    }
  }
}

// Quadratic approximation of a study's log likelihood in its block coefficients theta,
// linearized at theta_star: ll(theta) ~ -theta' G theta / 2 + theta' r + const.
// rc is the residual with the block removed.
inline void working(const Model& M, const Study& st, const arma::mat& XB, const arma::vec& rc,
                    const arma::vec& theta_star, arma::mat& G, arma::vec& r) {
  arma::vec xt = XB * theta_star, g, c;
  eta_derivs(M, st, rc - xt, g, c);
  G = XB.t() * (XB.each_col() % c);
  G = 0.5 * (G + G.t());
  r = XB.t() * (g + c % xt);
}

inline arma::uvec mask_idx(int mask, int m) {
  std::vector<arma::uword> out;
  for (int t = 0; t < m; ++t) if (mask & (1 << t)) out.push_back(t);
  return arma::conv_to<arma::uvec>::from(out);
}

inline double log_sum_exp(const arma::vec& x) {
  double mx = x.max();
  return mx + std::log(arma::sum(arma::exp(x - mx)));
}

// Working conditional of x_A under exp(-x'Px/2 + l'x) with x outside A at zero: mean mu,
// upper Cholesky factor R of P_AA, and the log evidence against a N(0, sd_prior^2) working
// prior whose precision is already inside P.
struct Cond {
  arma::vec mu;
  arma::mat R;
  double logZ = 0.0;
};

inline Cond cond(const arma::mat& P, const arma::vec& l, const arma::uvec& A, double sd_prior) {
  Cond out;
  if (A.n_elem == 0) return out;
  out.R = arma::chol(arma::mat(P.submat(A, A)));
  arma::vec y = arma::solve(arma::trimatl(out.R.t()), arma::vec(l.elem(A)));
  out.mu = arma::solve(arma::trimatu(out.R), y);
  out.logZ = -(double) A.n_elem * std::log(sd_prior) - arma::sum(arma::log(out.R.diag())) + 0.5 * arma::dot(y, y);
  return out;
}

// (1 - tw) N(mu, P^-1) + tw t_3(mu, P^-1)
inline double log_mix(const arma::vec& x, const Cond& cd, double tw) {
  int d = x.n_elem;
  if (d == 0) return 0.0;
  arma::vec z = cd.R * (x - cd.mu);
  double delta = arma::dot(z, z), half_logdet = arma::sum(arma::log(cd.R.diag()));
  double ln = -0.5 * d * std::log(2.0 * M_PI) + half_logdet - 0.5 * delta;
  if (tw <= 0.0) return ln;
  double lt = std::lgamma(0.5 * (3.0 + d)) - std::lgamma(1.5) - 0.5 * d * std::log(3.0 * M_PI) + half_logdet -
    0.5 * (3.0 + d) * std::log1p(delta / 3.0);
  double a = std::log1p(-tw) + ln, b = std::log(tw) + lt, mx = std::max(a, b);
  return mx + std::log(std::exp(a - mx) + std::exp(b - mx));
}

inline arma::vec draw_mix(const Cond& cd, double tw) {
  int d = cd.mu.n_elem;
  arma::vec z(d);
  for (int i = 0; i < d; ++i) z(i) = R::rnorm(0.0, 1.0);
  double scale = (R::runif(0.0, 1.0) < tw) ? 1.0 / std::sqrt(R::rchisq(3.0) / 3.0) : 1.0;
  return cd.mu + arma::solve(arma::trimatu(cd.R), z) * scale;
}

// Index drawn from normalized log probabilities
inline int draw_log(const arma::vec& lp) {
  double u = R::runif(0.0, 1.0), acc = 0.0;
  for (arma::uword i = 0; i < lp.n_elem; ++i) {
    acc += std::exp(lp(i));
    if (u < acc) return i;
  }
  return lp.n_elem - 1;
}

// Index drawn from cumulative weights
inline arma::uword draw_cum(const double* cw, arma::uword n) {
  double u = R::runif(0.0, 1.0) * cw[n - 1];
  return std::upper_bound(cw, cw + n, u) - cw;
}

// Mask distribution of one source block given the target block: log probabilities lp,
// the per-mask conditionals and the unnormalized log total lse.
struct SourceProp {
  arma::vec lp;
  std::vector<Cond> cd;
  double lse;
};

inline SourceProp source_prop(const arma::mat& P, const arma::vec& l, int m, int s_out, int p, double lambda,
                              double sd_prior) {
  int nm = 1 << m;
  SourceProp sp;
  sp.lp.set_size(nm);
  sp.cd.resize(nm);
  for (int mk = 0; mk < nm; ++mk) {
    arma::uvec A = mask_idx(mk, m);
    sp.cd[mk] = cond(P, l, A, sd_prior);
    sp.lp(mk) = log_supp(s_out + A.n_elem, p, lambda) + sp.cd[mk].logZ;
  }
  sp.lse = log_sum_exp(sp.lp);
  sp.lp -= sp.lse;
  return sp;
}

// Exact draw of a study's threshold and gates given its support (the G kernel).
inline void gate_refresh(double* b, const std::vector<char>& active, int p, double lambda) {
  int s = 0;
  for (int i = 0; i < p; ++i) s += active[i];
  // omega = 1 - Phi(psi) ~ Beta(s + 1, lambda + p - s), drawn as the gamma ratio gx / (gx + gy) so that
  // log Phi(psi) = log(1 - omega) = -log1p(gx / gy) keeps its precision when omega is close to 0 or 1
  // (a dense support with a small lambda puts omega within rounding of 1). a0 = Phi^{-1}(Phi(psi)^lambda).
  double gx = R::rgamma(s + 1.0, 1.0), gy = R::rgamma(lambda + p - s, 1.0);
  double a0 = R::qnorm(-lambda * std::log1p(gx / gy), 0.0, 1.0, 1, 1);
  double thr = threshold_from_a0(a0, lambda);
  b[2 * p] = a0;
  double log_up = R::pnorm(thr, 0.0, 1.0, 0, 1), log_lo = R::pnorm(thr, 0.0, 1.0, 1, 1);
  for (int i = 0; i < p; ++i) {
    double lu = std::log(R::runif(0.0, 1.0));
    b[p + i] = active[i] ? R::qnorm(lu + log_up, 0.0, 1.0, 0, 1) : R::qnorm(lu + log_lo, 0.0, 1.0, 1, 1);
    if ((b[p + i] > thr) != (bool) active[i]) {
      Rcpp::stop("gate refresh changed the support (gate %d: %.17g, threshold %.17g, a0 %.17g, %d of %d active, lambda %g)",
                 i, b[p + i], thr, a0, s, p, lambda);
    }
  }
}

inline std::vector<char> support(const double* b, int p, double lambda) {
  double thr = threshold_from_a0(b[2 * p], lambda);
  std::vector<char> act(p);
  for (int i = 0; i < p; ++i) act[i] = b[p + i] > thr;
  return act;
}

// One collapsed MH move on block B. src < 0: joint target/source move; src = k >= 1:
// source k only, with the target held fixed. Returns 0 if rejected, 1 if accepted with
// unchanged supports, 2 if accepted with a support change.
inline int factor_move(const Model& M, const Settings& E, std::vector<Study>& st, const arma::uvec& B, int src) {
  int p = M.p, m = B.n_elem, J = st.size();
  bool joint = src < 0;
  std::vector<int> studies;
  if (joint) for (int j = 0; j < J; ++j) studies.push_back(j);
  else studies.push_back(src);

  // Current block state; residual anchors with the block removed
  std::vector<int> mask(J, 0), s_out(J, 0);
  std::vector<arma::vec> coef(J, arma::zeros(m)), rc(J);
  std::vector<arma::mat> XB(J);
  std::vector<double> tau(J), sdp(J);
  for (int j = 0; j < J; ++j) {
    if (!joint && j != 0 && j != src) continue;
    const double* b = st[j].b;
    double thr = threshold_from_a0(b[2 * p], st[j].lambda);
    tau[j] = std::abs(*st[j].xi);
    sdp[j] = E.working_sd * tau[j];
    int s_tot = 0;
    for (int i = 0; i < p; ++i) s_tot += b[p + i] > thr;
    for (int t = 0; t < m; ++t) {
      int i = B(t);
      if (b[p + i] > thr) {
        mask[j] |= 1 << t;
        coef[j](t) = tau[j] * H(b[i], M.slab);
      }
    }
    s_out[j] = s_tot - __builtin_popcount(mask[j]);
    XB[j] = st[j].X->cols(B);
  }
  for (int j : studies) rc[j] = *st[j].resid + XB[j] * (j == 0 ? coef[0] : arma::vec(coef[0] + coef[j]));

  int n_it = (!M.aft && M.fam == 1) ? 1 : E.irls_iter;
  int nm = 1 << m;
  std::vector<arma::mat> G(J), P(J);
  std::vector<arma::vec> r(J);
  std::vector<arma::vec> new_coef(J);
  std::vector<int> new_mask(J);
  double lq_fwd = 0.0, lq_rev = 0.0;

  if (joint) {
    // Working Gaussian iterations from zero; Schur complement for the target block
    arma::vec u_star = arma::zeros(m);
    std::vector<arma::vec> v_star(J, arma::zeros(m));
    arma::mat S;
    arma::vec lu;
    for (int it = 0; it < n_it; ++it) {
      working(M, st[0], XB[0], rc[0], u_star, G[0], r[0]);
      S = G[0] + arma::eye(m, m) / (sdp[0] * sdp[0]);
      lu = r[0];
      for (int k = 1; k < J; ++k) {
        working(M, st[k], XB[k], rc[k], u_star + v_star[k], G[k], r[k]);
        P[k] = G[k] + arma::eye(m, m) / (sdp[k] * sdp[k]);
        // With P = G + d I: G - G P^{-1} G = d P^{-1} G and r - G P^{-1} r = d P^{-1} r (no cancellation)
        double dk = 1.0 / (sdp[k] * sdp[k]);
        S += dk * arma::solve(P[k], G[k]);
        lu += dk * arma::solve(P[k], r[k]);
      }
      S = 0.5 * (S + S.t());
      if (it < n_it - 1) {
        u_star = arma::solve(S, lu);
        for (int k = 1; k < J; ++k) v_star[k] = arma::solve(P[k], arma::vec(r[k] - G[k] * u_star));
      }
    }

    // Target mask distribution, optionally corrected by the source mixture normalizers at
    // the working target means
    arma::vec lp0(nm);
    std::vector<Cond> cd0(nm);
    arma::mat P0 = G[0] + arma::eye(m, m) / (sdp[0] * sdp[0]);
    for (int mk = 0; mk < nm; ++mk) {
      arma::uvec A = mask_idx(mk, m);
      cd0[mk] = cond(S, lu, A, sdp[0]);
      double l = log_supp(s_out[0] + A.n_elem, p, st[0].lambda);
      if (!E.mask_correction) {
        l += cd0[mk].logZ;
      } else {
        l += cond(P0, r[0], A, sdp[0]).logZ;
        arma::vec u_hat = arma::zeros(m);
        if (A.n_elem) u_hat.elem(A) = cd0[mk].mu;
        for (int k = 1; k < J; ++k) {
          SourceProp sp = source_prop(P[k], r[k] - G[k] * u_hat, m, s_out[k], p, st[k].lambda, sdp[k]);
          l += -0.5 * arma::dot(u_hat, G[k] * u_hat) + arma::dot(r[k], u_hat) + sp.lse;
        }
      }
      lp0(mk) = l;
    }
    lp0 -= log_sum_exp(lp0);

    // Forward draw and density
    new_mask[0] = draw_log(lp0);
    new_coef[0] = arma::zeros(m);
    arma::uvec A0 = mask_idx(new_mask[0], m), A0_old = mask_idx(mask[0], m);
    if (A0.n_elem) new_coef[0].elem(A0) = draw_mix(cd0[new_mask[0]], E.t_weight);
    lq_fwd += lp0(new_mask[0]) + log_mix(new_coef[0].elem(A0), cd0[new_mask[0]], E.t_weight);
    lq_rev += lp0(mask[0]) + log_mix(coef[0].elem(A0_old), cd0[mask[0]], E.t_weight);
    for (int k = 1; k < J; ++k) {
      SourceProp sp = source_prop(P[k], r[k] - G[k] * new_coef[0], m, s_out[k], p, st[k].lambda, sdp[k]);
      new_mask[k] = draw_log(sp.lp);
      new_coef[k] = arma::zeros(m);
      arma::uvec A = mask_idx(new_mask[k], m);
      if (A.n_elem) new_coef[k].elem(A) = draw_mix(sp.cd[new_mask[k]], E.t_weight);
      lq_fwd += sp.lp(new_mask[k]) + log_mix(new_coef[k].elem(A), sp.cd[new_mask[k]], E.t_weight);
      // reverse: source densities at the current target block
      SourceProp sr = source_prop(P[k], r[k] - G[k] * coef[0], m, s_out[k], p, st[k].lambda, sdp[k]);
      arma::uvec A_old = mask_idx(mask[k], m);
      lq_rev += sr.lp(mask[k]) + log_mix(coef[k].elem(A_old), sr.cd[mask[k]], E.t_weight);
    }
  } else {
    int k = src;
    const arma::vec& u = coef[0];
    arma::vec v_star = arma::zeros(m), l;
    for (int it = 0; it < n_it; ++it) {
      working(M, st[k], XB[k], rc[k], u + v_star, G[k], r[k]);
      P[k] = G[k] + arma::eye(m, m) / (sdp[k] * sdp[k]);
      l = r[k] - G[k] * u;
      if (it < n_it - 1) v_star = arma::solve(P[k], l);
    }
    SourceProp sp = source_prop(P[k], l, m, s_out[k], p, st[k].lambda, sdp[k]);
    new_mask[k] = draw_log(sp.lp);
    new_coef[k] = arma::zeros(m);
    arma::uvec A = mask_idx(new_mask[k], m), A_old = mask_idx(mask[k], m);
    if (A.n_elem) new_coef[k].elem(A) = draw_mix(sp.cd[new_mask[k]], E.t_weight);
    lq_fwd += sp.lp(new_mask[k]) + log_mix(new_coef[k].elem(A), sp.cd[new_mask[k]], E.t_weight);
    lq_rev += sp.lp(mask[k]) + log_mix(coef[k].elem(A_old), sp.cd[mask[k]], E.t_weight);
    new_coef[0] = coef[0];
  }

  // Collapsed acceptance ratio: likelihood, support prior, slab density, proposal
  double log_a = lq_rev - lq_fwd;
  std::vector<arma::vec> new_resid(J);
  for (int j : studies) {
    new_resid[j] = rc[j] - XB[j] * (j == 0 ? new_coef[0] : arma::vec(new_coef[0] + new_coef[j]));
    log_a += loglik(M, st[j], new_resid[j]) - loglik(M, st[j], *st[j].resid);
    log_a += log_supp(s_out[j] + __builtin_popcount(new_mask[j]), p, st[j].lambda) -
      log_supp(s_out[j] + __builtin_popcount(mask[j]), p, st[j].lambda);
    for (int t = 0; t < m; ++t) {
      if (new_mask[j] & (1 << t)) log_a += log_slab(new_coef[j](t), tau[j], M.slab);
      if (mask[j] & (1 << t)) log_a -= log_slab(coef[j](t), tau[j], M.slab);
    }
  }
  if (!(std::log(R::runif(0.0, 1.0)) < log_a)) return 0;

  // Lift: active weights through the inverse slab map, inactive block weights from the prior,
  // gates and threshold of each study whose support changed from their exact conditional
  bool supp_changed = false;
  for (int j : studies) {
    double* b = st[j].b;
    std::vector<char> act;
    if (new_mask[j] != mask[j]) act = support(b, p, st[j].lambda);
    for (int t = 0; t < m; ++t) {
      int i = B(t);
      bool on = new_mask[j] & (1 << t);
      b[i] = on ? H_inv(new_coef[j](t) / tau[j], M.slab) : R::rnorm(0.0, 1.0);
      if (new_mask[j] != mask[j]) act[i] = on;
    }
    if (new_mask[j] != mask[j]) {
      gate_refresh(b, act, p, st[j].lambda);
      supp_changed = true;
    }
    st[j].resid->swap(new_resid[j]);
  }
  return supp_changed ? 2 : 1;
}

// Slab-scale interweaving: slice update of eta = log tau given the active coefficients
// beta_A, which stay fixed; active weights are rebuilt and the sign of xi redrawn.
inline void asis(const Model& M, Study& st) {
  int p = M.p;
  double* b = st.b;
  double thr = threshold_from_a0(b[2 * p], st.lambda), tau = std::abs(*st.xi);
  std::vector<int> A;
  std::vector<double> beta;
  for (int i = 0; i < p; ++i) {
    if (b[p + i] > thr) {
      A.push_back(i);
      beta.push_back(tau * H(b[i], M.slab));
    }
  }
  if (A.empty()) { // the scale shadow has its prior conditional
    *st.xi = R::rnorm(0.0, st.s_tau);
    return;
  }
  auto logf = [&](double eta) {
    double t = std::exp(eta), out = eta - t * t / (2.0 * st.s_tau * st.s_tau);
    for (double bb : beta) out += log_slab(bb, t, M.slab);
    return out;
  };
  // Stepping-out slice sampler (width 1, at most 50 steps) with shrinkage
  double eta = std::log(tau);
  double ly = logf(eta) + std::log(R::runif(0.0, 1.0));
  double lo = eta - R::runif(0.0, 1.0), hi = lo + 1.0;
  int jl = (int) std::floor(50 * R::runif(0.0, 1.0)), kr = 49 - jl;
  while (jl-- > 0 && logf(lo) > ly) lo -= 1.0;
  while (kr-- > 0 && logf(hi) > ly) hi += 1.0;
  double eta_new;
  while (true) {
    eta_new = lo + R::runif(0.0, 1.0) * (hi - lo);
    if (logf(eta_new) > ly) break;
    if (eta_new < eta) lo = eta_new;
    else hi = eta_new;
  }
  double t_new = std::exp(eta_new);
  for (size_t q = 0; q < A.size(); ++q) b[A[q]] = H_inv(beta[q] / t_new, M.slab);
  *st.xi = (R::runif(0.0, 1.0) < 0.5 ? 1.0 : -1.0) * t_new;
}

// GHF kernel: G (gate refresh), H (joint ESS on all thresholds), F (empty-model refresh)
inline void ghf(const Model& M, Settings& E, std::vector<Study>& st) {
  int p = M.p, J = st.size();
  E.counts(6) += 1;
  for (int j = 0; j < J; ++j) gate_refresh(st[j].b, support(st[j].b, p, st[j].lambda), p, st[j].lambda);

  // H: one ESS update of (a0_target, a0_source_1, ...) with gates, weights and scales fixed
  std::vector<arma::vec> tau_h(J), coef(J), coef_prop(J), resid_prop(J);
  double ll = 0.0;
  for (int j = 0; j < J; ++j) {
    double tau = std::abs(*st[j].xi), thr = threshold_from_a0(st[j].b[2 * p], st[j].lambda);
    tau_h[j].set_size(p);
    coef[j].set_size(p);
    for (int i = 0; i < p; ++i) {
      tau_h[j](i) = tau * H(st[j].b[i], M.slab);
      coef[j](i) = st[j].b[p + i] > thr ? tau_h[j](i) : 0.0;
    }
    ll += loglik(M, st[j], *st[j].resid);
  }
  arma::vec f_curr(J), nu(J);
  for (int j = 0; j < J; ++j) {
    f_curr(j) = st[j].b[2 * p];
    nu(j) = R::rnorm(0.0, 1.0);
  }
  double log_y = ll + std::log(R::runif(0.0, 1.0));
  double theta = R::runif(0.0, 2.0 * M_PI), theta_min = theta - 2.0 * M_PI, theta_max = theta;
  for (int n_s = 0; n_s < M.S_max; ++n_s) {
    arma::vec f_prop = f_curr * std::cos(theta) + nu * std::sin(theta);
    bool changed = false;
    for (int j = 0; j < J; ++j) {
      double thr = threshold_from_a0(f_prop(j), st[j].lambda);
      coef_prop[j].set_size(p);
      for (int i = 0; i < p; ++i) coef_prop[j](i) = st[j].b[p + i] > thr ? tau_h[j](i) : 0.0;
    }
    double ll_prop = 0.0;
    arma::vec d0 = coef_prop[0] - coef[0];
    for (int j = 0; j < J; ++j) {
      arma::vec d = j == 0 ? d0 : arma::vec(d0 + coef_prop[j] - coef[j]);
      arma::uvec nz = arma::find(d != 0.0);
      changed = changed || nz.n_elem > 0;
      resid_prop[j] = *st[j].resid - st[j].X->cols(nz) * d.elem(nz);
      ll_prop += loglik(M, st[j], resid_prop[j]);
    }
    if (ll_prop > log_y) {
      for (int j = 0; j < J; ++j) {
        st[j].b[2 * p] = f_prop(j);
        st[j].resid->swap(resid_prop[j]);
      }
      if (changed) E.counts(7) += 1;
      break;
    }
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = R::runif(theta_min, theta_max);
  }

  // F: when every support is empty, draw the full coefficient state from the prior
  // conditional on the empty support (weights, signed scale shadows, gates, thresholds).
  for (int j = 0; j < J; ++j) {
    std::vector<char> act = support(st[j].b, p, st[j].lambda);
    if (std::any_of(act.begin(), act.end(), [](char a) { return a; })) return;
  }
  E.counts(8) += 1;
  for (int j = 0; j < J; ++j) {
    for (int i = 0; i < p; ++i) st[j].b[i] = R::rnorm(0.0, 1.0);
    *st[j].xi = R::rnorm(0.0, st[j].s_tau);
    gate_refresh(st[j].b, std::vector<char>(p, 0), p, st[j].lambda);
  }
}

// Source sweep of update_source_joint_general / update_source_joint_aft (one-dimensional ESS
// on each latent coordinate of each source in turn), except that a weight whose gate is
// inactive leaves the likelihood unchanged and is drawn from its N(0, 1) conditional.
// ll_S(s, resid) is source s's log likelihood. Kept separate so that the original sweeps
// compile, and run, exactly as before.
template <class LLS>
void update_sources_direct(arma::mat& bs_c, std::vector<arma::vec>& resid_S, const std::vector<arma::mat>& X_s,
                           const arma::vec& lambda_S, const arma::vec& tau_S, int S_max, int slab_code, LLS& ll_S,
                           arma::vec& N_s) {
  int p = (bs_c.n_rows - 1) / 2, S = bs_c.n_cols, n_params = bs_c.n_rows;
  N_s.zeros(n_params);
  arma::vec resid_prop;
  for (int s = 0; s < S; s++) {
    arma::vec& resid = resid_S[s];
    const arma::mat& X = X_s[s];
    double tau_s = tau_S(s), lambda_s = lambda_S(s);
    double ll_curr = ll_S(s, resid);
    double threshold = threshold_from_a0(bs_c(2 * p, s), lambda_s);

    for (int j = 0; j < n_params; j++) {
      bool is_a0 = (j == 2 * p);
      int k = j % p;
      if (j < p && !(bs_c(k + p, s) > threshold)) {
        bs_c(j, s) = R::rnorm(0, 1);
        N_s(j) += 1;
        continue;
      }

      double f_curr = bs_c(j, s);
      double nu = R::rnorm(0, 1);
      double log_y_threshold = ll_curr + std::log(R::runif(0, 1));
      double theta = R::runif(0, 2 * M_PI);
      double theta_min = theta - 2 * M_PI;
      double theta_max = theta;
      int n_s = 0;

      double beta_old = 0.0;
      arma::vec a_s, bias_old, bias_prop, delta, tau_h;
      arma::uvec nz;
      if (is_a0) {
        arma::vec w_s = bs_c.col(s).subvec(0, p - 1);
        a_s = bs_c.col(s).subvec(p, 2 * p - 1);
        bias_old = get_beta(w_s, a_s, tau_s, f_curr, lambda_s, slab_code);
        tau_h = tau_s * slab_weight_vec(w_s, slab_code);
        bias_prop.set_size(p);
      } else {
        beta_old = calc_scalar_beta(bs_c(k, s), bs_c(k + p, s), threshold, tau_s, slab_code);
      }

      while (n_s < S_max) {
        n_s++;
        double f_prop = f_curr * std::cos(theta) + nu * std::sin(theta);
        bool changed;
        if (is_a0) {
          double threshold_prop = threshold_from_a0(f_prop, lambda_s);
          for (int i = 0; i < p; i++) bias_prop(i) = tau_h(i) * activation_scalar(a_s(i), threshold_prop, slab_code);
          delta = bias_prop - bias_old;
          nz = arma::find(delta != 0.0);
          changed = nz.n_elem > 0;
          if (changed) resid_prop = resid - X.cols(nz) * delta.elem(nz);
        } else {
          double beta_new = (j < p) ? calc_scalar_beta(f_prop, bs_c(k + p, s), threshold, tau_s, slab_code)
                                    : calc_scalar_beta(bs_c(k, s), f_prop, threshold, tau_s, slab_code);
          double d_val = beta_new - beta_old;
          changed = (d_val != 0.0);
          if (changed) resid_prop = resid - X.col(k) * d_val;
        }
        double ll_prop = changed ? ll_S(s, resid_prop) : ll_curr;
        if (!changed || ll_prop > log_y_threshold) {
          bs_c(j, s) = f_prop;
          if (changed) {
            resid.swap(resid_prop);
            ll_curr = ll_prop;
          }
          break;
        }
        if (theta < 0) theta_min = theta;
        else theta_max = theta;
        theta = R::runif(theta_min, theta_max);
      }
      N_s(j) += n_s;
    }
  }
}

// The added transitions, applied after the original ESS sweep of a TLChain. When any of
// them ran, the chain recomputes its residuals and target coefficients exactly at the end.
template <class Chain>
void sweep(Chain& ch, Settings& E) {
  int S = ch.S();
  Model M{ch.fam.code, ch.slab, ch.fam.aft, ch.fam.df, ch.p(), ch.S_max};
  std::vector<Study> st(S + 1);
  st[0] = Study{&ch.X_T, &ch.Y_T, ch.fam.aft ? &ch.C_T : nullptr, ch.sig_T, ch.lambda_T, ch.xi_prior,
                ch.bt.memptr(), &ch.xi, &ch.resid_T};
  for (int s = 0; s < S; ++s) {
    // 0.2 is the source slab-scale prior sd used by the source scale update in step_sources()
    st[s + 1] = Study{&ch.X_s[s], &ch.Y_s[s], ch.fam.aft ? &ch.C_s[s] : nullptr, ch.sig_s(s), ch.lambda_s(s), 0.2,
                      ch.bs.colptr(s), &ch.xi_s(s), &ch.resid_S[s]};
  }

  for (int r = 0; r < E.factor_reps; ++r) {
    arma::uvec B;
    if (E.quads.n_rows > 0 && R::runif(0.0, 1.0) < E.quad_prob) {
      B = E.quads.row(draw_cum(E.quad_cw.memptr(), E.quad_cw.n_elem)).t();
    } else if (R::runif(0.0, 1.0) < E.pair_prob) {
      B = E.pairs.row(draw_cum(E.pair_cw.memptr(), E.pair_cw.n_elem)).t();
    } else {
      B = arma::uvec{draw_cum(E.single_cw.memptr(), E.single_cw.n_elem)};
    }
    int res = factor_move(M, E, st, B, -1);
    E.counts(0) += 1;
    E.counts(1) += res > 0;
    E.counts(2) += res == 2;
  }
  for (int s = 0; s < S; ++s) {
    for (int r = 0; r < E.source_pair_reps; ++r) {
      arma::uvec B = E.pairs.row(draw_cum(E.source_pair_cw.colptr(s), E.source_pair_cw.n_rows)).t();
      int res = factor_move(M, E, st, B, s + 1);
      E.counts(3) += 1;
      E.counts(4) += res > 0;
      E.counts(5) += res == 2;
    }
  }
  if (E.asis) {
    for (Study& s : st) asis(M, s);
    E.counts(9) += 1;
  }
  bool ran_ghf = E.ghf_prob > 0.0 && R::runif(0.0, 1.0) < E.ghf_prob;
  if (ran_ghf) ghf(M, E, st);
  if (E.factor_reps > 0 || E.source_pair_reps > 0 || E.asis || ran_ghf) ch.reset_resid();
}

} // namespace ext
} // namespace sstl
