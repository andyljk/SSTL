#pragma once
// Extension of the ESS sweep for the ReLU-gate slabs ('exp', 'poly') of non-grouped
// transfer models (SSTL_efficient_ReLU.pdf). Per population the coefficient is
//   c_i = tau sign(w_i) g(|w_i|) (a_i - psi)_+,  g(r) = r ('exp') or exp(r^2 / 2) ('poly'),
// so, unlike a hard gate, the threshold psi and the gate excess a_i - psi scale every active
// coefficient. Added to the original sweep:
//   * joint target/source support and coefficient blocks with |w|, tau and psi held fixed
//     (factor_move_relu), and source-only pair blocks;
//   * signed pair exchanges of the latent (w, a) of two correlated coordinates in every
//     population at once (exchange; also used with hard gates);
//   * interweaving at fixed coefficients: radii |w| of active coordinates, inactive weights,
//     the slab scale and the threshold (interweave_relu);
//   * with probability ghf_prob, the GHF selector adapted to ReLU gates (ghf_relu);
// and the slab-scale updates of the outer sweep evaluate residuals from unit-scale
// predictions instead of rescaling the cached residuals (scale_target_stable,
// scale_sources_stable). Every move leaves the SSTL posterior invariant.
#include "sstl_ext.h"

namespace sstl {
namespace ext {

// log g(r)
inline double log_g(double r, int slab) { return slab == 1 ? std::log(r) : 0.5 * r * r; }

// log(1 - exp(x)) for x <= 0
inline double log1mexp(double x) { return x > -M_LN2 ? std::log(-std::expm1(x)) : std::log1p(-std::exp(x)); }

// Univariate slice sampler (stepping out with width 1 and at most 50 steps, then shrinkage)
template <class F>
double slice1(F& logf, double x) {
  double lx = logf(x);
  if (!std::isfinite(lx)) Rcpp::stop("slice update started where the log density is %g", lx);
  double ly = lx + std::log(R::runif(0.0, 1.0));
  double lo = x - R::runif(0.0, 1.0), hi = lo + 1.0;
  int jl = (int) std::floor(50 * R::runif(0.0, 1.0)), kr = 49 - jl;
  while (jl-- > 0 && logf(lo) > ly) lo -= 1.0;
  while (kr-- > 0 && logf(hi) > ly) hi += 1.0;
  while (true) {
    double x_new = lo + R::runif(0.0, 1.0) * (hi - lo);
    if (logf(x_new) > ly) return x_new;
    if (x_new < x) lo = x_new;
    else hi = x_new;
  }
}

// Representable states. A gate meant to be active must be a finite double strictly above its
// threshold, which fails in floating point when the gate excess is below rounding of the threshold
// or overflows. The kernels below target the posterior restricted to states whose gates are
// representable (a set of essentially full probability): a proposal or slice point that would
// need an unrepresentable gate is rejected or lies outside the slice. check_gate then guards
// rebuilt states and stops only if that bookkeeping is wrong.
inline bool gate_ok(double thr, double excess) {
  double a = thr + excess;
  return std::isfinite(a) && a > thr;
}

inline void check_gate(double a, double thr, bool active, const char* where) {
  if (!std::isfinite(a) || (a > thr) != active) {
    Rcpp::stop("%s: a gate meant to be %s its threshold is not representable (gate %.17g, threshold %.17g)",
               where, active ? "above" : "at or below", a, thr);
  }
}

// Working conditional as in cond(), with a N(0, diag(sd^2)) working prior on the block.
inline Cond cond_diag(const arma::mat& P, const arma::vec& l, const arma::uvec& A, const arma::vec& sd) {
  Cond out;
  if (A.n_elem == 0) return out;
  out.R = arma::chol(arma::mat(P.submat(A, A)));
  arma::vec y = arma::solve(arma::trimatl(out.R.t()), arma::vec(l.elem(A)));
  out.mu = arma::solve(arma::trimatu(out.R), y);
  out.logZ = -arma::sum(arma::log(arma::vec(sd.elem(A)))) - arma::sum(arma::log(out.R.diag())) + 0.5 * arma::dot(y, y);
  return out;
}

// Mask distribution of one source block given the target block (as source_prop): each
// coordinate is active with prior probability 1 - Phi(psi), log values lon / loff.
inline SourceProp source_prop_relu(const arma::mat& P, const arma::vec& l, int m, double lon, double loff,
                                   const arma::vec& sd) {
  int nm = 1 << m;
  SourceProp sp;
  sp.lp.set_size(nm);
  sp.cd.resize(nm);
  for (int mk = 0; mk < nm; ++mk) {
    arma::uvec A = mask_idx(mk, m);
    sp.cd[mk] = cond_diag(P, l, A, sd);
    sp.lp(mk) = A.n_elem * lon + (m - A.n_elem) * loff + sp.cd[mk].logZ;
  }
  sp.lse = log_sum_exp(sp.lp);
  sp.lp -= sp.lse;
  return sp;
}

// Density of one coefficient given its radius r, the scale tau and the threshold psi, relative
// to (point mass at 0) + Lebesgue measure, with amplitude amp = tau g(r):
//   Phi(psi) at c = 0, phi(psi + |c| / amp) / (2 amp) otherwise. loff = log Phi(psi).
inline double log_h(double c, double psi, double amp, double loff) {
  if (c == 0.0) return loff;
  return R::dnorm(psi + std::abs(c) / amp, 0.0, 1.0, 1) - std::log(2.0 * amp);
}

// One MH move on the coefficients of block B (joint target/source move if src < 0, source
// src only otherwise) with every radius |w|, scale and threshold held fixed. The proposal is
// the factorized working proposal of factor_move with a per-coordinate Gaussian working prior
// (sd = working_sd times the mean active |c| under the prior) and Bernoulli(1 - Phi(psi)) mask
// priors. On acceptance each block coordinate is rebuilt from its exact conditional: active
// c gives w = sign(c) r and a = psi + |c| / amp; c = 0 gives w = +-r with a fair sign and a
// gate drawn below psi. Returns 0 if rejected, 1 if accepted, 2 if accepted with a support change.
inline int factor_move_relu(const Model& M, const Settings& E, std::vector<Study>& st, const arma::uvec& B, int src) {
  int p = M.p, m = B.n_elem, J = st.size();
  bool joint = src < 0;
  std::vector<int> studies;
  if (joint) for (int j = 0; j < J; ++j) studies.push_back(j);
  else studies.push_back(src);

  std::vector<int> mask(J, 0);
  std::vector<arma::vec> coef(J, arma::zeros(m)), rc(J), amp(J), sdw(J), prec(J);
  std::vector<arma::mat> XB(J);
  std::vector<double> thr(J), lon(J), loff(J);
  for (int j = 0; j < J; ++j) {
    if (!joint && j != 0 && j != src) continue;
    const double* b = st[j].b;
    double tau = std::abs(*st[j].xi);
    thr[j] = threshold_from_a0(b[2 * p], st[j].lambda);
    loff[j] = R::pnorm(thr[j], 0.0, 1.0, 1, 1);
    lon[j] = R::pnorm(thr[j], 0.0, 1.0, 0, 1);
    double excess = std::exp(R::dnorm(thr[j], 0.0, 1.0, 1) - lon[j]) - thr[j]; // E(a - psi | a > psi)
    amp[j].set_size(m);
    for (int t = 0; t < m; ++t) {
      int i = B(t);
      amp[j](t) = tau * std::exp(log_g(std::abs(b[i]), M.slab));
      if (b[p + i] > thr[j]) {
        mask[j] |= 1 << t;
        coef[j](t) = sgn(b[i]) * amp[j](t) * (b[p + i] - thr[j]);
      }
    }
    sdw[j] = E.working_sd * excess * amp[j];
    prec[j] = 1.0 / arma::square(sdw[j]);
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
      S = G[0] + arma::diagmat(prec[0]);
      lu = r[0];
      for (int k = 1; k < J; ++k) {
        working(M, st[k], XB[k], rc[k], u_star + v_star[k], G[k], r[k]);
        P[k] = G[k] + arma::diagmat(prec[k]);
        // With P = G + D: G - G P^{-1} G = D P^{-1} G and r - G P^{-1} r = D P^{-1} r
        S += arma::diagmat(prec[k]) * arma::solve(P[k], G[k]);
        lu += prec[k] % arma::solve(P[k], r[k]);
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
    arma::mat P0 = G[0] + arma::diagmat(prec[0]);
    for (int mk = 0; mk < nm; ++mk) {
      arma::uvec A = mask_idx(mk, m);
      cd0[mk] = cond_diag(S, lu, A, sdw[0]);
      double l = A.n_elem * lon[0] + (m - A.n_elem) * loff[0];
      if (!E.mask_correction) {
        l += cd0[mk].logZ;
      } else {
        l += cond_diag(P0, r[0], A, sdw[0]).logZ;
        arma::vec u_hat = arma::zeros(m);
        if (A.n_elem) u_hat.elem(A) = cd0[mk].mu;
        for (int k = 1; k < J; ++k) {
          SourceProp sp = source_prop_relu(P[k], r[k] - G[k] * u_hat, m, lon[k], loff[k], sdw[k]);
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
      SourceProp sp = source_prop_relu(P[k], r[k] - G[k] * new_coef[0], m, lon[k], loff[k], sdw[k]);
      new_mask[k] = draw_log(sp.lp);
      new_coef[k] = arma::zeros(m);
      arma::uvec A = mask_idx(new_mask[k], m);
      if (A.n_elem) new_coef[k].elem(A) = draw_mix(sp.cd[new_mask[k]], E.t_weight);
      lq_fwd += sp.lp(new_mask[k]) + log_mix(new_coef[k].elem(A), sp.cd[new_mask[k]], E.t_weight);
      // reverse: source densities at the current target block
      SourceProp sr = source_prop_relu(P[k], r[k] - G[k] * coef[0], m, lon[k], loff[k], sdw[k]);
      arma::uvec A_old = mask_idx(mask[k], m);
      lq_rev += sr.lp(mask[k]) + log_mix(coef[k].elem(A_old), sr.cd[mask[k]], E.t_weight);
    }
  } else {
    int k = src;
    const arma::vec& u = coef[0];
    arma::vec v_star = arma::zeros(m), l;
    for (int it = 0; it < n_it; ++it) {
      working(M, st[k], XB[k], rc[k], u + v_star, G[k], r[k]);
      P[k] = G[k] + arma::diagmat(prec[k]);
      l = r[k] - G[k] * u;
      if (it < n_it - 1) v_star = arma::solve(P[k], l);
    }
    SourceProp sp = source_prop_relu(P[k], l, m, lon[k], loff[k], sdw[k]);
    new_mask[k] = draw_log(sp.lp);
    new_coef[k] = arma::zeros(m);
    arma::uvec A = mask_idx(new_mask[k], m), A_old = mask_idx(mask[k], m);
    if (A.n_elem) new_coef[k].elem(A) = draw_mix(sp.cd[new_mask[k]], E.t_weight);
    lq_fwd += sp.lp(new_mask[k]) + log_mix(new_coef[k].elem(A), sp.cd[new_mask[k]], E.t_weight);
    lq_rev += sp.lp(mask[k]) + log_mix(coef[k].elem(A_old), sp.cd[mask[k]], E.t_weight);
    new_coef[0] = coef[0];
  }

  // A proposed active coefficient whose gate a = psi + |c| / amp is not representable has
  // target density zero (see gate_ok)
  for (int j : studies) {
    for (int t = 0; t < m; ++t) {
      if (new_coef[j](t) != 0.0 && !gate_ok(thr[j], std::abs(new_coef[j](t)) / amp[j](t))) return 0;
    }
  }

  // Acceptance ratio: likelihood, coefficient densities h, proposal
  double log_a = lq_rev - lq_fwd;
  std::vector<arma::vec> new_resid(J);
  for (int j : studies) {
    new_resid[j] = rc[j] - XB[j] * (j == 0 ? new_coef[0] : arma::vec(new_coef[0] + new_coef[j]));
    log_a += loglik(M, st[j], new_resid[j]) - loglik(M, st[j], *st[j].resid);
    for (int t = 0; t < m; ++t) {
      log_a += log_h(new_coef[j](t), thr[j], amp[j](t), loff[j]) - log_h(coef[j](t), thr[j], amp[j](t), loff[j]);
    }
  }
  if (!(std::log(R::runif(0.0, 1.0)) < log_a)) return 0;

  // Rebuild every block coordinate of each moved study from its exact conditional
  bool supp_changed = false;
  for (int j : studies) {
    double* b = st[j].b;
    for (int t = 0; t < m; ++t) {
      int i = B(t);
      double c = new_coef[j](t), rad = std::abs(b[i]);
      if (c != 0.0) {
        b[i] = sgn(c) * rad;
        b[p + i] = thr[j] + std::abs(c) / amp[j](t);
      } else {
        b[i] = (R::runif(0.0, 1.0) < 0.5 ? 1.0 : -1.0) * rad;
        b[p + i] = R::qnorm(std::log(R::runif(0.0, 1.0)) + loff[j], 0.0, 1.0, 1, 1);
      }
      check_gate(b[p + i], thr[j], c != 0.0, "ReLU block move");
    }
    supp_changed = supp_changed || new_mask[j] != mask[j];
    st[j].resid->swap(new_resid[j]);
  }
  return supp_changed ? 2 : 1;
}

// Signed exchange of coordinates i and k: (w_i, w_k) -> (s w_k, s w_i) and (a_i, a_k) -> (a_k, a_i)
// in the target and every source at once, with thresholds and scales fixed. The map is an
// involution with unit Jacobian that preserves the latent N(0, 1) priors, so the MH ratio is the
// likelihood ratio. It sends (c_i, c_k) to (s c_k, s c_i) in every population. Returns acceptance.
inline bool exchange(const Model& M, std::vector<Study>& st, arma::uword i, arma::uword k, double sign) {
  int p = M.p, J = st.size();
  std::vector<double> ci(J), ck(J);
  for (int j = 0; j < J; ++j) {
    const double* b = st[j].b;
    double thr = threshold_from_a0(b[2 * p], st[j].lambda), tau = std::abs(*st[j].xi);
    ci[j] = calc_scalar_beta(b[i], b[p + i], thr, tau, M.slab);
    ck[j] = calc_scalar_beta(b[k], b[p + k], thr, tau, M.slab);
  }
  // Linear predictor change (X_i - s X_k)(s c_k - c_i), with sources adding their biases to the target
  std::vector<arma::vec> new_resid(J);
  std::vector<char> moved(J, 0);
  double log_a = 0.0;
  for (int j = 0; j < J; ++j) {
    double d = j == 0 ? sign * ck[0] - ci[0] : sign * (ck[0] + ck[j]) - (ci[0] + ci[j]);
    if (d == 0.0) continue;
    moved[j] = 1;
    new_resid[j] = *st[j].resid - (st[j].X->col(i) - sign * st[j].X->col(k)) * d;
    log_a += loglik(M, st[j], new_resid[j]) - loglik(M, st[j], *st[j].resid);
  }
  if (!(std::log(R::runif(0.0, 1.0)) < log_a)) return false;
  for (int j = 0; j < J; ++j) {
    double* b = st[j].b;
    double wi = b[i];
    b[i] = sign * b[k];
    b[k] = sign * wi;
    std::swap(b[p + i], b[p + k]);
    if (moved[j]) st[j].resid->swap(new_resid[j]);
  }
  return true;
}

// Interweaving at fixed coefficients for one population (likelihood unchanged):
//  1. each active radius r = |w| on z = log r, target z - e^{2z}/2 - log g(e^z) - (psi + |c| / (tau g(e^z)))^2 / 2,
//     then w = sign(w) r and a = psi + |c| / (tau g(r)); inactive weights from N(0, 1);
//  2. t = log tau, with d_i = |c_i| / g(r_i) over the s active coordinates:
//     -e^{2t} / (2 s_tau^2) + (1 - s) t - psi e^{-t} sum d_i - e^{-2t} sum d_i^2 / 2, then a sign for xi;
//  3. psi: -(s + 1) psi^2 / 2 - psi sum (a_i - psi) + (lambda + p - s - 1) log Phi(psi), stored as
//     a0 = Phi^{-1}(Phi(psi)^lambda); active gates keep their excess, inactive gates are drawn below psi.
// Each step is a slice update of the stated conditional, so the coefficients never change.
inline void interweave_relu(const Model& M, Study& st) {
  int p = M.p;
  double* b = st.b;
  double thr = threshold_from_a0(b[2 * p], st.lambda), tau = std::abs(*st.xi);
  std::vector<int> A;
  for (int i = 0; i < p; ++i) {
    if (!(b[p + i] > thr)) {
      b[i] = R::rnorm(0.0, 1.0);
      continue;
    }
    A.push_back(i);
    double r = std::abs(b[i]), log_k = std::log(b[p + i] - thr) + log_g(r, M.slab); // log(|c| / tau)
    auto logf = [&](double z) {
      double rr = std::exp(z), lg = log_g(rr, M.slab), ex = std::exp(log_k - lg), e = thr + ex;
      if (!gate_ok(thr, ex)) return R_NegInf;
      return z - 0.5 * rr * rr - lg - 0.5 * e * e;
    };
    double r_new = std::exp(slice1(logf, std::log(r)));
    b[i] = sgn(b[i]) * r_new;
    b[p + i] = thr + std::exp(log_k - log_g(r_new, M.slab));
    check_gate(b[p + i], thr, true, "ReLU radius interweaving");
  }
  int s = A.size();

  // Slab scale
  std::vector<double> d(s);
  double d1 = 0.0, d2 = 0.0, dmin = R_PosInf, dmax = 0.0;
  for (int q = 0; q < s; ++q) {
    d[q] = tau * (b[p + A[q]] - thr);
    d1 += d[q];
    d2 += d[q] * d[q];
    dmin = std::min(dmin, d[q]);
    dmax = std::max(dmax, d[q]);
  }
  auto logf_t = [&](double t) {
    double e = std::exp(-t);
    if (s > 0 && !(gate_ok(thr, dmin * e) && gate_ok(thr, dmax * e))) return R_NegInf;
    return -0.5 * std::exp(2.0 * t) / (st.s_tau * st.s_tau) + (1.0 - s) * t - thr * e * d1 - 0.5 * e * e * d2;
  };
  double tau_new = std::exp(slice1(logf_t, std::log(tau)));
  for (int q = 0; q < s; ++q) {
    b[p + A[q]] = thr + d[q] / tau_new;
    check_gate(b[p + A[q]], thr, true, "ReLU scale interweaving");
  }
  *st.xi = (R::runif(0.0, 1.0) < 0.5 ? 1.0 : -1.0) * tau_new;

  // Threshold
  std::vector<double> ex(s);
  double e1 = 0.0, emin = R_PosInf, emax = 0.0;
  for (int q = 0; q < s; ++q) {
    ex[q] = b[p + A[q]] - thr;
    e1 += ex[q];
    emin = std::min(emin, ex[q]);
    emax = std::max(emax, ex[q]);
  }
  double lam = st.lambda;
  auto logf_psi = [&](double psi) {
    // the stored threshold is rebuilt from a0, exactly as below
    double lphi = R::pnorm(psi, 0.0, 1.0, 1, 1);
    double t = threshold_from_a0(R::qnorm(lam * lphi, 0.0, 1.0, 1, 1), lam);
    if (!std::isfinite(t) || (s > 0 && !(gate_ok(t, emin) && gate_ok(t, emax)))) return R_NegInf;
    return -0.5 * (s + 1.0) * psi * psi - psi * e1 + (lam + p - s - 1.0) * lphi;
  };
  double psi = slice1(logf_psi, thr);
  b[2 * p] = R::qnorm(lam * R::pnorm(psi, 0.0, 1.0, 1, 1), 0.0, 1.0, 1, 1);
  double thr_new = threshold_from_a0(b[2 * p], lam), loff = R::pnorm(thr_new, 0.0, 1.0, 1, 1);
  int q = 0;
  for (int i = 0; i < p; ++i) {
    bool on = q < s && A[q] == i;
    b[p + i] = on ? thr_new + ex[q++] : R::qnorm(std::log(R::runif(0.0, 1.0)) + loff, 0.0, 1.0, 1, 1);
    check_gate(b[p + i], thr_new, on, "ReLU threshold interweaving");
  }
}

// GHF selector for ReLU gates. For each population the threshold and gates are encoded, given
// the support A (s active), by their conditional probability integral transforms: U_psi for
// 1 - Phi(psi) ~ Beta(s + 1, lambda + p - s), and U_i for the gate's normal law truncated above or
// below psi. G draws an auxiliary hard selector (a0*, g*) from its prior given the same support;
// H is one ESS update of all a0* with g*, U, weights and scales fixed, each proposal decoded into
// physical thresholds and gates for its own support; F refreshes the full physical state from the
// prior given that every support is empty. Uniforms are kept as (log U, log(1 - U)). A proposal
// whose decoded threshold or gates are not representable lies outside the slice (see gate_ok).
inline void ghf_relu(const Model& M, Settings& E, std::vector<Study>& st) {
  int p = M.p, J = st.size();
  E.counts(6) += 1;
  std::vector<std::vector<char>> act(J);
  std::vector<double> lu_psi(J), l1u_psi(J);
  std::vector<arma::vec> lu(J), l1u(J), gstar(J), tau_h(J), coef(J);
  arma::vec astar(J);
  double ll = 0.0;
  for (int j = 0; j < J; ++j) {
    const double* b = st[j].b;
    double lam = st[j].lambda, thr = threshold_from_a0(b[2 * p], lam), tau = std::abs(*st[j].xi);
    act[j] = support(b, p, lam);
    int s = 0;
    for (int i = 0; i < p; ++i) s += act[j][i];
    double sa = s + 1.0, sb = lam + p - s;
    if (thr > 0.0) { // omega = 1 - Phi(psi) is the smaller tail
      double om = R::pnorm(thr, 0.0, 1.0, 0, 0);
      lu_psi[j] = R::pbeta(om, sa, sb, 1, 1);
      l1u_psi[j] = R::pbeta(om, sa, sb, 0, 1);
    } else { // P(omega <= x) = P(1 - omega >= 1 - x) with 1 - omega ~ Beta(lambda + p - s, s + 1)
      double om1 = R::pnorm(thr, 0.0, 1.0, 1, 0);
      lu_psi[j] = R::pbeta(om1, sb, sa, 0, 1);
      l1u_psi[j] = R::pbeta(om1, sb, sa, 1, 1);
    }
    double lup = R::pnorm(thr, 0.0, 1.0, 0, 1), llo = R::pnorm(thr, 0.0, 1.0, 1, 1);
    lu[j].set_size(p);
    l1u[j].set_size(p);
    tau_h[j].set_size(p);
    coef[j].set_size(p);
    for (int i = 0; i < p; ++i) {
      double a = b[p + i];
      if (act[j][i]) {
        l1u[j](i) = R::pnorm(a, 0.0, 1.0, 0, 1) - lup;
        lu[j](i) = log1mexp(l1u[j](i));
      } else {
        lu[j](i) = R::pnorm(a, 0.0, 1.0, 1, 1) - llo;
        l1u[j](i) = log1mexp(lu[j](i));
      }
      tau_h[j](i) = tau * sgn(b[i]) * std::exp(log_g(std::abs(b[i]), M.slab));
      coef[j](i) = act[j][i] ? tau_h[j](i) * (a - thr) : 0.0;
    }
    // G: auxiliary selector from its prior given the current support
    arma::vec aux(2 * p + 1);
    gate_refresh(aux.memptr(), act[j], p, lam);
    gstar[j] = aux.subvec(p, 2 * p - 1);
    astar(j) = aux(2 * p);
    ll += ghf_ll(M, E, st[j], *st[j].resid);
  }

  // H: one ESS update of a0* (prior N(0, I))
  arma::vec nu(J);
  for (int j = 0; j < J; ++j) nu(j) = R::rnorm(0.0, 1.0);
  double log_y = ll + std::log(R::runif(0.0, 1.0));
  double theta = R::runif(0.0, 2.0 * M_PI), theta_min = theta - 2.0 * M_PI, theta_max = theta;
  std::vector<std::vector<char>> act_p(J, std::vector<char>(p));
  std::vector<arma::vec> gate_p(J), coef_p(J), resid_p(J);
  std::vector<double> a0_p(J);
  std::vector<char> changed(J);
  for (int n_s = 0; n_s < M.S_max; ++n_s) {
    arma::vec a_prop = astar * std::cos(theta) + nu * std::sin(theta);
    bool any = false, ok = true;
    for (int j = 0; j < J; ++j) {
      double vt = threshold_from_a0(a_prop(j), st[j].lambda);
      int s = 0;
      for (int i = 0; i < p; ++i) {
        act_p[j][i] = gstar[j](i) > vt;
        s += act_p[j][i];
      }
      changed[j] = act_p[j] != act[j];
      any = any || changed[j];
      if (!changed[j]) continue;
      // Decode the threshold and gates for the proposed support
      double lam = st[j].lambda, sa = s + 1.0, sb = lam + p - s;
      double om = R::qbeta(lu_psi[j], sa, sb, 1, 1);
      double psi = om < 0.5 ? R::qnorm(om, 0.0, 1.0, 0, 0) : R::qnorm(R::qbeta(l1u_psi[j], sb, sa, 1, 1), 0.0, 1.0, 1, 0);
      a0_p[j] = R::qnorm(lam * R::pnorm(psi, 0.0, 1.0, 1, 1), 0.0, 1.0, 1, 1);
      double thr = threshold_from_a0(a0_p[j], lam);
      ok = ok && std::isfinite(thr);
      double lup = R::pnorm(thr, 0.0, 1.0, 0, 1), llo = R::pnorm(thr, 0.0, 1.0, 1, 1);
      gate_p[j].set_size(p);
      coef_p[j].set_size(p);
      for (int i = 0; ok && i < p; ++i) {
        double a = act_p[j][i] ? R::qnorm(l1u[j](i) + lup, 0.0, 1.0, 0, 1) : R::qnorm(lu[j](i) + llo, 0.0, 1.0, 1, 1);
        ok = std::isfinite(a) && (a > thr) == (bool) act_p[j][i];
        gate_p[j](i) = a;
        coef_p[j](i) = act_p[j][i] ? tau_h[j](i) * (a - thr) : 0.0;
      }
    }
    if (!any) break; // every support unchanged: the decoded state is the current state
    double ll_prop = R_NegInf;
    if (ok) {
      ll_prop = 0.0;
      arma::vec d0 = changed[0] ? arma::vec(coef_p[0] - coef[0]) : arma::zeros(p);
      for (int j = 0; j < J; ++j) {
        arma::vec d = j == 0 ? d0 : (changed[j] ? arma::vec(d0 + coef_p[j] - coef[j]) : d0);
        arma::uvec nz = arma::find(d != 0.0);
        resid_p[j] = *st[j].resid - st[j].X->cols(nz) * d.elem(nz);
        ll_prop += ghf_ll(M, E, st[j], resid_p[j]);
      }
    }
    if (ll_prop > log_y) {
      for (int j = 0; j < J; ++j) {
        if (changed[j]) {
          double* b = st[j].b;
          for (int i = 0; i < p; ++i) b[p + i] = gate_p[j](i);
          b[2 * p] = a0_p[j];
          act[j] = act_p[j];
        }
        st[j].resid->swap(resid_p[j]);
      }
      E.counts(7) += 1;
      break;
    }
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = R::runif(theta_min, theta_max);
  }

  // F: when every support is empty, draw the full physical state from the prior given that event
  for (int j = 0; j < J; ++j) {
    if (std::any_of(act[j].begin(), act[j].end(), [](char a) { return a; })) return;
  }
  E.counts(8) += 1;
  for (int j = 0; j < J; ++j) {
    for (int i = 0; i < p; ++i) st[j].b[i] = R::rnorm(0.0, 1.0);
    *st[j].xi = R::rnorm(0.0, st[j].s_tau);
    gate_refresh(st[j].b, std::vector<char>(p, 0), p, st[j].lambda);
  }
}

// The legacy ESS update of the target slab-scale shadow xi (prior N(0, sd_0^2), at most 20
// proposals before keeping the current value), with every residual evaluated as
// Y_adj - tau' z from the unit-scale prediction z instead of Y_adj - (Y_adj - r) tau' / tau,
// which amplifies any rounding in the cached residual r by tau' / tau.
template <class Chain, class LLT, class LLS>
double scale_target_stable(Chain& ch, double sd_0, LLT& ll_T, LLS& ll_S) {
  int S = ch.S();
  arma::vec c1 = ch.coef(ch.bt, ch.lambda_T, 1.0);
  arma::uvec nz = arma::find(c1 != 0.0);
  arma::vec cz = c1.elem(nz);
  arma::vec zT = ch.X_T.cols(nz) * cz, YT = ch.Y_T - ch.b0_T;
  std::vector<arma::vec> zS(S), YS(S);
  for (int s = 0; s < S; ++s) {
    zS[s] = ch.X_s[s].cols(nz) * cz;
    arma::vec cs = ch.coef(ch.bs.col(s), ch.lambda_s(s), std::abs(ch.xi_s(s)));
    arma::uvec nzs = arma::find(cs != 0.0);
    YS[s] = ch.Y_s[s] - ch.b0_s(s) - ch.X_s[s].cols(nzs) * cs.elem(nzs);
  }
  double xi = ch.xi;
  ch.resid_T = YT - std::abs(xi) * zT;
  double ll = ll_T(ch.resid_T);
  for (int s = 0; s < S; ++s) {
    ch.resid_S[s] = YS[s] - std::abs(xi) * zS[s];
    ll += ll_S(s, ch.resid_S[s]);
  }
  double nu = R::rnorm(0.0, sd_0);
  double log_y = ll + std::log(R::runif(0.0, 1.0));
  double theta = R::runif(0.0, 2.0 * M_PI), theta_min = theta - 2.0 * M_PI, theta_max = theta;
  arma::vec rT;
  std::vector<arma::vec> rS(S);
  for (int iter = 0; iter < 20; ++iter) {
    double xi_prop = xi * std::cos(theta) + nu * std::sin(theta), tau_prop = std::abs(xi_prop);
    rT = YT - tau_prop * zT;
    double ll_prop = ll_T(rT);
    for (int s = 0; s < S; ++s) {
      rS[s] = YS[s] - tau_prop * zS[s];
      ll_prop += ll_S(s, rS[s]);
    }
    if (ll_prop > log_y) {
      ch.resid_T.swap(rT);
      for (int s = 0; s < S; ++s) ch.resid_S[s].swap(rS[s]);
      return xi_prop;
    }
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = R::runif(theta_min, theta_max);
  }
  return xi;
}

// The legacy joint ESS update of the source slab-scale shadows xi_s (prior N(0, sd_0^2) each, at
// most 21 proposals), with residuals from unit-scale bias predictions as in scale_target_stable.
template <class Chain, class LLS>
arma::vec scale_sources_stable(Chain& ch, double sd_0, LLS& ll_S) {
  int S = ch.S();
  std::vector<arma::vec> zS(S), YS(S);
  arma::vec xi = ch.xi_s;
  double ll = 0.0;
  arma::uvec nzT = arma::find(ch.beta_T != 0.0);
  arma::vec bT = ch.beta_T.elem(nzT);
  for (int s = 0; s < S; ++s) {
    arma::vec c1 = ch.coef(ch.bs.col(s), ch.lambda_s(s), 1.0);
    arma::uvec nz = arma::find(c1 != 0.0);
    zS[s] = ch.X_s[s].cols(nz) * c1.elem(nz);
    YS[s] = ch.Y_s[s] - ch.b0_s(s) - ch.X_s[s].cols(nzT) * bT;
    ch.resid_S[s] = YS[s] - std::abs(xi(s)) * zS[s];
    ll += ll_S(s, ch.resid_S[s]);
  }
  arma::vec nu(S);
  for (int s = 0; s < S; ++s) nu(s) = R::rnorm(0.0, sd_0);
  double log_y = ll + std::log(R::runif(0.0, 1.0));
  double theta = R::runif(0.0, 2.0 * M_PI), theta_min = theta - 2.0 * M_PI, theta_max = theta;
  std::vector<arma::vec> rS(S);
  for (int iter = 0; iter <= 20; ++iter) {
    arma::vec xi_prop = xi * std::cos(theta) + nu * std::sin(theta);
    double ll_prop = 0.0;
    for (int s = 0; s < S; ++s) {
      rS[s] = YS[s] - std::abs(xi_prop(s)) * zS[s];
      ll_prop += ll_S(s, rS[s]);
    }
    if (ll_prop > log_y) {
      for (int s = 0; s < S; ++s) ch.resid_S[s].swap(rS[s]);
      return xi_prop;
    }
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = R::runif(theta_min, theta_max);
  }
  return xi;
}

} // namespace ext
} // namespace sstl
