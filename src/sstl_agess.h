#pragma once
// Adaptive generalized elliptical slice sampling (AGESS; Marco & Tokdar, arXiv:2605.21659) for the latent
// coefficient blocks, the default sampler. Settings follow AdaptEllipticalSliceSampler.jl:
// - t auxiliary with nu = 6;
// - the non-adaptive kernel (centred on the prior) with probability eps = 0.05;
// - for blocks of dimension >= 10, one-dimensional sweeps with probability ssp = 0.05, and always during
//   the first floor(0.5 N) * ssp iterations;
// - each block's background mean and covariance are updated every iteration with weight
//   i^-max(2/3, (P^(1/3) - 1) / P^(1/3)) and copied to the active kernel on the AirMCMC schedule
//   N_j = sum floor(l^0.5).
// Adaptation runs over the whole chain, EB included.
#include "sstl_helpers.h"

namespace sstl {

const double AGESS_NU = 6.0, AGESS_EPS = 0.05, AGESS_SSP = 0.05, AGESS_AIR_BETA = 0.5;

// Below this |w| every slab has |H(w)| < 1e72, so with tau < 1e200 the product tau * H(w) * T(a - thresh)
// is a zero whenever a <= thresh (T is 0 there for every slab). Such coefficients are set to 0.0 without
// evaluating H. The sign of a zero never matters: coefficients enter only through differences tested
// against 0. Residual updates keep the statement forms they were verified with: under gcc's
// -ffp-contract=fast the form decides which multiplies are fused into adds.
const double AGESS_W_SAFE = 18.0;

// Active (mu, L = lower Cholesky factor) and background (mu_bg, S_bg) adaptive parameters of one block,
// its non-adaptive kernel (mu0 = 0, L0 = diag(sd)) and the prior sd of its coordinates.
struct Adapt {
  arma::vec mu, mu_bg, mu0, sd;
  arma::mat L, S_bg, L0;
  double w_const;
};

inline Adapt new_adapt(const arma::vec& sd) {
  double c = std::cbrt((double) sd.n_elem);
  return Adapt{arma::zeros(sd.n_elem), arma::zeros(sd.n_elem), arma::zeros(sd.n_elem), sd,
               arma::diagmat(sd), arma::diagmat(arma::square(sd)), arma::diagmat(sd),
               std::max(2.0 / 3.0, (c - 1) / c)};
}

// Iteration counter of the chain (AdaptEllipticalSliceSampler.jl starts at 2), the background weight
// for blocks with w_const = 2/3, and whether one-dimensional sweeps are forced.
struct AgessStep {
  double iter, w23;
  bool one_d_phase;
};

// AirMCMC: copy the background parameters to the active kernel.
inline void agess_copy_background(Adapt& A) {
  A.mu = A.mu_bg;
  int P = A.mu.n_elem;
  for (int j = 0; j < P; ++j) { // lower Cholesky factor of S_bg
    double s = A.S_bg(j, j);
    for (int l = 0; l < j; ++l) s -= A.L(j, l) * A.L(j, l);
    if (s <= 0) Rcpp::stop("Adapted covariance is not positive definite.");
    A.L(j, j) = std::sqrt(s);
    for (int r = j + 1; r < P; ++r) {
      double t = A.S_bg(r, j);
      for (int l = 0; l < j; ++l) t -= A.L(r, l) * A.L(j, l);
      A.L(r, j) = t / A.L(j, j);
    }
  }
}

// One AGESS transition (Algorithm 2 of the paper) of coordinates co[0..d-1] of the block vector f, whose
// prior is N(0, diag(sd^2)): slice sampling on L* = L * pi0 / t_nu(mv, L L') along the ellipse through the
// current point and an auxiliary point drawn from its t conditional. L has leading dimension ld.
// block_ll(changed) returns the log-likelihood at the current f and sets changed = false when no
// coefficient changes; the likelihood terms then cancel in the slice test, also from a zero-likelihood
// state. accept(changed, ll) commits a proposal and f keeps it; otherwise f is restored. scratch holds 4 d
// doubles. Returns the number of slice evaluations.
template <class LL, class Accept>
int agess_transition(double* f, const int* co, int d, const double* mv, const double* L, int ld, const double* sd,
                     int S_max, const double& current_ll, double* scratch, LL&& block_ll, Accept&& accept) {
  double nu = AGESS_NU;
  double *x = scratch, *r = scratch + d, *z = scratch + 2 * d, *xp = scratch + 3 * d;
  for (int j = 0; j < d; ++j) x[j] = f[co[j]];
  // q = (v - mv)' Sigma^-1 (v - mv) by forward substitution
  auto quad = [&](const double* v) {
    double q = 0.0;
    for (int i = 0; i < d; ++i) {
      double s = v[i] - mv[i];
      for (int j = 0; j < i; ++j) s -= L[i + j * ld] * r[j];
      r[i] = s / L[i + i * ld];
      q += r[i] * r[i];
    }
    return q;
  };
  auto log_pi0 = [&](const double* v) {
    double s = 0.0;
    for (int j = 0; j < d; ++j) {
      double t = v[j] / sd[co[j]];
      s += t * t;
    }
    return -0.5 * s;
  };
  auto log_aux = [&](double qv) { return -0.5 * (nu + d) * std::log1p(qv / nu); };

  // Z | X = x from the joint t_{2d}(nu) with scale I_2 (x) Sigma: t_{nu+d} with scale (nu + q)/(nu + d) Sigma
  double q = quad(x);
  for (int j = 0; j < d; ++j) z[j] = R::norm_rand();
  double g = R::rgamma(0.5 * (nu + d), 2.0 / (nu + d));
  double scale = std::sqrt((nu + q) / (nu + d) / g);
  for (int i = d - 1; i >= 0; --i) {
    double s = 0.0;
    for (int j = 0; j <= i; ++j) s += L[i + j * ld] * z[j];
    z[i] = mv[i] + s * scale;
  }

  double ratio = log_pi0(x) - log_aux(q);
  double log_u = std::log(R::unif_rand());
  double theta = R::runif(0, 2 * M_PI);
  double theta_min = theta - 2 * M_PI;
  double theta_max = theta;

  int n_s = 0;
  while (n_s < S_max) {
    n_s++;
    double ct = std::cos(theta), st = std::sin(theta);
    for (int j = 0; j < d; ++j) {
      xp[j] = (x[j] - mv[j]) * ct + (z[j] - mv[j]) * st + mv[j];
      f[co[j]] = xp[j];
    }
    bool changed;
    double ll_prop = block_ll(changed);
    double ratio_prop = log_pi0(xp) - log_aux(quad(xp));
    bool in_slice = changed ? ll_prop + ratio_prop > current_ll + ratio + log_u : ratio_prop > ratio + log_u;
    if (in_slice) {
      accept(changed, ll_prop);
      return n_s;
    }
    if (theta < 0) theta_min = theta;
    else theta_max = theta;
    theta = R::runif(theta_min, theta_max);
  }
  for (int j = 0; j < d; ++j) f[co[j]] = x[j];
  return n_s;
}

// Kernel choice of AdaptEllipticalSliceSampler.jl for the block f of dimension P = A.mu.n_elem (co holds
// 0..P-1), then the background update of its mean and covariance. scratch holds 4 P doubles.
template <class LL, class Accept>
int agess_block(double* f, const int* co, Adapt& A, const AgessStep& it, int S_max, const double& current_ll,
                double* scratch, LL&& block_ll, Accept&& accept) {
  int P = A.mu.n_elem;
  auto full = [&](const arma::vec& mv, const arma::mat& L) {
    return agess_transition(f, co, P, mv.memptr(), L.memptr(), P, A.sd.memptr(), S_max, current_ll, scratch,
                            block_ll, accept);
  };
  auto one_d_sweep = [&]() {
    int n = 0;
    for (int j = 0; j < P; ++j) {
      n += agess_transition(f, co + j, 1, A.mu.memptr() + j, &A.L(j, j), 1, A.sd.memptr(), S_max, current_ll,
                            scratch, block_ll, accept);
    }
    return n;
  };
  int n_s;
  if (P >= 10) {
    if (it.one_d_phase) n_s = one_d_sweep();
    else if (R::unif_rand() > AGESS_EPS + AGESS_SSP) n_s = full(A.mu, A.L);
    else if (R::unif_rand() < AGESS_SSP / (AGESS_EPS + AGESS_SSP)) n_s = one_d_sweep();
    else n_s = full(A.mu0, A.L0);
  } else {
    n_s = R::unif_rand() > AGESS_EPS ? full(A.mu, A.L) : full(A.mu0, A.L0);
  }

  double w = A.w_const == 2.0 / 3.0 ? it.w23 : std::pow(it.iter, -A.w_const);
  double* r = scratch;
  for (int c = 0; c < P; ++c) r[c] = f[c] - A.mu_bg(c);
  for (int j = 0; j < P; ++j) {
    for (int i = 0; i < P; ++i) A.S_bg(i, j) = (1 - w) * A.S_bg(i, j) + w * r[i] * r[j];
  }
  for (int c = 0; c < P; ++c) A.mu_bg(c) = (1 - w) * A.mu_bg(c) + w * f[c];
  return n_s;
}

// AGESS counterpart of the target block updates: blocks (w_j..., a_j...) for each block of columns in id,
// then a0. Used by transfer models (update_target_aft/general; the likelihood is the target's plus every
// source's, since beta_T enters all of them) and by target-only models (update_blocks_aft/general, no
// sources). sd holds the prior sd of every latent coordinate. ll_T(resid) and ll_S(s, resid) give the
// log-likelihoods.
template <class LLT, class LLS>
void agess_update_target(arma::vec& bt, arma::vec& resid_T, std::vector<arma::vec>& resid_S,
                         const arma::mat& X_T, const std::vector<arma::mat>& X_s,
                         const std::vector<arma::uvec>& id, std::vector<Adapt>& ad, const AgessStep& it,
                         double lambda_T, double tau, int S_max, int slab, LLT&& ll_T, LLS&& ll_S,
                         arma::vec& N_t) {
  int p = X_T.n_cols, n_T = X_T.n_rows, S = X_s.size(), K = id.size();
  double thresh = threshold_from_a0(bt(2 * p), lambda_T);
  arma::vec beta = get_beta(bt.subvec(0, p - 1), bt.subvec(p, 2 * p - 1), tau, bt(2 * p), lambda_T, slab);
  double current_ll = ll_T(resid_T);
  for (int s = 0; s < S; ++s) current_ll += ll_S(s, resid_S[s]);
  N_t.zeros(K);

  int P_max = 0;
  for (const arma::uvec& idx : id) P_max = std::max(P_max, (int) idx.n_elem);
  arma::vec resid_T_prop(n_T), beta_prop(p), d_beta(p), f(P_max), scratch(4 * P_max);
  std::vector<int> co(P_max);
  for (int c = 0; c < P_max; ++c) co[c] = c;
  std::vector<arma::vec> resid_S_prop(S);
  for (int s = 0; s < S; ++s) resid_S_prop[s].set_size(X_s[s].n_rows);
  bool tau_ok = tau < 1e200;

  // Residuals at the proposed coefficients: subtract the changed columns times their change
  auto shift = [&](const arma::vec& resid, arma::vec& out, const arma::mat& X, int start, int cols) {
    int n = X.n_rows;
    std::copy(resid.begin(), resid.end(), out.begin());
    for (int c = 0; c < cols; ++c) {
      if (d_beta(c) == 0.0) continue;
      const double* xc = X.colptr(start + c);
      double dc = d_beta(c);
      for (int i = 0; i < n; ++i) out(i) -= xc[i] * dc;
    }
  };

  for (int k = 0; k < K; ++k) {
    const arma::uvec& idx = id[k];
    bool a0_block = k == K - 1;
    int m = idx.n_elem / 2; // columns of a (w, a) block
    int start = a0_block ? 0 : idx(0), cols = a0_block ? p : m;
    for (arma::uword c = 0; c < idx.n_elem; ++c) f(c) = bt(idx(c));
    double thresh_prop = thresh;

    auto block_ll = [&](bool& changed) {
      changed = false;
      if (a0_block) thresh_prop = threshold_from_a0(f(0), lambda_T);
      for (int c = 0; c < cols; ++c) {
        double wc = a0_block ? bt(c) : f(c);
        double ac = a0_block ? bt(p + c) : f(m + c);
        beta_prop(c) = !(ac > thresh_prop) && tau_ok && std::abs(wc) < AGESS_W_SAFE
          ? 0.0 : calc_scalar_beta(wc, ac, thresh_prop, tau, slab);
        d_beta(c) = beta_prop(c) - beta(start + c);
        if (d_beta(c) != 0.0) changed = true;
      }
      if (!changed) return current_ll;
      shift(resid_T, resid_T_prop, X_T, start, cols);
      double ll = ll_T(resid_T_prop);
      for (int s = 0; s < S; ++s) {
        shift(resid_S[s], resid_S_prop[s], X_s[s], start, cols);
        ll += ll_S(s, resid_S_prop[s]);
      }
      return ll;
    };
    auto accept = [&](bool changed, double ll) {
      thresh = thresh_prop;
      if (!changed) return;
      for (int c = 0; c < cols; ++c) beta(start + c) = beta_prop(c);
      resid_T.swap(resid_T_prop);
      for (int s = 0; s < S; ++s) resid_S[s].swap(resid_S_prop[s]);
      current_ll = ll;
    };

    N_t(k) = agess_block(f.memptr(), co.data(), ad[k], it, S_max, current_ll, scratch.memptr(), block_ll, accept);
    for (arma::uword c = 0; c < idx.n_elem; ++c) bt(idx(c)) = f(c);
  }
}

// AGESS counterpart of update_source_joint_aft/general: each source's latent coordinates (w, a, then a0)
// one at a time, with that source's likelihood only.
template <class LLS>
void agess_update_sources(arma::mat& bs, std::vector<arma::vec>& resid_S, const std::vector<arma::mat>& X_s,
                          std::vector<std::vector<Adapt>>& ad, const AgessStep& it,
                          const arma::vec& lambda_S, const arma::vec& tau_S, int S_max, int slab, LLS&& ll_S,
                          arma::vec& N_s) {
  int p = (bs.n_rows - 1) / 2, S = bs.n_cols;
  N_s.zeros(2 * p + 1);
  arma::vec bias_old(p), d_bias(p), scratch(4);
  const int co[1] = {0};

  for (int s = 0; s < S; ++s) {
    arma::vec& resid = resid_S[s];
    const arma::mat& X = X_s[s];
    int n = X.n_rows;
    arma::vec resid_prop(n);
    double tau_s = tau_S(s), lambda_s = lambda_S(s);
    double ll_curr = ll_S(s, resid);
    double threshold = threshold_from_a0(bs(2 * p, s), lambda_s);
    bool tau_ok = tau_s < 1e200;

    for (int j = 0; j <= 2 * p; ++j) {
      bool is_a0 = j == 2 * p;
      int k = j % p;
      double beta_old = 0.0;
      if (is_a0) {
        for (int i = 0; i < p; ++i) {
          bias_old(i) = !(bs(i + p, s) > threshold) && tau_ok && std::abs(bs(i, s)) < AGESS_W_SAFE
            ? 0.0 : calc_scalar_beta(bs(i, s), bs(i + p, s), threshold, tau_s, slab);
        }
      } else if (bs(k + p, s) > threshold || !tau_ok || !(std::abs(bs(k, s)) < AGESS_W_SAFE)) {
        beta_old = calc_scalar_beta(bs(k, s), bs(k + p, s), threshold, tau_s, slab);
      }
      double x[1] = {bs(j, s)};

      // resid_prop is filled from resid only for a changed proposal
      auto block_ll = [&](bool& changed) {
        if (is_a0) {
          double thresh_prop = threshold_from_a0(x[0], lambda_s);
          changed = false;
          for (int i = 0; i < p; ++i) {
            if (!(bs(i + p, s) > thresh_prop) && tau_ok && std::abs(bs(i, s)) < AGESS_W_SAFE) {
              if (bias_old(i) == 0.0) continue;
              d_bias(i) = 0.0 - bias_old(i);
            } else {
              d_bias(i) = calc_scalar_beta(bs(i, s), bs(i + p, s), thresh_prop, tau_s, slab) - bias_old(i);
              if (d_bias(i) == 0.0) continue;
            }
            if (!changed) std::copy(resid.begin(), resid.end(), resid_prop.begin());
            changed = true;
            const double* xc = X.colptr(i);
            for (int r = 0; r < n; ++r) resid_prop(r) -= xc[r] * d_bias(i);
          }
        } else {
          double wv = j < p ? x[0] : bs(k, s), av = j < p ? bs(k + p, s) : x[0];
          double beta_new = !(av > threshold) && tau_ok && std::abs(wv) < AGESS_W_SAFE ? 0.0
            : j < p ? calc_scalar_beta(x[0], bs(k + p, s), threshold, tau_s, slab)
                    : calc_scalar_beta(bs(k, s), x[0], threshold, tau_s, slab);
          double dv = beta_new - beta_old;
          changed = dv != 0.0;
          const double* xc = X.colptr(k);
          if (changed) {
            std::copy(resid.begin(), resid.end(), resid_prop.begin());
            for (int r = 0; r < n; ++r) resid_prop(r) -= xc[r] * dv;
          }
        }
        return changed ? ll_S(s, resid_prop) : ll_curr;
      };
      auto accept = [&](bool changed, double ll) {
        if (!changed) return;
        resid.swap(resid_prop);
        ll_curr = ll;
      };

      N_s(j) += agess_block(x, co, ad[s][j], it, S_max, ll_curr, scratch.memptr(), block_ll, accept);
      bs(j, s) = x[0];
    }
  }
}

// AGESS counterpart of update_target_group_aft/general: per group, the gate a_g and then each weight w_j
// of the group, one coordinate at a time; a0 last. ad holds one state per latent coordinate.
template <class LLT, class LLS>
void agess_update_target_group(arma::vec& bt, arma::vec& resid_T, std::vector<arma::vec>& resid_S,
                               const arma::mat& X_T, const std::vector<arma::mat>& X_s,
                               const std::vector<arma::uvec>& group_cols, const Rcpp::IntegerVector& group_map,
                               std::vector<Adapt>& ad, const AgessStep& it, double lambda_T, double tau,
                               int S_max, int slab, LLT&& ll_T, LLS&& ll_S, arma::vec& N_t) {
  int p = X_T.n_cols, S = X_s.size(), G = group_cols.size();
  arma::vec w_T = bt.subvec(0, p - 1);
  arma::vec a_T = bt.subvec(p, p + G - 1);
  double threshold_T = threshold_from_a0(bt(p + G), lambda_T);
  arma::vec beta_T = get_beta_group(w_T, a_T, group_map, tau, bt(p + G), lambda_T, slab);
  double current_ll = ll_T(resid_T);
  for (int s = 0; s < S; ++s) current_ll += ll_S(s, resid_S[s]);
  N_t.zeros(p + G + 1);
  arma::vec resid_T_prop, scratch(4);
  std::vector<arma::vec> resid_S_prop(S);
  const int co[1] = {0};

  // Proposed residuals for a coefficient change delta on the columns cols, and their log-likelihood
  auto propose = [&](const arma::uvec& cols, const arma::vec& delta) {
    resid_T_prop = resid_T - X_T.cols(cols) * delta;
    double ll = ll_T(resid_T_prop);
    for (int s = 0; s < S; ++s) {
      resid_S_prop[s] = resid_S[s] - X_s[s].cols(cols) * delta;
      ll += ll_S(s, resid_S_prop[s]);
    }
    return ll;
  };
  auto commit = [&](double ll) {
    resid_T.swap(resid_T_prop);
    for (int s = 0; s < S; ++s) resid_S[s].swap(resid_S_prop[s]);
    current_ll = ll;
  };

  for (int g = 0; g < G; ++g) {
    const arma::uvec& cols = group_cols[g];
    int idx_a = p + g;

    // Group gate first
    double x[1] = {bt(idx_a)};
    double act_old = activation_scalar(x[0], threshold_T, slab);
    arma::vec h_group = slab_weight_vec(w_T.elem(cols), slab);
    arma::vec delta;
    auto gate_ll = [&](bool& changed) {
      double act_new = activation_scalar(x[0], threshold_T, slab);
      changed = act_new != act_old;
      if (!changed) return current_ll;
      delta = tau * h_group * (act_new - act_old);
      return propose(cols, delta);
    };
    auto gate_accept = [&](bool changed, double ll) {
      if (!changed) return;
      beta_T.elem(cols) += delta;
      commit(ll);
    };
    N_t(idx_a) = agess_block(x, co, ad[idx_a], it, S_max, current_ll, scratch.memptr(), gate_ll, gate_accept);
    bt(idx_a) = x[0];
    a_T(g) = x[0];

    // Then the group's weights
    for (arma::uword pos = 0; pos < cols.n_elem; ++pos) {
      arma::uword j = cols(pos);
      double xw[1] = {bt(j)};
      double beta_old = calc_scalar_beta(xw[0], a_T(g), threshold_T, tau, slab);
      double beta_new = beta_old;
      auto w_ll = [&](bool& changed) {
        beta_new = calc_scalar_beta(xw[0], a_T(g), threshold_T, tau, slab);
        changed = beta_new - beta_old != 0.0;
        if (!changed) return current_ll;
        return propose(arma::uvec{j}, arma::vec{beta_new - beta_old});
      };
      auto w_accept = [&](bool changed, double ll) {
        beta_T(j) = beta_new;
        if (changed) commit(ll);
      };
      N_t(j) = agess_block(xw, co, ad[j], it, S_max, current_ll, scratch.memptr(), w_ll, w_accept);
      bt(j) = xw[0];
      w_T(j) = xw[0];
    }
  }

  // Global threshold last
  int idx_a0 = p + G;
  double x[1] = {bt(idx_a0)};
  arma::vec tau_h = tau * slab_weight_vec(w_T, slab);
  arma::vec beta_T_prop(p), delta;
  arma::uvec nz;
  auto a0_ll = [&](bool& changed) {
    double threshold_prop = threshold_from_a0(x[0], lambda_T);
    for (int j = 0; j < p; ++j) beta_T_prop(j) = tau_h(j) * activation_scalar(a_T(group_map[j] - 1), threshold_prop, slab);
    delta = beta_T_prop - beta_T;
    nz = find(delta != 0.0);
    changed = nz.n_elem > 0;
    if (!changed) return current_ll;
    return propose(nz, arma::vec(delta.elem(nz)));
  };
  auto a0_accept = [&](bool changed, double ll) {
    beta_T = beta_T_prop;
    if (changed) commit(ll);
  };
  N_t(idx_a0) = agess_block(x, co, ad[idx_a0], it, S_max, current_ll, scratch.memptr(), a0_ll, a0_accept);
  bt(idx_a0) = x[0];
}

// AGESS counterpart of update_source_joint_group_aft/general: per source, each gate, its group's weights,
// and a0, one coordinate at a time with that source's likelihood.
template <class LLS>
void agess_update_sources_group(arma::mat& bs, std::vector<arma::vec>& resid_S, const std::vector<arma::mat>& X_s,
                                const std::vector<arma::uvec>& group_cols, const Rcpp::IntegerVector& group_map,
                                std::vector<std::vector<Adapt>>& ad, const AgessStep& it,
                                const arma::vec& lambda_S, const arma::vec& tau_S, int S_max, int slab, LLS&& ll_S,
                                arma::vec& N_s) {
  int p = group_map.size(), S = bs.n_cols, G = group_cols.size(), idx_a0 = p + G;
  N_s.zeros(p + G + 1);
  arma::vec resid_prop, scratch(4);
  const int co[1] = {0};

  for (int s = 0; s < S; ++s) {
    arma::vec& resid = resid_S[s];
    const arma::mat& X = X_s[s];
    double tau_s = tau_S(s);
    double ll_curr = ll_S(s, resid);
    double thresh_s = threshold_from_a0(bs(idx_a0, s), lambda_S(s));
    auto commit = [&](double ll) {
      resid.swap(resid_prop);
      ll_curr = ll;
    };

    for (int g = 0; g < G; ++g) {
      const arma::uvec& cols = group_cols[g];
      int idx_a = p + g;

      double x[1] = {bs(idx_a, s)};
      double act_old = activation_scalar(x[0], thresh_s, slab);
      arma::vec w_group(cols.n_elem);
      for (arma::uword pos = 0; pos < cols.n_elem; ++pos) w_group(pos) = bs(cols(pos), s);
      arma::vec h_group = slab_weight_vec(w_group, slab);
      auto gate_ll = [&](bool& changed) {
        double act_new = activation_scalar(x[0], thresh_s, slab);
        changed = act_new != act_old;
        if (!changed) return ll_curr;
        resid_prop = resid - X.cols(cols) * (tau_s * h_group * (act_new - act_old));
        return ll_S(s, resid_prop);
      };
      auto gate_accept = [&](bool changed, double ll) { if (changed) commit(ll); };
      N_s(idx_a) += agess_block(x, co, ad[s][idx_a], it, S_max, ll_curr, scratch.memptr(), gate_ll, gate_accept);
      bs(idx_a, s) = x[0];

      double a_val = bs(idx_a, s);
      for (arma::uword pos = 0; pos < cols.n_elem; ++pos) {
        arma::uword j = cols(pos);
        double xw[1] = {bs(j, s)};
        double beta_old = calc_scalar_beta(xw[0], a_val, thresh_s, tau_s, slab);
        auto w_ll = [&](bool& changed) {
          double d = calc_scalar_beta(xw[0], a_val, thresh_s, tau_s, slab) - beta_old;
          changed = d != 0.0;
          if (!changed) return ll_curr;
          resid_prop = resid - X.col(j) * d;
          return ll_S(s, resid_prop);
        };
        auto w_accept = [&](bool changed, double ll) { if (changed) commit(ll); };
        N_s(j) += agess_block(xw, co, ad[s][j], it, S_max, ll_curr, scratch.memptr(), w_ll, w_accept);
        bs(j, s) = xw[0];
      }
    }

    double x[1] = {bs(idx_a0, s)};
    arma::vec bs_col = bs.col(s);
    arma::vec bias_old = calc_bias_vec_group(bs_col, tau_s, group_map, lambda_S(s), slab);
    arma::vec tau_h = tau_s * slab_weight_vec(bs_col.subvec(0, p - 1), slab);
    arma::vec bias_prop(p);
    auto a0_ll = [&](bool& changed) {
      double thresh_prop = threshold_from_a0(x[0], lambda_S(s));
      for (int j = 0; j < p; ++j) bias_prop(j) = tau_h(j) * activation_scalar(bs_col(p + group_map[j] - 1), thresh_prop, slab);
      arma::vec delta = bias_prop - bias_old;
      arma::uvec nz = find(delta != 0.0);
      changed = nz.n_elem > 0;
      if (!changed) return ll_curr;
      resid_prop = resid - X.cols(nz) * delta.elem(nz);
      return ll_S(s, resid_prop);
    };
    auto a0_accept = [&](bool changed, double ll) { if (changed) commit(ll); };
    N_s(idx_a0) += agess_block(x, co, ad[s][idx_a0], it, S_max, ll_curr, scratch.memptr(), a0_ll, a0_accept);
    bs(idx_a0, s) = x[0];
  }
}

} // namespace sstl
