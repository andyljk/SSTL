#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(cpp17)]]

#include "sstl_updates.h"

using namespace Rcpp;
using namespace arma;

// Chains, empirical Bayes and the two entry points called by SSTL().
// Each chain runs the Gibbs sweep of the former R samplers (same update order
// and settings); EB (burn-in, MCEM warm start, SAEM with Polyak averaging)
// runs the same chains in blocks.

namespace {

// Progress bar drawn like utils::txtProgressBar(style = 3).
class ProgressBar {
 public:
  ProgressBar(int total, bool show, int console_width)
    : total_(total), show_(show && total > 0), width_(std::max(console_width - 10, 1)) {
    update(0);
  }
  void update(int value) {
    if (!show_) return;
    int nb = (int) std::nearbyint(width_ * (double) value / total_);
    int pc = (int) std::nearbyint(100.0 * value / total_);
    if (nb == nb_ && pc == pc_) return;
    std::string bar = "\r  |" + std::string(nb, '=') + std::string(width_ - nb, ' ');
    Rprintf("%s| %3d%%", bar.c_str(), pc);
    R_FlushConsole();
    nb_ = nb;
    pc_ = pc;
  }
  void finish() { if (show_) Rprintf("\n"); }
 private:
  int total_;
  bool show_;
  int width_;
  int nb_ = 0, pc_ = -1;
};

inline void check_interrupt(int i) {
  if (i % 16 == 0) Rcpp::checkUserInterrupt();
}

struct Family {
  int code;
  bool aft;
  double df;
};

// Latent blocks: (w_j, a_j) for j in each block of columns, then a0 alone.
std::vector<uvec> make_blocks(int p, int block_size) {
  std::vector<uvec> id;
  for (int start = 0; start < p; start += block_size) {
    int m = std::min(block_size, p - start);
    uvec idx(2 * m);
    for (int j = 0; j < m; ++j) {
      idx(j) = start + j;
      idx(m + j) = start + j + p;
    }
    id.push_back(idx);
  }
  id.push_back(uvec{(uword) (2 * p)});
  return id;
}

// Column indices of each group (group_map holds labels 1..G).
std::vector<uvec> make_groups(const IntegerVector& group_map) {
  int G = Rcpp::max(group_map);
  std::vector<std::vector<uword>> cols(G);
  for (int j = 0; j < group_map.size(); ++j) cols[group_map[j] - 1].push_back(j);
  std::vector<uvec> out(G);
  for (int g = 0; g < G; ++g) out[g] = conv_to<uvec>::from(cols[g]);
  return out;
}

double mean_log_pnorm(const rowvec& x) {
  double s = 0.0;
  for (double v : x) s += R::pnorm(v, 0.0, 1.0, 1, 1);
  return s / x.n_elem;
}

NumericVector as_numeric(const vec& x) { return NumericVector(x.begin(), x.end()); }

// ---------------------------------------------------------------------------
// Target-only chain
// ---------------------------------------------------------------------------
struct TOChain {
  const mat& X;
  const vec& Y;
  const vec& C;
  Family fam;
  int slab;
  bool intercept;
  int S_max;
  std::vector<uvec> id;
  vec sd_0;

  vec b;
  double xi, sd_y, b0, lambda;
  vec resid, N_s;

  TOChain(const mat& X_, const vec& Y_, const vec& C_, Family fam_, int slab_, bool intercept_)
    : X(X_), Y(Y_), C(C_), fam(fam_), slab(slab_), intercept(intercept_) {}

  int p() const { return X.n_cols; }
  vec beta() const {
    int p = this->p();
    return sstl::get_beta(b.subvec(0, p - 1), b.subvec(p, 2 * p - 1), std::exp(xi), b(2 * p), lambda, slab);
  }
  void reset_resid() { resid = Y - b0 - X * beta(); }

  void step() {
    int fc = fam.code;
    if (intercept) {
      b0 = fam.aft ? sstl::update_intercept_to_aft(b0, resid, Y, C, sd_y, fc, 2.0)
                   : sstl::update_intercept_to_general(b0, resid, Y, sd_y, fc, 2.0, fam.df);
    }
    if (fam.aft) sstl::update_blocks_aft(b, resid, X, Y, C, id, sd_0, lambda, std::exp(xi), sd_y, S_max, fc, slab, N_s);
    else sstl::update_blocks_general(b, resid, X, Y, id, sd_0, lambda, std::exp(xi), sd_y, S_max, fc, slab, fam.df, N_s);

    vec Y_scale = Y - b0;
    xi = fam.aft ? sstl::update_scale_aft(xi, Y_scale, resid, Y, C, sd_y, 2.0, fc)
                 : sstl::update_scale_general(xi, Y_scale, resid, Y, sd_y, 2.0, fc, fam.df);

    // Outcome scale, shape or precision (logistic and Poisson have none)
    if (fam.aft) sd_y = sstl::update_sigma_to_aft(resid, Y, C, sd_y, fc, 0.1);
    else if (fc != 2 && fc != 4) sd_y = sstl::update_sigma_to_general(resid, Y, sd_y, fc, fc >= 5 ? 0.3 : 0.1, fam.df);
  }

  // Runs n iterations and returns the a0 draws.
  rowvec run(int n) {
    rowvec a0(n);
    for (int i = 0; i < n; ++i) {
      check_interrupt(i);
      step();
      a0(i) = b(2 * p());
    }
    return a0;
  }
};

List eb_to(TOChain& ch, const List& eb, bool verbose, int width) {
  int N = eb["N"], burn = eb["burn"], K_block = eb["K_block"];
  double gamma_power = eb["gamma_power"], lr = eb["lr"], schedule = eb["schedule"];
  double lambda_min = eb["lambda_min"], lambda_max = eb["lambda_max"], polyak_start = eb["polyak_start"];
  bool adagrad = as<std::string>(eb["optimizer"]) == "adagrad";
  double adagrad_eps = eb["adagrad_eps"], max_log_step = eb["max_log_step"];
  int n_blocks = N / K_block;

  // Burn-in
  ch.reset_resid();
  rowvec a0 = ch.run(burn);
  if (!ch.intercept) ch.b0 = 0.0;
  int from = std::max(0, burn - 1001);
  double B_hat = mean_log_pnorm(a0.subvec(from, burn - 1)); // initial guess for E[log Phi(a0)]

  double lambda = ch.lambda;
  double theta = std::log(lambda);
  double G_acc = 0.0;
  vec mc_lam(n_blocks);

  ProgressBar pb(n_blocks, verbose, width);
  for (int t = 1; t <= n_blocks; ++t) {
    ch.reset_resid();
    a0 = ch.run(K_block);

    // Stochastic approximation of B(lambda) = E[log Phi(a0)]
    double gamma_t = std::pow(t, -gamma_power);
    B_hat = (1 - gamma_t) * B_hat + gamma_t * mean_log_pnorm(a0);

    if (!adagrad) {
      double lambda_target = -lambda / B_hat;
      double lr_t = lr / std::pow(t, schedule);
      lambda = std::max(lambda_min, std::min(lambda_max, (1 - lr_t) * lambda + lr_t * lambda_target));
      theta = std::log(lambda);
    } else { // AdaGrad on theta = log(lambda)
      double d_t = std::log(-1 / B_hat);
      G_acc += d_t * d_t;
      double lr_t = lr / (std::pow(t, schedule) * std::sqrt(G_acc + adagrad_eps));
      theta += std::max(-max_log_step, std::min(max_log_step, lr_t * d_t));
      theta = std::max(std::log(lambda_min), std::min(std::log(lambda_max), theta));
      lambda = std::exp(theta);
    }
    ch.lambda = lambda;
    mc_lam(t - 1) = lambda;
    pb.update(t);
  }
  pb.finish();

  // Polyak average in the tail for a more stable final estimate
  double lambda_final = lambda;
  if (polyak_start < 1 && n_blocks > 0) {
    int i0 = std::max(1, (int) std::floor(polyak_start * n_blocks));
    lambda_final = mean(mc_lam.subvec(i0 - 1, n_blocks - 1));
  }

  List out = List::create(Named("mc.lam") = as_numeric(mc_lam),
                          Named("lambda_final") = lambda_final,
                          Named("b_c") = as_numeric(ch.b),
                          Named("xi") = ch.xi,
                          Named("sd_y") = ch.sd_y);
  if (ch.intercept) out["b0_final"] = ch.b0;
  return out;
}

// ---------------------------------------------------------------------------
// Transfer-learning chain (grouped or not)
// ---------------------------------------------------------------------------
struct TLChain {
  const mat& X_T;
  const vec& Y_T;
  const vec& C_T;
  const std::vector<mat>& X_s;
  const std::vector<vec>& Y_s;
  const std::vector<vec>& C_s;
  Family fam;
  int slab;
  bool intercept;
  bool grouped;
  IntegerVector group_map;
  std::vector<uvec> groups;

  // sampler settings
  int S_max;
  std::vector<uvec> id;
  double xi_prior;
  vec sd_T;

  // state
  vec bt;
  mat bs;
  double xi, sig_T, b0_T, lambda_T;
  vec xi_s, sig_s, b0_s, lambda_s;
  vec resid_T, beta_T, N_t, N_s;
  std::vector<vec> resid_S;

  TLChain(const mat& X_T_, const vec& Y_T_, const vec& C_T_,
          const std::vector<mat>& X_s_, const std::vector<vec>& Y_s_, const std::vector<vec>& C_s_,
          Family fam_, int slab_, bool intercept_, Nullable<IntegerVector> group_map_)
    : X_T(X_T_), Y_T(Y_T_), C_T(C_T_), X_s(X_s_), Y_s(Y_s_), C_s(C_s_),
      fam(fam_), slab(slab_), intercept(intercept_), grouped(group_map_.isNotNull()) {
    if (grouped) {
      group_map = IntegerVector(group_map_);
      groups = make_groups(group_map);
    }
  }

  int S() const { return X_s.size(); }
  int p() const { return X_T.n_cols; }
  int d() const { return grouped ? p() + (int) groups.size() + 1 : 2 * p() + 1; }

  void configure(int S_max_, int block_size, double xi_prior_) {
    S_max = S_max_;
    id = grouped ? groups : make_blocks(p(), block_size);
    xi_prior = xi_prior_;
    sd_T = ones(d());
  }

  vec coef(const vec& b, double lambda, double tau) const {
    int p = this->p();
    if (grouped) {
      int G = groups.size();
      return sstl::get_beta_group(b.subvec(0, p - 1), b.subvec(p, p + G - 1), group_map, tau, b(p + G), lambda, slab);
    }
    return sstl::get_beta(b.subvec(0, p - 1), b.subvec(p, 2 * p - 1), tau, b(2 * p), lambda, slab);
  }

  void reset_resid() {
    beta_T = coef(bt, lambda_T, std::abs(xi));
    resid_T = Y_T - b0_T - X_T * beta_T;
    resid_S.resize(S());
    for (int s = 0; s < S(); ++s) {
      resid_S[s] = Y_s[s] - b0_s(s) - X_s[s] * (beta_T + coef(bs.col(s), lambda_s(s), std::abs(xi_s(s))));
    }
  }

  bool has_sigma() const { return fam.aft || (fam.code != 2 && fam.code != 4); }
  double sigma_step() const { return (!fam.aft && !grouped && fam.code >= 5) ? 0.3 : 0.1; }
  double sigma_update(const vec& resid, const vec& Y, int s, double sigma) const {
    if (fam.aft) return sstl::update_sigma_jeffreys_tl_aft(resid, Y, s < 0 ? C_T : C_s[s], sigma, 0.1, fam.code);
    return sstl::update_sigma_mh_tl_general(resid, Y, sigma, sigma_step(), fam.code, fam.df);
  }

  void step() {
    int S = this->S(), fc = fam.code;

    // Target: intercept, latent coefficients, slab scale, outcome scale
    if (intercept) {
      b0_T = fam.aft ? sstl::update_target_intercept_tl_aft(b0_T, resid_T, Y_T, C_T, sig_T, fc, 2.0)
                     : sstl::update_target_intercept_tl_general(b0_T, resid_T, Y_T, sig_T, fc, 2.0, fam.df);
    }
    double tau = std::abs(xi);
    if (grouped) {
      if (fam.aft) sstl::update_target_group_aft(bt, resid_T, resid_S, X_T, Y_T, C_T, X_s, Y_s, C_s, id, group_map, sd_T,
                                                 lambda_T, tau, sig_T, sig_s, S_max, fc, slab, N_t);
      else sstl::update_target_group_general(bt, resid_T, resid_S, X_T, Y_T, X_s, Y_s, id, group_map, sd_T,
                                             lambda_T, tau, sig_T, sig_s, S_max, fc, slab, fam.df, N_t);
    } else {
      if (fam.aft) sstl::update_target_aft(bt, resid_T, resid_S, X_T, Y_T, C_T, X_s, Y_s, C_s, id, sd_T,
                                           lambda_T, tau, sig_T, sig_s, S_max, fc, slab, N_t);
      else sstl::update_target_general(bt, resid_T, resid_S, X_T, Y_T, X_s, Y_s, id, sd_T,
                                       lambda_T, tau, sig_T, sig_s, S_max, fc, slab, fam.df, N_t);
    }

    vec Y_T_scale = Y_T - b0_T;
    std::vector<vec> Y_s_scale(S);
    for (int s = 0; s < S; ++s) {
      Y_s_scale[s] = Y_s[s] - b0_s(s) - X_s[s] * coef(bs.col(s), lambda_s(s), std::abs(xi_s(s)));
    }
    double sd_0 = grouped ? 2.0 : xi_prior;
    xi = fam.aft ? sstl::update_target_scale_aft(xi, sd_0, Y_T_scale, resid_T, Y_s_scale, resid_S,
                                                 Y_T, C_T, Y_s, C_s, sig_T, sig_s, fc)
                 : sstl::update_target_scale_general(xi, sd_0, Y_T_scale, resid_T, Y_s_scale, resid_S,
                                                     Y_T, Y_s, sig_T, sig_s, fc, fam.df);
    beta_T = coef(bt, lambda_T, std::abs(xi));
    if (has_sigma()) sig_T = sigma_update(resid_T, Y_T, -1, sig_T);

    if (S == 0) return;

    // Sources: intercepts, biases, slab scales, outcome scales
    if (intercept) {
      if (fam.aft) sstl::update_source_intercepts_tl_aft(b0_s, resid_S, Y_s, C_s, sig_s, fc, 2.0);
      else sstl::update_source_intercepts_tl_general(b0_s, resid_S, Y_s, sig_s, fc, 2.0, fam.df);
    }
    vec tau_S = abs(xi_s);
    if (grouped) {
      if (fam.aft) sstl::update_source_joint_group_aft(bs, resid_S, X_s, Y_s, C_s, id, group_map, lambda_s, tau_S,
                                                       sig_s, S_max, fc, slab, N_s);
      else sstl::update_source_joint_group_general(bs, resid_S, X_s, Y_s, id, group_map, lambda_s, tau_S,
                                                   sig_s, S_max, fc, slab, fam.df, N_s);
    } else {
      if (fam.aft) sstl::update_source_joint_aft(bs, resid_S, X_s, Y_s, C_s, lambda_s, tau_S, sig_s, S_max, fc, slab, N_s);
      else sstl::update_source_joint_general(bs, resid_S, X_s, Y_s, lambda_s, tau_S, sig_s, S_max, fc, slab, fam.df, N_s);
    }

    for (int s = 0; s < S; ++s) Y_s_scale[s] = Y_s[s] - b0_s(s) - X_s[s] * beta_T;
    xi_s = fam.aft ? sstl::update_source_scales_aft(xi_s, 0.2, Y_s_scale, resid_S, Y_s, C_s, sig_s, fc)
                   : sstl::update_source_scales_general(xi_s, 0.2, Y_s_scale, resid_S, Y_s, sig_s, fc, fam.df);
    if (has_sigma()) {
      for (int s = 0; s < S; ++s) sig_s(s) = sigma_update(resid_S[s], Y_s[s], s, sig_s(s));
    }
  }

  // Runs n iterations and returns the a0 draws (row 0: target, row s + 1: source s).
  mat run(int n) {
    int a0 = d() - 1;
    mat draws(S() + 1, n);
    for (int i = 0; i < n; ++i) {
      check_interrupt(i);
      step();
      draws(0, i) = bt(a0);
      for (int s = 0; s < S(); ++s) draws(s + 1, i) = bs(a0, s);
    }
    return draws;
  }

  // The former R samplers restarted each EB block from |xi|.
  void fold_scales() {
    xi = std::abs(xi);
    xi_s = abs(xi_s);
  }

  void set_lambda(const vec& lam) {
    lambda_T = lam(0);
    lambda_s = lam.tail(S());
  }
};

List eb_tl(TLChain& ch, const List& eb, bool verbose, int width) {
  int N = eb["N"], burn = eb["burn"], K_block = eb["K_block"];
  double gamma_power = eb["gamma_power"], lr = eb["lr"], schedule = eb["schedule"];
  double lambda_min = eb["lambda_min"], lambda_max = eb["lambda_max"], polyak_start = eb["polyak_start"];
  bool polyak = eb["polyak"], warm_start = eb["warm_start"];
  int S = ch.S();
  int n_blocks = N / K_block;
  const int warm_start_iter = 10, warm_start_N = 300;

  vec lam(S + 1);
  lam(0) = ch.lambda_T;
  lam.tail(S) = ch.lambda_s;
  lam = clamp(lam, lambda_min, lambda_max);
  std::vector<bool> active(S + 1);
  for (int k = 0; k <= S; ++k) active[k] = lam(k) > lambda_min && lam(k) < lambda_max;
  ch.set_lambda(lam);

  auto clip = [](double x) { return std::max(std::min(x, -1e-6), -30.0); };

  // Burn-in
  ch.reset_resid();
  mat a0 = ch.run(burn);
  ch.fold_scales();
  if (!ch.intercept) {
    ch.b0_T = 0.0;
    ch.b0_s.zeros();
  }
  int from = std::max(0, burn - 1000);
  vec B_hat(S + 1);
  for (int k = 0; k <= S; ++k) B_hat(k) = mean_log_pnorm(a0(k, span(from, burn - 1)));

  // MCEM warm start: fast approximation to the neighborhood of the solution
  mat mc_lam_mcem(warm_start ? warm_start_iter : 0, S + 1);
  if (warm_start) {
    for (int it = 0; it < warm_start_iter; ++it) {
      if (verbose) Rprintf("warm start iteration: %d/%d\n", it + 1, warm_start_iter);
      ch.reset_resid();
      a0 = ch.run(warm_start_N);
      ch.fold_scales();
      for (int k = 0; k <= S; ++k) {
        double m = clip(mean_log_pnorm(a0.row(k)));
        if (active[k]) {
          lam(k) = std::min(lambda_max, std::max(lambda_min, -lam(k) / m));
          active[k] = lam(k) > lambda_min && lam(k) < lambda_max;
        }
        B_hat(k) = m;
      }
      ch.set_lambda(lam);
      mc_lam_mcem.row(it) = lam.t();
    }
  }

  // SAEM with Robbins-Monro steps and Polyak-Ruppert averaging
  int polyak_from = std::max(1, (int) std::floor(polyak_start * n_blocks));
  vec lam_avg = zeros(S + 1);
  int lam_avg_n = 0;
  mat mc_lam(n_blocks, S + 1);

  ProgressBar pb(n_blocks, verbose, width);
  for (int t = 1; t <= n_blocks; ++t) {
    ch.reset_resid();
    a0 = ch.run(K_block);
    ch.fold_scales();

    double gamma_t = std::pow(t, -gamma_power);
    for (int k = 0; k <= S; ++k) {
      double m = mean_log_pnorm(a0.row(k));
      if (active[k]) B_hat(k) = clip((1 - gamma_t) * B_hat(k) + gamma_t * m);
    }
    double lr_t = lr / std::pow(t, schedule);
    for (int k = 0; k <= S; ++k) {
      if (!active[k]) continue;
      double lambda_target = -lam(k) / B_hat(k);
      lam(k) = std::min(lambda_max, std::max(lambda_min, (1 - lr_t) * lam(k) + lr_t * lambda_target));
      active[k] = lam(k) > lambda_min && lam(k) < lambda_max;
    }
    ch.set_lambda(lam);
    mc_lam.row(t - 1) = lam.t();
    if (polyak && t >= polyak_from) {
      lam_avg_n++;
      lam_avg += (lam - lam_avg) / lam_avg_n;
    }
    pb.update(t);
  }
  pb.finish();

  vec final_lam = (polyak && lam_avg_n > 0) ? lam_avg : lam;
  for (int k = 0; k <= S; ++k) if (!active[k]) final_lam(k) = lam(k);

  CharacterVector lam_names(S + 1);
  lam_names[0] = "lambda_T";
  for (int s = 0; s < S; ++s) lam_names[s + 1] = "lambda_s" + std::to_string(s + 1);
  NumericMatrix mc_lam_r = wrap(mc_lam), mc_lam_mcem_r = wrap(mc_lam_mcem);
  colnames(mc_lam_r) = lam_names;
  colnames(mc_lam_mcem_r) = lam_names;

  List out = List::create(Named("mc.lam") = mc_lam_r,
                          Named("final_lam") = as_numeric(final_lam),
                          Named("bt_c") = as_numeric(ch.bt),
                          Named("bs_c") = ch.bs,
                          Named("xi") = ch.xi,
                          Named("xi_s") = as_numeric(ch.xi_s),
                          Named("sig_T") = ch.sig_T,
                          Named("sig_s") = as_numeric(ch.sig_s));
  if (ch.intercept) {
    out["b0_T"] = ch.b0_T;
    out["b0_s"] = as_numeric(ch.b0_s);
  }
  out["mc_lam_mcem"] = mc_lam_mcem_r;
  return out;
}

std::vector<mat> list_mats(const List& x) { return sstl::mat_views(x); }
std::vector<vec> list_vecs(const List& x) { return sstl::vec_views(x); }

} // namespace

// [[Rcpp::export]]
List sstl_fit_to(const arma::mat& X, const arma::vec& Y, const arma::vec& C,
                 int fam_code, bool aft, double df, int slab_code, bool intercept,
                 double lambda, const List& init, const List& mcmc, Nullable<List> eb,
                 std::string label, int verbose, int width) {
  TOChain ch(X, Y, C, Family{fam_code, aft, df}, slab_code, intercept);
  int p = X.n_cols;
  bool show = verbose == 1;
  List eb_out;

  if (eb.isNotNull()) {
    List ebl(eb);
    List eb_init = ebl["init"];
    ch.S_max = ebl["S.max"];
    ch.id = make_blocks(p, ebl["block_size"]);
    ch.sd_0 = ones(2 * p + 1);
    ch.b = as<vec>(eb_init["b.c"]);
    ch.xi = eb_init["xi"];
    ch.sd_y = eb_init["sd_y"];
    ch.b0 = eb_init["b0"];
    ch.lambda = lambda;
    if (show) Rprintf("Empirical Bayes (%s)\n", label.c_str());
    eb_out = eb_to(ch, ebl, show, width);
    ch.lambda = eb_out["lambda_final"];
    if (!intercept) ch.b0 = as<double>(init["b0"]);
  } else {
    ch.b = as<vec>(init["b.c"]);
    ch.xi = init["xi"];
    ch.sd_y = init["sd_y"];
    ch.b0 = init["b0"];
    ch.lambda = lambda;
  }
  ch.S_max = mcmc["S.max"];
  ch.id = make_blocks(p, mcmc["block_size"]);
  ch.sd_0 = as<vec>(init["sd.0"]);
  ch.reset_resid();

  int N = mcmc["N"];
  bool debug = mcmc["debug"];
  int K = ch.id.size();
  mat MC_beta(N, p), mc_b, N_s;
  vec mc_sigma(N), mc_b0(N), mc_tau;
  if (debug) {
    mc_b.set_size(N, 2 * p + 1);
    N_s.set_size(N, K);
    mc_tau.set_size(N);
  }

  if (show) Rprintf("Main MCMC (%s)\n", label.c_str());
  ProgressBar pb(N, show, width);
  for (int i = 0; i < N; ++i) {
    check_interrupt(i);
    ch.step();
    if (debug) {
      mc_b.row(i) = ch.b.t();
      mc_tau(i) = std::exp(ch.xi);
      N_s.row(i) = ch.N_s.t();
    }
    mc_sigma(i) = ch.sd_y;
    mc_b0(i) = ch.b0;
    MC_beta.row(i) = ch.beta().t();
    pb.update(i + 1);
  }
  pb.finish();

  List out;
  if (debug) {
    out = List::create(Named("mc_b") = mc_b, Named("MC_beta") = MC_beta, Named("N_s") = N_s,
                       Named("mc_tau") = as_numeric(mc_tau), Named("mc_sigma") = as_numeric(mc_sigma));
    if (intercept) out["mc_b0"] = as_numeric(mc_b0);
  } else {
    out = List::create(Named("MC_beta") = MC_beta, Named("mc_sigma") = as_numeric(mc_sigma));
    if (intercept) out["MC_b0"] = as_numeric(mc_b0);
  }
  out["lambda"] = ch.lambda;
  if (eb.isNotNull()) out["EB"] = eb_out;
  return out;
}

// [[Rcpp::export]]
List sstl_fit_tl(const arma::mat& X_T, const arma::vec& Y_T, const arma::vec& C_T,
                 const List& X_s_list, const List& Y_s_list, const List& C_s_list,
                 Nullable<IntegerVector> group_map,
                 int fam_code, bool aft, double df, int slab_code, bool intercept,
                 double lambda_T, const arma::vec& lambda_s,
                 const List& init, const List& mcmc, Nullable<List> eb,
                 std::string label, int verbose, int width) {
  std::vector<mat> X_s = list_mats(X_s_list);
  std::vector<vec> Y_s = list_vecs(Y_s_list), C_s = list_vecs(C_s_list);
  TLChain ch(X_T, Y_T, C_T, X_s, Y_s, C_s, Family{fam_code, aft, df}, slab_code, intercept, group_map);
  int S = ch.S(), p = ch.p(), d = ch.d();
  bool show = verbose == 1;
  List eb_out;

  auto set_state = [&](const List& st) {
    ch.bt = as<vec>(st["bt.c"]);
    ch.bs = as<mat>(st["bs.c"]);
    ch.xi = st["xi"];
    ch.xi_s = as<vec>(st["xi_s"]);
    ch.sig_T = st["sig_T"];
    ch.sig_s = as<vec>(st["sig_s"]);
    ch.b0_T = st["b0_T"];
    ch.b0_s = as<vec>(st["b0_s"]);
  };

  if (eb.isNotNull()) {
    List ebl(eb);
    set_state(ebl["init"]);
    ch.configure(ebl["S.max"], ebl["block_size"], 1.0); // EB uses the default slab-scale prior
    ch.lambda_T = lambda_T;
    ch.lambda_s = lambda_s;
    if (show) Rprintf("Empirical Bayes (%s)\n", label.c_str());
    eb_out = eb_tl(ch, ebl, show, width);
    NumericVector final_lam = eb_out["final_lam"];
    ch.set_lambda(as<vec>(final_lam));
    if (!intercept) {
      ch.b0_T = as<double>(init["b0_T"]);
      ch.b0_s = as<vec>(init["b0_s"]);
    }
  } else {
    set_state(init);
    ch.lambda_T = lambda_T;
    ch.lambda_s = lambda_s;
  }
  ch.configure(mcmc["S.max"], mcmc["block_size"], mcmc["xi_prior"]);
  ch.reset_resid();

  int N = mcmc["N"];
  bool debug = mcmc["debug"];
  int K = ch.grouped ? d : (int) ch.id.size(); // length of the target's slice-iteration record
  mat MC_beta(N, p), mc_sig_s(S, N), MC_b0_S(N, S), mc_bt, n_t, n_s, mc_tau_S;
  vec mc_sig_T(N), MC_b0_T(N), mc_tau_T;
  cube mc_bs;
  if (debug) {
    mc_bt.set_size(N, d);
    n_t.set_size(N, K);
    n_s.set_size(N, d);
    n_s.fill(NA_REAL);
    mc_bs.set_size(d, S, N);
    mc_tau_T.set_size(N);
    mc_tau_S.set_size(N, S);
  }

  if (show) Rprintf("Main MCMC (%s)\n", label.c_str());
  ProgressBar pb(N, show, width);
  for (int i = 0; i < N; ++i) {
    check_interrupt(i);
    ch.step();
    mc_sig_T(i) = ch.sig_T;
    if (S > 0) mc_sig_s.col(i) = ch.sig_s;
    if (debug) {
      mc_bt.row(i) = ch.bt.t();
      n_t.row(i) = ch.N_t.t();
      mc_tau_T(i) = std::abs(ch.xi);
      if (S > 0) {
        mc_bs.slice(i) = ch.bs;
        n_s.row(i) = ch.N_s.t();
        mc_tau_S.row(i) = abs(ch.xi_s).t();
      }
    }
    MC_b0_T(i) = ch.b0_T;
    if (S > 0) MC_b0_S.row(i) = ch.b0_s.t();
    MC_beta.row(i) = ch.beta_T.t();
    pb.update(i + 1);
  }
  pb.finish();

  List out;
  if (debug) {
    out = List::create(Named("mc_bt") = mc_bt, Named("mc_bs") = mc_bs, Named("MC_beta") = MC_beta,
                       Named("n_t") = n_t, Named("n_s") = n_s,
                       Named("mc_sig_T") = as_numeric(mc_sig_T), Named("mc_sig_s") = mc_sig_s,
                       Named("mc_tau_T") = as_numeric(mc_tau_T), Named("mc_tau_S") = mc_tau_S);
    if (intercept) {
      out["mc_b0_T"] = as_numeric(MC_b0_T);
      out["mc_b0_S"] = MC_b0_S;
    }
  } else {
    out = List::create(Named("MC_beta") = MC_beta,
                       Named("mc_sig_T") = as_numeric(mc_sig_T), Named("mc_sig_s") = mc_sig_s);
    if (intercept) {
      out["MC_b0_T"] = as_numeric(MC_b0_T);
      out["MC_b0_S"] = MC_b0_S;
    }
  }
  vec lam(S + 1);
  lam(0) = ch.lambda_T;
  lam.tail(S) = ch.lambda_s;
  out["lambda"] = as_numeric(lam);
  if (eb.isNotNull()) out["EB"] = eb_out;
  return out;
}
