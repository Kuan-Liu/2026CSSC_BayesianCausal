// =============================================================================
// Bayesian G-computation for Time-Varying Treatments
// 2026 CSSC Skills Workshop
//
// Data structure (2 visits):
//   N        : number of subjects
//   Nt       : number of time points (= 2)
//   n_coeffs : number of lagged coefficients (= 1 for Nt=2)
//   st, ed   : index vectors tracking which beta_L/theta entries
//              correspond to each time point (length Nt-1)
//   A        : N x Nt matrix of binary treatments
//   L        : N x Nt matrix of time-varying confounders (one per visit)
//   Y        : N-vector of continuous end-of-study outcomes
//
// For Nt=2: st = {1}, ed = {1}, n_coeffs = 1
// =============================================================================

data {
  int<lower=1> N;          // number of subjects
  int<lower=2> Nt;         // number of time points
  int<lower=1> n_coeffs;   // total lagged coefficients = sum_{t=2}^{Nt} (t-1)

  // index trackers: st[t-1] to ed[t-1] gives the slice of beta_L / theta
  // that corresponds to confounder model at time t
  int st[Nt-1];
  int ed[Nt-1];

  matrix[N, Nt] A;          // binary treatment history
  matrix[N, Nt] L;          // time-varying confounder history
  vector[N]     Y;           // continuous outcome
}

parameters {

  // --- Confounder model parameters ---
  vector[Nt]       beta_int;  // intercepts for each confounder model
  vector[n_coeffs] beta_L;    // lagged confounder effects (ridge prior)
  vector[n_coeffs] theta;     // lagged treatment effects on confounders (ridge prior)

  // --- Outcome model parameters ---
  real         int_y;         // outcome intercept
  vector[Nt]   beta_Ly;       // confounder history effects on Y
  vector[Nt]   theta_y;       // treatment history effects on Y

  // --- Variance parameters ---
  real<lower=0> phi_L0;       // SD for confounder at t=1 (baseline)
  real<lower=0> phi_Lt;       // SD for confounders at t>1
  real<lower=0> phi_y;        // SD for outcome model
}

model {

  // -------------------------------------------------------------------------
  // Ridge prior on lagged confounder and treatment effects
  // Shrinkage increases for lags further back in time:
  //   b - st[t-1] = 0 (most recent lag) -> SD = 1.5^0 = 1.0 (least shrinkage)
  //   b - st[t-1] = 1                   -> SD = 1.5^1 = 1.5
  //   b - st[t-1] = k (oldest lag)      -> SD = 1.5^k (most shrinkage)
  // -------------------------------------------------------------------------
  for (t in 2:Nt) {
    for (b in st[t-1]:ed[t-1]) {
      beta_L[b] ~ normal(0, (1.5^(b - st[t-1])) * 1);
      theta[b]  ~ normal(0, (1.5^(b - st[t-1])) * 1);
    }
  }

  // --- Moderately informative priors on outcome model parameters ---
  int_y   ~ normal(0, 0.5);
  beta_Ly ~ normal(0, 0.5);
  theta_y ~ normal(0, 0.5);
  phi_y   ~ cauchy(0, 2);

  // --- Priors on confounder model intercepts and variances ---
  beta_int ~ normal(0, 1);
  phi_L0   ~ cauchy(0, 2);
  phi_Lt   ~ cauchy(0, 2);

  // -------------------------------------------------------------------------
  // Likelihood
  // -------------------------------------------------------------------------

  // Confounder model at t=1 (baseline — no history to condition on)
  L[, 1] ~ normal(beta_int[1], phi_L0);

  // Sequential confounder models for t=2,...,Nt
  // Each conditions on the full past confounder and treatment history
  for (t in 2:Nt) {
    L[, t] ~ normal(
      beta_int[t]
        + L[, 1:(t-1)] * beta_L[st[t-1]:ed[t-1]]
        + A[, 1:(t-1)] * theta[st[t-1]:ed[t-1]],
      phi_Lt);
  }

  // Outcome model conditioning on full treatment and confounder history
  Y ~ normal(int_y + L * beta_Ly + A * theta_y, phi_y);
}

generated quantities {

  // -------------------------------------------------------------------------
  // G-computation: simulate confounder trajectories under two interventions
  //   mu1 = E[Y | always treated,    simulated L]
  //   mu0 = E[Y | never treated,     simulated L]
  //   ATE = mu1 - mu0  (computed in R after extracting these draws)
  // -------------------------------------------------------------------------

  row_vector[Nt] L_pred1;   // simulated L under always treated (A=1)
  row_vector[Nt] L_pred0;   // simulated L under never treated  (A=0)
  real mu1;
  real mu0;

  // --- Always treated (A=1 at all visits) ---
  L_pred1[1] = normal_rng(beta_int[1], phi_L0);

  for (t in 2:Nt) {
    L_pred1[t] = normal_rng(
      beta_int[t]
        + L_pred1[1:(t-1)] * beta_L[st[t-1]:ed[t-1]]
        + sum(theta[st[t-1]:ed[t-1]]),   // all treatments set to 1
      phi_Lt);
  }

  mu1 = int_y + L_pred1 * beta_Ly + sum(theta_y);

  // --- Never treated (A=0 at all visits) ---
  L_pred0[1] = normal_rng(beta_int[1], phi_L0);

  for (t in 2:Nt) {
    L_pred0[t] = normal_rng(
      beta_int[t]
        + L_pred0[1:(t-1)] * beta_L[st[t-1]:ed[t-1]],  // no treatment term
      phi_Lt);
  }

  mu0 = int_y + L_pred0 * beta_Ly;  // theta_y terms are zero (A=0)
}
