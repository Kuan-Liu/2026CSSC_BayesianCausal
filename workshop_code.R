# =============================================================================
# Bayesian Methods for Causal Effect Estimation
# 2026 CSSC Skills Workshop
# Kuan Liu, University of Toronto
#
# Required packages:
# install.packages(c("tidyverse","rstanarm","rstan","BART","bcf",
#                    "bayesmsm","gtools"))
# =============================================================================

library(tidyverse)
library(rstanarm)
library(rstan)
library(BART)
library(gtools)       # for rdirichlet()
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# =============================================================================
# PART I: Data Preparation - RHC Dataset
# =============================================================================

sas_origin <- as.Date("1960-01-01")

rhc <- read_csv("https://hbiostat.org/data/repo/rhc.csv") |>
  mutate(
    sadmdte  = as.Date(sadmdte,  origin = sas_origin),
    dthdte   = as.Date(dthdte,   origin = sas_origin),
    dschdte  = as.Date(dschdte,  origin = sas_origin),
    lstctdte = as.Date(lstctdte, origin = sas_origin),
    A        = as.integer(swang1 == "RHC"),
    Y_death  = as.integer(death == "Yes"),
    Y_los    = as.numeric(coalesce(dschdte, lstctdte) - sadmdte)
  )

print(rhc |> count(A, swang1))

# Confounders
covars <- c("age","sex","race","cat1",
            "meanbp1","hrt1","resp1","temp1","wtkilo1")

# =============================================================================
# PART II Section 1: Parametric Bayesian G-computation
# =============================================================================

fit_parametric <- stan_glm(
  Y_death ~ A + age + sex + race + cat1 +
            meanbp1 + hrt1 + resp1 + temp1 + wtkilo1,
  data            = rhc,
  family          = binomial(link = "logit"),
  prior           = normal(0, 2.5),
  prior_intercept = normal(0, 5),
  chains = 4, iter = 2000, warmup = 1000,
  seed = 42, refresh = 500)

print(summary(fit_parametric, probs = c(0.025, 0.975)))

# --- Bayesian Bootstrap G-computation ---
M <- nrow(as.matrix(fit_parametric))
n <- nrow(rhc)

# Counterfactual datasets
rhc_a1 <- rhc_a0 <- rhc
rhc_a1$A <- 1
rhc_a0$A <- 0

# Posterior predicted probabilities under each intervention
# mu_a1 and mu_a0 are M x n matrices
mu_a1_param <- posterior_epred(fit_parametric, newdata = rhc_a1)
mu_a0_param <- posterior_epred(fit_parametric, newdata = rhc_a0)

# Bayesian bootstrap: one fresh Dirichlet draw per MCMC iteration
set.seed(42)
psi_parametric <- numeric(M)
for(m in 1:M){
  bb_w            <- as.vector(rdirichlet(1, rep(1, n)))
  psi_parametric[m] <- sum(bb_w * (mu_a1_param[m,] - mu_a0_param[m,]))
}

# Parametric Bayesian G-computation ATE (Risk Difference)
round(quantile(psi_parametric, c(0.025, 0.5, 0.975)), 4)
# P(ATE > 0 | data)
round(mean(psi_parametric > 0), 3)

# Plot posterior
p1 <- ggplot(data.frame(psi = psi_parametric), aes(x = psi)) +
  geom_density(fill = "steelblue", alpha = 0.4) +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "red") +
  geom_vline(xintercept = median(psi_parametric),
             linetype = "solid", colour = "steelblue") +
  labs(x = expression(Psi ~ "(Risk Difference)"),
       y = "Posterior Density",
       title = "Parametric Bayes G-computation",
       subtitle = "Effect of RHC on 30-day Mortality") +
  theme_minimal(base_size = 13)

print(p1)

# =============================================================================
# PART II - Section 2: Nonparametric Bayesian G-computation via BART
# =============================================================================

X_train <- model.matrix(~ A + ., data = rhc[, c("A", covars)])[, -1]
Y_train <- rhc$Y_death

# Stack counterfactual test sets
X_a1 <- X_a0 <- X_train
X_a1[, "A"] <- 1
X_a0[, "A"] <- 0
X_test <- rbind(X_a1, X_a0)   # 2n rows

# Fit probit BART
set.seed(42)
bart_fit <- gbart(
  x.train = X_train,
  y.train = Y_train,
  x.test  = X_test,
  type    = "pbart",
  ndpost  = 1000,
  nskip   = 500)

# Extract posterior predictions (convert probit to probability)
mu_a1_bart <- t(pnorm(bart_fit$yhat.test[, 1:n]))          # n x M
mu_a0_bart <- t(pnorm(bart_fit$yhat.test[, (n+1):(2*n)])) # n x M

# Bayesian bootstrap standardization
bayes_boot <- function(mu_a1, mu_a0, seed = 42){
  set.seed(seed)
  n <- nrow(mu_a1); M <- ncol(mu_a1)
  sapply(1:M, function(m){
    w <- as.vector(rdirichlet(1, rep(1, n)))
    sum(w * (mu_a1[,m] - mu_a0[,m]))
  })
}

psi_bart <- bayes_boot(mu_a1_bart, mu_a0_bart)

# BART G-computation ATE (Risk Difference)
round(quantile(psi_bart, c(0.025, 0.5, 0.975)), 4)
# P(ATE > 0 | data) 
round(mean(psi_bart > 0), 3)

# =============================================================================
# PART II - Section 3: Bayesian Propensity Score Weighting
# =============================================================================

ps_fit <- stan_glm(
  A ~ age + sex + race + cat1 + meanbp1 + hrt1 + resp1 + temp1 + wtkilo1,
  data            = rhc,
  family          = binomial(link = "logit"),
  prior           = normal(0, 2.5),
  prior_intercept = normal(0, 5),
  chains = 4, iter = 2000, warmup = 1000,
  seed = 42, refresh = 500)

# Posterior draws of P(A=1|L): M x n matrix
ps_draws <- posterior_epred(ps_fit)
M_ps     <- nrow(ps_draws)
p_treat  <- mean(rhc$A)

# Posterior stabilized weights: M x n matrix
sw_draws <- ifelse(
  matrix(rhc$A, nrow = M_ps, ncol = n, byrow = TRUE) == 1,
  p_treat     / ps_draws,
  (1-p_treat) / (1 - ps_draws))

# Weighted outcome model per posterior weight draw
psi_ipw <- numeric(M_ps)
for(m in 1:M_ps){
  fit_m      <- lm(Y_death ~ A, data = rhc, weights = sw_draws[m,])
  psi_ipw[m] <- coef(fit_m)["A"]
}

# Bayesian PS Weighting ATE (Risk Difference
round(quantile(psi_ipw, c(0.025, 0.5, 0.975)), 3)

# --- Compare all three point-treatment estimators ---
results_pt <- data.frame(
  Method = c("Parametric Bayes G-comp",
             "BART G-comp",
             "Bayesian PS Weighting"),
  Mean   = c(mean(psi_parametric), mean(psi_bart), mean(psi_ipw)),
  Q2.5   = c(quantile(psi_parametric, 0.025),
             quantile(psi_bart,       0.025),
             quantile(psi_ipw,        0.025)),
  Q97.5  = c(quantile(psi_parametric, 0.975),
             quantile(psi_bart,       0.975),
             quantile(psi_ipw,        0.975))
)

print(round(results_pt[, -1], 4) |> cbind(Method = results_pt$Method))

p_compare <- ggplot(results_pt,
                    aes(x = Method, y = Mean, ymin = Q2.5, ymax = Q97.5)) +
  geom_pointrange(colour = "steelblue", size = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  labs(y = "ATE: Risk Difference (30-day Mortality)",
       title = "Effect of RHC - Method Comparison",
       caption = "Error bars: 95% Posterior Credible Intervals") +
  theme_minimal(base_size = 13) +
  coord_flip()
print(p_compare)

# =============================================================================
# PART III - Section 1: Bayesian MSM via bayesmsm
# =============================================================================
dat <- read_csv("continuous_outcome_data.csv")
glimpse(dat)

library(bayesmsm)

# Step 1: Bayesian weight estimation
bayes_wt <- bayesweight(
  trtmodel.list = list(
    a_1 ~ w1 + w2 + L1_1 + L2_1,
    a_2 ~ w1 + w2 + L1_1 + L2_1 + L1_2 + L2_2 + a_1),
  data     = dat,
  n.iter   = 2500, n.burnin = 1500, n.thin = 5,
  parallel = TRUE, n.chains = 2, seed = 42)

# Step 2: Bayesian MSM estimation
msm_fit <- bayesmsm(
  ymodel     = y ~ a_1 + a_2,
  family     = "gaussian",
  nvisit     = 2,
  reference  = c(0, 0),      # never treated
  comparator = c(1, 1),      # always treated
  data       = dat,
  wmean      = bayes_wt$weights,
  nboot      = 1000,
  optim_method = "BFGS",
  parallel   = TRUE,
  ncore = 4,
  seed       = 42)

summary_bayesmsm(msm_fit)

# Plot posterior ATE
plot_ATE(msm_fit,
         col_density = "steelblue",
         main = "Bayesian MSM: Posterior ATE\nAlways vs Never Treated (Continuous Outcome)")

# =============================================================================
# PART IV: Bayesian Sensitivity Analysis
# =============================================================================

# Stan model as a string - intentionally omits one confounder
sens_model_code <- "
data {
  int<lower=0> N;
  int<lower=0,upper=1> Y[N];   // binary outcome
  int<lower=0,upper=1> A[N];   // binary treatment
  vector[N] L;                  // observed confounder (standardized)
  real xi1;                     // sensitivity param: U -> Y (log-OR)
  real xi2;                     // sensitivity param: U -> A (log-OR)
}
 
parameters {
  // Outcome model parameters
  real eta0;
  real eta1;
  real eta2;
 
  // Treatment model parameters
  real gam0;
  real gam1;
 
  // Latent unmeasured confounder - one value per subject
  // Sampled as missing data at every MCMC iteration
  vector[N] U;
}
 
model {
  // ---- Priors ----
  eta0 ~ normal(0, 3);
  eta1 ~ normal(0, 3);
  eta2 ~ normal(0, 3);
  gam0 ~ normal(0, 3);
  gam1 ~ normal(0, 3);
 
  // U ~ N(0,1): standardized latent confounder, independent of L
  // This is conservative: if U correlated with L, adjusting for L
  // would partially remove the confounding
  U ~ normal(0, 1);
 
  // ---- Likelihood ----
  // Both outcome and treatment depend on U - this is what makes U a confounder
  Y ~ bernoulli_logit(eta0 + eta1*to_vector(A) + eta2*L + xi1*U);
  A ~ bernoulli_logit(gam0 + gam1*L + xi2*U);
}
 
generated quantities {
  // G-computation: posterior ATE integrating over posterior U draws
  // For each MCMC draw, set A=1 and A=0 for all subjects,
  // keeping U fixed at its current posterior draw
  vector[N] mu1;
  vector[N] mu0;
  real psi;
 
  for(i in 1:N){
    mu1[i] = inv_logit(eta0 + eta1*1 + eta2*L[i] + xi1*U[i]);
    mu0[i] = inv_logit(eta0 + eta1*0 + eta2*L[i] + xi1*U[i]);
  }
  psi = mean(mu1) - mean(mu0);
}
"

sens_mod <- stan_model(model_code = sens_model_code)

# Use RHC: standardize Y_death and age, omit a known confounder
stan_data_base <- list(
  N = nrow(rhc),
  Y = rhc$Y_death,                        # not rhc$Y
  A = rhc$A,
  L = as.vector(scale(rhc$age))           # not rhc$L_age
)

xi_grid <- expand.grid(
  xi1 = c(0, 0.5, 1.0, 1.5),    # U -> Y
  xi2 = c(0, 0.5, 1.0, 1.5)     # U -> A
)

fit_sens <- stan(
  model_code = sensitivity_model,
  data   = stan_data_sens,
  chains = 4, iter = 2000, warmup = 1000,
  seed   = 42, refresh = 500)

# Running sensitivity analysis across grid
psi_std <- extract(fit_sens, "psi_standard")[[1]]
psi_sen <- extract(fit_sens, "psi_sensitive")[[1]]

psi_grid <- vector("list", nrow(xi_grid))

for(k in 1:nrow(xi_grid)){
  
  xi1_k <- xi_grid$xi1[k]
  xi2_k <- xi_grid$xi2[k]
  
  cat(sprintf("Fitting model %d/%d: xi1=%.1f, xi2=%.1f\n",
              k, nrow(xi_grid), xi1_k, xi2_k))
  
  stan_data_k <- c(stan_data_base,
                   list(xi1 = xi1_k, xi2 = xi2_k))
  
  fit_k <- sampling(
    sens_mod,
    data    = stan_data_k,
    chains  = 2,
    iter    = 1500,
    warmup  = 500,
    seed    = 42,
    refresh = 0,    # suppress per-iteration output
    control = list(adapt_delta = 0.9,
                   max_treedepth = 10))
  
  # Check for divergences
  div <- sum(get_sampler_params(fit_k, inc_warmup = FALSE)[[1]][,"divergent__"])
  if(div > 0) cat(sprintf("  WARNING: %d divergent transitions\n", div))
  
  # Extract posterior draws of ATE (psi)
  psi_draws <- extract(fit_k, pars = "psi")[[1]]
  
  psi_grid[[k]] <- data.frame(
    xi1       = xi1_k,
    xi2       = xi2_k,
    xi1_label = paste0("xi1 = ", xi1_k,
                       " (OR=", round(exp(xi1_k), 2), ")"),
    xi2_label = paste0("xi2 = ", xi2_k,
                       " (OR=", round(exp(xi2_k), 2), ")"),
    mean      = mean(psi_draws),
    median    = median(psi_draws),
    lo        = quantile(psi_draws, 0.025),
    hi        = quantile(psi_draws, 0.975)
  )
  
  cat(sprintf("  ATE: %.4f [%.4f, %.4f]  P(ATE<0)=%.3f\n",
              mean(psi_draws),
              quantile(psi_draws, 0.025),
              quantile(psi_draws, 0.975)))
}

# Combine results
results <- bind_rows(psi_grid)


# round(results[, c("xi1","xi2","mean","lo","hi","p_neg")], 4)

# ---- Plot 1: Posterior ATE across xi1 for each xi2 value ----
p1 <- ggplot(results,
             aes(x      = factor(xi1),
                 y      = mean,
                 ymin   = lo,
                 ymax   = hi,
                 colour = factor(xi2),
                 group  = factor(xi2))) +
  geom_pointrange(position = position_dodge(0.4), size = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  scale_colour_brewer(
    palette = "Blues",
    labels  = paste0("xi2 = ", unique(results$xi2),
                     " (OR=", round(exp(unique(results$xi2)), 2), ")")) +
  labs(
    x      = expression(xi[1] ~ "(effect of U on Y, log-OR)"),
    y      = expression(Psi ~ "(Risk Difference: ATE)"),
    colour = expression(xi[2] ~ "(U -> A)"),
    title  = "Posterior ATE across Unmeasured Confounding Grid",
    subtitle = "RHC effect on 30-day mortality | Latent variable sensitivity analysis",
    caption = "Error bars: 95% posterior credible intervals\nRed dashed line: ATE = 0") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "right")

print(p1)

# ---- Plot 2: Heatmap of posterior mean ATE ----
p2 <- ggplot(results,
             aes(x    = factor(xi1),
                 y    = factor(xi2),
                 fill = mean)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = round(mean, 3)),
            colour = "white", size = 4, fontface = "bold") +
  scale_fill_gradient2(
    low      = "steelblue",
    mid      = "white",
    high     = "darkred",
    midpoint = 0) +
  labs(
    x    = expression(xi[1] ~ "(U -> Y, log-OR)"),
    y    = expression(xi[2] ~ "(U -> A, log-OR)"),
    fill = "Mean ATE",
    title   = "Heatmap: Posterior Mean ATE",
    subtitle = "Blue = RHC harmful, Red = RHC beneficial, White = null") +
  theme_minimal(base_size = 13)

print(p2)


# =============================================================================
# PART V: Bayesian Causal Forests (BCF) - CATE Estimation
# =============================================================================

library(bcf)

X      <- model.matrix(~ ., data = rhc[, covars])[, -1]
Y_bcf  <- rhc$Y_death
A_bcf  <- rhc$A

# Step 1: Estimate propensity scores (logistic regression)
ps_glm <- glm(A_bcf ~ X, family = binomial)
ps_hat <- fitted(ps_glm)
print(summary(ps_hat))

# Step 2: Fit BCF
set.seed(42)
bcf_fit <- bcf(
  y          = Y_bcf,
  z          = A_bcf,
  x_control  = X,
  x_moderate = X,
  pihat      = ps_hat,
  nburn      = 500,
  nsim       = 1000)

# Step 3: Extract posterior draws
tau_draws    <- bcf_fit$tau
tau_hat      <- colMeans(tau_draws)               # was rowMeans - fix this
tau_lo       <- apply(tau_draws, 2, quantile, 0.025)
tau_hi       <- apply(tau_draws, 2, quantile, 0.975)
prob_benefit <- colMeans(tau_draws > 0)           # was rowMeans - fix this

# Population ATE from BCF
round(mean(tau_hat), 4)

# CATE by age subgroup
cate_df <- data.frame(
  tau_hat      = tau_hat,
  tau_lo       = tau_lo,
  tau_hi       = tau_hi,
  prob_benefit = prob_benefit,
  age          = rhc$age) |>
  mutate(age_q = cut(age,
                     quantile(age, 0:4/4),
                     labels = c("Q1 (youngest)","Q2","Q3","Q4 (oldest)"),
                     include.lowest = TRUE))

cate_sub <- cate_df |>
  group_by(age_q) |>
  summarise(
    mean_tau = mean(tau_hat),
    lo       = mean(tau_lo),
    hi       = mean(tau_hi),
    prob_ben = mean(prob_benefit),
    n        = n(),
    .groups  = "drop")

print(round(cate_sub[, -1], 4) |> cbind(age_q = cate_sub$age_q))

# mean_tau      lo     hi prob_ben    n         age_q
# 1   0.0514  0.0011 0.1066   0.9700 1434 Q1 (youngest)
# 2   0.0485 -0.0015 0.0997   0.9636 1434            Q2
# 3   0.0481 -0.0010 0.0973   0.9649 1433            Q3
# 4   0.0474 -0.0009 0.0956   0.9655 1434   Q4 (oldest)


