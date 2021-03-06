library(rstanarm)
library(rstan)
library(glmnet)
library(caret)
randseed <- 1592223649
set.seed(randseed)

# load data
DATA_PATH <- file.path(getwd(), "data/")
OUT_PATH <- file.path(getwd(), "res/")
# load causes
x_df <-  read.table(file.path(DATA_PATH, "simulated_causes.txt"))
names(x_df) <- seq(1, ncol(x_df), 1)
# load substitute confounders
T_hat_pmf <- read.table(file.path(DATA_PATH, "x_post_np_PMF.txt"))
x_t_df_pmf <- as.data.frame(cbind(x_df, T_hat_pmf))
T_hat_def <- read.table(file.path(DATA_PATH, "x_post_np_DEF.txt"))
x_t_df_def <- as.data.frame(cbind(x_df, T_hat_def))
# load true confounders
C <- read.table(file.path(DATA_PATH, "simulated_multicause_conf.txt"))
x_c_df <- as.data.frame(cbind(x_df, C))
# load outcome
ys <- read.table(file.path(DATA_PATH, "simulated_outcomes.txt"))
ys <- t(ys)
# load true coefficients
betas <- read.table(file.path(DATA_PATH, "simulated_true_coeffs.txt"))
betas <- t(betas)
n_causes <- dim(x_df)[2]+1
n_sims = dim(ys)[2]

# Run outcome models
summary_stats <- array(NA, dim=c(n_sims, 4, 4))
max_iter = 1e6
tol_rel_obj = 0.001
for (sim in seq(1, n_sims, 1)){
  print(paste0("SIMULATION ###", sim))
  y <- ys[,sim]
  beta <- betas[,sim]

  #fit ridge models
  fitridge_no_control = stan_glm(y~., data = x_df, family = gaussian(), prior = normal(),
                                 algorithm = "meanfield", adapt_delta = NULL, QR = TRUE,
                                 sparse = FALSE, iter = max_iter, tol_rel_obj = tol_rel_obj, seed = randseed)
  fitridge_oracle = stan_glm(y~., data = x_c_df, family = gaussian(), prior = normal(),
                          algorithm = "meanfield", adapt_delta = NULL, QR = TRUE,
                          sparse = FALSE, iter = max_iter, tol_rel_obj = tol_rel_obj, seed = randseed)
  fitridge_pmf = stan_glm(y~., data = x_t_df_pmf, family = gaussian(), prior = normal(),
                          algorithm = "meanfield", adapt_delta = NULL, QR = TRUE,
                          sparse = FALSE, iter = max_iter, tol_rel_obj = tol_rel_obj, seed = randseed)
  fitridge_def = stan_glm(y~., data = x_t_df_def, family = gaussian(), prior = normal(),
                          algorithm = "meanfield", adapt_delta = NULL, QR = TRUE,
                          sparse = FALSE, iter = max_iter, tol_rel_obj = tol_rel_obj, seed = randseed)
  
  no_control_coefs <- fitridge_no_control$coefficients[2:n_causes]
  oracle_coefs <- fitridge_oracle$coefficients[2:n_causes]
  pmf_coefs <- fitridge_pmf$coefficients[2:n_causes]
  def_coefs <- fitridge_def$coefficients[2:n_causes]
  
  rmse_no_control <- sqrt(mean((beta - no_control_coefs)**2))
  rmse_oracle <- sqrt(mean((beta - oracle_coefs)**2))
  rmse_pmf <- sqrt(mean((beta - pmf_coefs)**2))
  rmse_def <- sqrt(mean((beta - def_coefs)**2))
  
  # CI
  ci95_no_control <- posterior_interval(fitridge_no_control, prob = 0.95)
  ci95_oracle <- posterior_interval(fitridge_oracle, prob = 0.95)
  ci95_pmf <- posterior_interval(fitridge_pmf, prob = 0.95)
  ci95_def <- posterior_interval(fitridge_def, prob = 0.95)
  # coverage: if the 95ci covers the true coefficients
  nc_coverage <-  (beta >=ci95_no_control[2:n_causes,1]) & (beta <= ci95_no_control[2:n_causes,2])
  oracle_coverage <-  (beta >=ci95_oracle[2:n_causes,1]) & (beta <= ci95_oracle[2:n_causes,2])
  pmf_coverage <-  (beta >=ci95_pmf[2:n_causes,1]) & (beta <= ci95_pmf[2:n_causes,2])
  def_coverage <-  (beta >=ci95_def[2:n_causes,1]) & (beta <= ci95_def[2:n_causes,2])
  
  truth <- as.factor(ifelse(beta != 0, 1, 0)) # factor of positive / negative cases
  
  oracle_all_coverage <- sum(oracle_coverage)/50
  nc_all_coverage <- sum(nc_coverage)/50
  pmf_all_coverage <- sum(pmf_coverage)/50
  def_all_coverage <- sum(def_coverage)/50
  
  oracle_causal_coverage <- sum(oracle_coverage[truth==1])/10
  nc_causal_coverage <- sum(nc_coverage[truth==1])/10
  pmf_causal_coverage <- sum(pmf_coverage[truth==1])/10
  def_causal_coverage <- sum(def_coverage[truth==1])/10
  
  oracle_noncausal_coverage <- sum(oracle_coverage[truth==0])/40
  nc_noncausal_coverage <- sum(nc_coverage[truth==0])/40
  pmf_noncausal_coverage <- sum(pmf_coverage[truth==0])/40
  def_noncausal_coverage <- sum(def_coverage[truth==0])/40
  
  summary_stats[sim,,] <- rbind(cbind(rmse_oracle, oracle_all_coverage, oracle_causal_coverage, oracle_noncausal_coverage),
                                cbind(rmse_no_control, nc_all_coverage, nc_causal_coverage, nc_noncausal_coverage),
                                cbind(rmse_pmf, pmf_all_coverage, pmf_causal_coverage, pmf_noncausal_coverage),
                                cbind(rmse_def, def_all_coverage, def_causal_coverage, def_noncausal_coverage))
}

# average over all simulations to output final table
final_table <- as.data.frame(apply(summary_stats, c(2,3), FUN=mean))
row.names(final_table) <-c('Oracle', 'Unadjusted', 'PMF', 'DEF')
names(final_table) <- c('RMSE', 'All', 'Causal', 'Non-causal')
final_table <-round(final_table, 2)
print(final_table)
write.csv(final_table, file.path(OUT_PATH, "multi_cause_sim_final_table.csv"))
