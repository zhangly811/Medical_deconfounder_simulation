# Multi-Cause Simulation

## System requirements

1. Python 2.7
2. Edward 1.3.5
3. tensorflow 1.5.0
4. R 3.6.0
5. rstan 2.19.2
6. rstanarm 2.19.2

## How to execute the scripts

1. Run `python Multi_cause_sim_part1.py`. This code

+ simulates the data

+ fits the factor models, a Poisson matrix factorization (PMF) and deep exponential family (DEF)

2. Run `Rscript Multi_cause_sim_part2.R`. This file

+ fits the outcome models (no control, oracle, control for PMF confounders, control for DEF confounders)

+ evaluates causal estimates

## Output

The files `data/*` include the simulated causes, confounders, causal
effects and outcomes.

The file `res/x_post_np_PMF.txt` and file `res/x_post_np_DEF.txt` are
the reconstructed causes from Possion factorization and the deep
exponential family, respectively.

The file `res/multi_cause_sim_final_table.txt` contains the results of
this simulation, including RMSE and the posterior coverage of causal
estimates (Table 3).

## Differences from Zhang et al. (2019)

This output differs from the results in Table 3 of Zhang et al.
(2019). Both the PMF deconfounder and DEF deconfounder outperform the
unadjusted model in terms of % coverage of true causal effect, but
here no difference between the PMF deconfounder and DEF deconfounder
is observed.

1. This implementation fits PMF with latent dimension K=175 and a
   two-layer DEF with dimensions [50, 10]. These parameters are tuned
   with respect to the specific simulated dataset, and tuning for each
   new dataset is always recommended.  Specifically, we estimate
   factor models from K=25 to K=500 in increments of 25.  We then
   select the value of K for which the predictive check is closest
   to 0.5.  (Note: This procedure does not use the outcome variables.
   It is an analysis of the observed causes.)

2. The `stan_glm.fit` function is used to fit outcome models with
   `QR=TRUE`, `iter=1e6`, and `tol_rel_obj=0.001`. The QR
   decomposition does not change the likelihood of the data but is
   recommended for computational reasons.  This implementation uses
   rstan and rstanarm 2.19.2.

## Notes

In this implementation of the medical deconfounder, the uncertainty of
the causal effect estimates may be underestimated. The reason is that
the uncertainty estimates do not account for the uncertainty in
estimating the reconstructed causes.

(For a more accurate estimate of the uncertainty, we suggest
propagating the uncertainty in the reconstructed causes
downstream. Operationally, this means drawing samples from the full
posterior of the reconstructed causes and fitting the outcome model
for each sample.)

To fit the factor model, we recommend fitting it multiple times with
different initializations (e.g., random seeds) and selecting the fit
with the highest evidence lower bound (ELBO). The reason is that
variational Bayes performs a non-convex optimization to fit factor
models; it can lead to different local optima with different
initializations.
