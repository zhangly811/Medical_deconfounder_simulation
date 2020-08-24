# Multi-Cause Simulation

## System requirement
1. Python 2.7
2. Edward 1.3.5
3. tensorflow 1.5.0
4. R 3.6.0
5. rstan 2.19.2
6. rstanarm 2.19.2

## How to execute the scripts

1. Run `python Multi_cause_sim_part1.py`. This file contains 

+ simulation of the data

+ fit of the factor models (Poisson matrix factorization (PMF) and deep exponential family (DEF))


2. Run `Rscript Multi_cause_sim_part2.R`. This file contains 

+ fit of the outcome models (no control, oracle, control for PMF confounders, control for DEF confounders)

+ evaluation of causal estimates


## Output

The files `data/*` include the simulated causes, confounders, causal effects and outcomes. 

The file `res/x_post_np_PMF.txt` and file `res/x_post_np_DEF.txt` are the reconstructed causes from possion factorization and deep exponential family respectively.

The file `res/multi_cause_sim_final_table.txt` contains the results of this simulation, including RMSE and the posterior coverage of causal estimates. (Table 3)


## Differences from Zhang et al. (2019)

This output differs from the results in Table 3 of Zhang et al.
(2019).Both PMF deconfounder and DEF deconfounder outperform the unadjusted model in terms of % coverage of true causal effect, but no difference between PMF deconfounder and DEF deconfounder is observed.

1. This implementation fits PMF with latent dimension K=175 and a two-layer DEF with [50, 10] latent variables. These parameters are tuned with respect to the specific simulated dataset, and tuning for each new dataset is always recommended.

1. The `stan_glm.fit` function is used to fit outcome models with `QR=TRUE`, `iter=1e6`, and `tol_rel_obj=0.001`. QR decomposition does not change the likelihood of the data but are recommended for computational reasons when there are multiple predictors. 
This implementation uses rstan and rstanarm 2.19.2.

## Notes
In this implementation of the medical deconfounder, the uncertainty of
the causal effect estimates may be underestimated. The reason is that
the uncertainty estimates do not account for the uncertainty in
estimating the reconstructed causes. For a more accurate estimate of
the uncertainty, we recommend propagating the uncertainty in the
reconstructed causes downstream. Operationally, it means we should
draw samples from the full posterior of the reconstructed causes, and
fit the outcome model for each sample.

To fit a factor model, we need to fit it multiple times with different initialization and select the fit with the highest evidence lower bound (ELBO). The reason is that variational Bayes performs a non-convex optimization to fit factor models; it can get stuck in different local optima with different initializations. Operationally, it means to fit factor models multiple times with different random seeds and select the fit with the highest ELBO.

