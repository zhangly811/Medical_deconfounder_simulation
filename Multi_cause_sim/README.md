# Multi-Cause Simulation

## Code

`Multi_cause_sim_part1.ipynb` contains 

+ simulation of the data

+ fit of the factor models (Poisson matrix factorization and deep exponential family)


`Multi_cause_sim_part2.Rmd` contains 

+ fit of the outcome model in R


`Multi_cause_sim_part2.html` contains 

+ The output from Multi_cause_sim_part2.Rmd. (Table 3)


## System requirement
1. Python 2.7
2. Edward 1.3.5
3. tensorflow 1.5.0
4. R 3.6.0
5. rstan 2.18.2
6. rstanarm 2.18.2


## Notes
This implementation of the medical deconfounder adjusts for the mean of the posterior of reconstructed causes, rather than repeated samples from the posterior. Thus, the uncertainty of the causal effect estimates does not reflect the uncertainty in estimating the posterior of the reconstructed causes. To propagate the uncertainty of the posterior into estimating the causal effects, we recommend fitting the outcome model with repeated samples from the posterior of the reconstructed causes.

To fit a factor model, we need to fit it multiple times with different initialization and select the fit with the highest evidence lower bound (ELBO). The reason is that variational Bayes performs a non-convex optimization to fit factor models; it can get stuck in different local optima with different initializations. Operationally, it means to fit factor models multiple times with different random seeds and select the fit with the highest ELBO.

