# Two-Cause Simulation

## Code

`Two_cause_sim.ipynb` contains 

+ simulation of the data

+ fit of the factor model (probabilistic PCA)

+ fit of the outcome model

+ output of the causal effect estimates (Table 1 and Table 2)


## System requirement
1. Python 2.7
2. Edward 1.3.5
3. tensorflow 1.5.0


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



