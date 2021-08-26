# Two-Cause Simulation

## Code

`Two_cause_sim.ipynb` contains code to

+ simulate the data

+ fit the factor model (probabilistic PCA)

+ fit the outcome model

+ estimate causal effects (Table 1 and Table 2)

## How to execute this scripts

Open the Two_cause_sim.ipynb in Jupiter Notebook, and run all cells.

## System requirement

1. Python 2.7
2. Edward 1.3.5
3. tensorflow 1.5.0

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
different initializations (e.g., random seeds) and then selecting the
fit with the highest evidence lower bound (ELBO). The reason is that
variational Bayes performs a non-convex optimization to fit factor
models; it can lead to different local optima with different
initializations.



