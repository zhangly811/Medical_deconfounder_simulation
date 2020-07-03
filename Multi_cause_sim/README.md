# Multi-Cause Simulation

## Code

`Multi_cause_sim_part1.ipynb` contains 

+ simulation of the data

+ fit of the factor models (Poisson matrix factorization and deep exponential family)


`Multi_cause_sim_part2.Rmd` contains 

+ fit of the outcome model

+ output (Table 3)


## System requirement
1. Python 2.7
2. Edward 1.3.5
3. tensorflow 1.5.0
4. R 3.6.0
5. rstan 2.18.2
6. rstanarm 2.18.2


## Note on uncertainty 
This implementation of the medical deconfounder adjusts for the mean of reconstructed causes, rather than samples from the reconstructed causes. Thus, the uncertainty of the causal effect estimates are underestimated.

