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


## Note on uncertainty 
This implementation of the medical deconfounder adjusts for the mean of reconstructed causes, rather than samples from the reconstructed causes. Thus, the uncertainty of the causal effect estimates are underestimated.

