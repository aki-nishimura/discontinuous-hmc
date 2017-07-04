# Discontinuous Hamiltonian Monte Carlo for sampling discrete parameters

Discontinuous Hamiltonian Monte Carlo (DHMC) is an extension of HMC that can sample from a piecewise-smooth target density and hence from a discrete parameter space through embedding it into a continuous space. The details of the algorithm and comparisons to other samplers can be found in "Discontinuous Hamiltonian Monte Carlo for sampling discrete parameters" by Nishimura et. al. (2017).

The 'demo' folder contains Jupyter notebooks to illustrate the use and efficiency of DHMC by applying the sampler to the Jolly-Seber model (open population capture-recapture) and PAC Bayesian inference (PAC = probabily approximately correct).
