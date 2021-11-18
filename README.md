# swyft-CMB

This repository allows one to reproduce computations presented in "Fast and Credible Likelihood-Free Cosmology with Truncated Marginal Neural Ratio Estimation," https://arxiv.org/abs/2111.08030. Here we use `swyft` (available at https://github.com/undark-lab-swyft). 
This repository uses and older commit of `swyft` -- this will be updated (to include parallel simulation and new interface) in the future.
Install this version by entering the directory and running `pip install -e .`.
The main simulator relies on the publicly available Boltzmann code `class` https://github.com/lesgourg/class_public/. See that repo for installation details.

We include a demonstration notebook in `notebooks/demo-TTTEEE.ipynb`, which performs inference for the
forecasting CMB simulator and demonstrates the empirical credibility test. Accompanying this is a small datastore (roughly 10,000 samples) with CMB power spectra and BAO measurements.
This is about 500 MB.

The `HiLLiPoP` likelihood is available at https://github.com/planck-npipe/hillipop. It is quite large.
Detailed instructions for modifying that code and defining the corresponding simulator will appear here soon.
