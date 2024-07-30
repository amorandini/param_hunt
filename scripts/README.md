The folder contains the scripts used to automate the most intensive parts. Most of the results are saved in the performances folder.

- [Cij_exp_obs](Cij_exp_obs.py) and [Cij_exp_post](Cij_exp_post.py) export the compatibility measures for ECO and EPO respectively.
- counting_(bump/ML) returns the TS distribution for bump hunting and ML approaches (both ECO and EPO)
- [extract_post](extract_post.py) takes an observable file as input and returns the ML-inferred posterior
- [performance_full](performance_full.py) returns the performances for fixed threshold and varying background/signal strength
- [plot_sensitivities](plot_sensitivities.py) plots the sensitivity and sensitivity improvement figures. 