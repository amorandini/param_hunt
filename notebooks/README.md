These are the main notebooks used, if you are new to the project start here.

- [cl_train](cl_train.ipynb) allows the training of new NN models. The hyperparameters and architecture can be changed here, by default we use the optimized values.
- [cl_eval](cl_eval.ipynb) contains the evaluation of how good the networks are. Namely, the ROC curves for bump/ECO/EPO hunt, the coverage test for the posteriors and the explicit commands used to extract the posteriors
- [cl_comparison](cl_comparison.ipynb) presents a performance comparison (in rejecting bkg only hypothesis) between bump, ECO and EPO hunt. At the end of the notebook, we study how the performance depends on the threshold choice
- [cl_combination](cl_combination.ipynb) shows how the cleaning algorithm operates and the posterior resulting from the combination of multiple single events posteriors. We show the cases where signal and bkg are correctly identified, where the bkg is accepted (but signal is OK) and the case where signal events are rejected (but bkg is OK). A coverage test for the bad scenarios is also shown. Only specific numbers of observed events are considered here. 
