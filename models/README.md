Different ML models (saved both in .tf and .keras format)

- model_post_(uncertainty)_0 are the models used to extract the posteriors
- model_01_(uncertainty)_(i_model) are the ECO hunt models
- model_post_01_(uncertainty)_(i_model) are the EPO hunt models

Here we have used as placeholder:

- (uncertainty) can be either "small" or "large" for small and large uncertainties respectively.
- (i_model) is a model counter from 0 to 4 