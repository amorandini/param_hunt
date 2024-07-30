These are the packages commonly used in the study:
- [architecture](architecture.py) introduces the classifier model classes (with/without convolutional layers) used in the project
- [feat_extractor](feat_extractor.py) extracts the features from a simulation file or the posterior from a posterior file
- [hunt](hunt.py) provides functions for bumphunting and performance computations
- [localpaths](localpaths.py) introduces the link to the folders for data and posteriors (to be modified if these heavy files are in an external folder)
- [style](style.py) contains the standardized plotting style
- [utility](utility.py) introduces a function used for folder naming and one for the regularization of the log-posterior