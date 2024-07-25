In this folder we provide the code for the simulations:
- [ALP_decay](ALP_decay.py) and [generate_dataset](generate_dataset.py) are the scripts for the simulations.
- the merge*.py files are used to merge different simulations outputs. [mergeh](mergeh.py) combines datasets horizontally (merge events), [merge](merge.py) combines datasets vertically (merge samples) and [merge_post](merge_post.py) combines the posterior files. 
- the files event_train contain the training samples. train_00 are bkg events for ECO/EPO, train_11 are sig events for ECO/EPO and train_post are the ones for training the posterior extractor.
- event_bkg are test samples for bkg events, event_sig are test samples for the sig (for three different benchmark masse 0.2GeV, 1GeV and 4 GeV).
- post_bkg_13 and post_sig_8 are the only posterior files provided for the case of large uncertainty and a signal mass of 1GeV. Other posterior files need to be obtained from the provided data files.