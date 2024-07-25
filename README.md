The repository contains the necessary tools to simulate and analyze beamdump simulations.

The simulations are in the folder [data](data), this contains:
- python scripts used to generate the data, especially [generate](data/generate_dataset.py). Prior is log-uniform on lifetime over mass and on mass. Current setup is for short on-axis experiment (SHiP design is implemented on my workstation but not synced yet until I'm happy with it)
- python scripts used to merge datasets: [merge](data/merge.py), [mergeh](data/mergeh.py) and [merge_post](data/merge_post.py). These are only used to bring the files in a more covnenient form for analysis (for instance they can be used to merge 5 files at different masses and lifetimes to create event_bkg_5). These scripts are used a very limited number of times so at the moment I write explicitly what I want to merge and how, this could be changed for later convenience.
- the actual datasets used:
    - [dataset](data/event_train_post_m_0.1_4.5_tm_0.01_100_c_10_35_1.25_1.25_0.csv) used to train the posterior inference 
    - [dataset00](data/event_train_00_m_0.1_4.5_tm_0.01_100_c_10_35_1.25_1.25_0.csv) and [dataset11](data/event_train_11_m_0.1_4.5_tm_0.01_100_c_10_35_1.25_1.25_0.csv) used to train bkg and sig discrimination (00 contains two incompatible events and 11 contains two compatible events)
    - [bkg5](data/event_bkg_5_m_0.1_4.5_tm_0.01_100_c_10_35_1.25_1.25_0) and [bkg13](data/event_bkg_13_m_0.1_4.5_tm_0.01_100_c_10_35_1.25_1.25_0.csv) which contain 5 and 13 bkg events respectively
    - [sig8](data/event_sig_8_m_1.0_1.0_tm_1.0_1.0_c_10_35_1.25_1.25_0.csv) which contains 8 signal events generated for a mass of 1GeV and a lifetime of 1m (so lifetime over mass is 1m/GeV)

The folder [packages](packages) contains some frequently used scripts:
- [architecture](packages/architecture.py) contains the classifier(s) architecture and it takes as arguments the various hyperparameters
- [feat_extractor](packages/feat_extractor.py) is used to extract the low level observables (and smear them) from the dataset files
- [hunt](packages/hunt.py) contains some useful functions for classifier bumphuntin and standard bumphunting, namely invariant mass reconstruction and the binning + counting algorithms with and without overlapping bins. It also performs evaluation of performance measures given histograms
- [extract_post](packages/extract_post.py) takes as input a file with observables and returns a file with posterior (evaluated with a proper network)

The folder [classifier](classifier) has the main content of the repository:
- [hyper](classifier/hyper) contains the scripts used to run the hyperparameter optmization (folders with the results are not synced since they are big and not super useful to keep synced since they are used once)
- [models](classifier/models) contain the classifier models (for fixed optimal hyperparameters)
- [counting](classifier/counting.py) is a script that extracts histogram countings while [performances](classifier/performances.py) extracts directly the mean p-value and mean log-pvalue. Classifier applied to posterior is not working right now since it relied on previous formatting... to be adapted when we include posteriors
- [cl_bump](classifier/cl_bump.ipynb) is used to train and save the classifier models. It also takes in the performances and counting files to allow comparison between different approaches (at this stage this notebook is not very commented and efficient)
