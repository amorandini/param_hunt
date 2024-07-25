import numpy as np
import random

import os
import scipy.stats as sts
import pandas as pd

import sys
import pickle

sys.path.append('./../packages')
data_path = "../data/"
post_path = "../data/"
from hunt import perf_pval, count_from_df

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nsteps', type=int, help="number of mu space steps", default=5)
parser.add_argument('--mfixed', type=float, help="mass of the signal", default=1.)
parser.add_argument('--smear', type=int, help="small or large smearing", default=0)


args = parser.parse_args()

Nsamples = 250000
N_steps= args.nsteps
mfixed = args.mfixed
smear = args.smear

if smear ==0:
    sigs, labsm = [0.001,0.01,0.005, 0.005 ], "small" # small smearing case
elif smear == 1:
    sigs, labsm = [0.001,0.05,0.01,  0.01  ], "large"

# increase number of spaces for better visualization
mubspace = np.linspace(0.1, 6, N_steps)
musspace = np.linspace(0.1, 4, N_steps)

def TS_par(dfc_opt, mub,mus):
    nbkgsig = np.array([sts.poisson(mub).rvs(Nsamples), sts.poisson(mus).rvs(Nsamples)])
    nnurbkg = np.array([sts.poisson(mub).rvs(Nsamples), np.zeros(Nsamples)])

    countB, countS = count_from_df(dfc_opt, nnurbkg, nbkgsig)

    return([mub, mus, perf_pval(countB, countS).meanp(), perf_pval(countB, countS).meanlogp(), perf_pval(countB, countS).meansigma()])

perf_bump = []
with open("../performances/counts/count_bump_"+labsm+"_m_"+str(mfixed)+".pickle", "rb") as fp:  
    countsS = pickle.load(fp)
dfc = pd.DataFrame(countsS, columns=["nbkg", "nsig", "nbins", "counts"])
for opt_bin in range(4, 50):
    dfc_opt = dfc.loc[dfc["nbins"]==opt_bin]
    TS_bump = np.array((Parallel(n_jobs=num_cores)(delayed(TS_par)(dfc_opt, mub, mus) for mub in mubspace for mus in musspace )))
    perf_bump.append([opt_bin, np.mean(TS_bump[:,3]), np.mean(TS_bump[:,4])])

perf_ECO = []
with open("../performances/counts/count_cl_"+labsm+"_m_"+str(mfixed)+".pickle", "rb") as fp:  
    countsS = pickle.load(fp)
dfc = pd.DataFrame(countsS, columns=["nbkg", "nsig", "thresh", "counts"])
for opt_thro in np.linspace(0.4,0.96, 15):
    dfc_opt = dfc.loc[round(dfc["thresh"],2)==round(opt_thro,2)]
    TS_ECO = np.array((Parallel(n_jobs=num_cores)(delayed(TS_par)(dfc_opt, mub, mus) for mub in mubspace for mus in musspace )))
    perf_ECO.append([opt_thro, np.mean(TS_ECO[:,3]), np.mean(TS_ECO[:,4])])

perf_EPO = []
with open("../performances/counts/count_cl_post_"+labsm+"_m_"+str(mfixed)+".pickle", "rb") as fp:  
    countsS = pickle.load(fp)
dfc = pd.DataFrame(countsS, columns=["nbkg", "nsig", "thresh", "counts"])
for opt_thrp in np.linspace(0.4,0.96, 15):
    dfc_opt = dfc.loc[round(dfc["thresh"],2)==round(opt_thrp,2)]
    TS_EPO = np.array((Parallel(n_jobs=num_cores)(delayed(TS_par)(dfc_opt, mub, mus) for mub in mubspace for mus in musspace )))
    perf_EPO.append([opt_thrp, np.mean(TS_EPO[:,3]), np.mean(TS_EPO[:,4])])

np.savetxt("../performances/TS/perf_bump_"+labsm+"_m_"+str(mfixed)+".csv", perf_bump)
np.savetxt("../performances/TS/perf_cl_"+labsm+"_m_"+str(mfixed)+".csv", perf_ECO)
np.savetxt("../performances/TS/perf_cl_post_"+labsm+"_m_"+str(mfixed)+".csv", perf_EPO)
