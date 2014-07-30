import pandas as pd
import numpy as np
import pymbar
import time

def load_100_oscillators():
    name = "100 oscillators"
    n_states = 100
    n_samples = 250
    O_k = np.linspace(1, 5, n_states)
    k_k = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output


timedata = []
for sysgen in [load_100_oscillators, pymbar.testsystems.pymbar_datasets.load_gas_data, pymbar.testsystems.pymbar_datasets.load_8proteins_data]:
    name, u_kn, N_k = sysgen()
    time0 = time.time()
    mbar = pymbar.MBAR(u_kn, N_k)
    wsum = np.linalg.norm(np.exp(mbar.Log_W_nk).sum(0) - 1.0)
    wdot = np.linalg.norm(np.exp(mbar.Log_W_nk).dot(N_k) - 1.0)
    grad_norm = np.linalg.norm(pymbar.mbar_solvers.mbar_gradient(u_kn, N_k, mbar.f_k))
    obj = pymbar.mbar_solvers.mbar_obj(u_kn, N_k, mbar.f_k)
    timedata.append([name, "2.0", time.time() - time0, grad_norm, wsum, wdot])
    time0 = time.time()
    mbar = pymbar.old_mbar.MBAR(u_kn, N_k)
    wsum = np.linalg.norm(np.exp(mbar.Log_W_nk).sum(0) - 1.0)
    wdot = np.linalg.norm(np.exp(mbar.Log_W_nk).dot(N_k) - 1.0)
    grad_norm = np.linalg.norm(pymbar.mbar_solvers.mbar_gradient(u_kn, N_k, mbar.f_k))
    obj = pymbar.mbar_solvers.mbar_obj(u_kn, N_k, mbar.f_k)
    timedata.append([name, "1.0", time.time() - time0, grad_norm, wsum, wdot])


timedata = pd.DataFrame(timedata, columns=["name", "version", "time", "grad", "wsum", "wdot"])
