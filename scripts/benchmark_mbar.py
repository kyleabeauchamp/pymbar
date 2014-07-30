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
for sysgen in [load_100_oscillators, pymbar_datasets.load_gas_data, pymbar_datasets.load_8proteins_data]:
    name, u_kn, N_k = sysgen()
    time0 = time.time()
    mbar = pymbar.MBAR(u_kn, N_k)
    timedata.append([name, "2.0", time.time() - time0])
    time0 = time.time()
    mbar = pymbar.old_mbar.MBAR(u_kn, N_k)
    timedata.append([name, "1.0", time.time() - time0])

timedata = pd.DataFrame(timedata, columns=["name", "version", "time"])
