import pandas as pd
import numpy as np
import pymbar
from pymbar.testsystems.pymbar_datasets import load_gas_data, load_8proteins_data
import time

def load_oscillators(n_states):
    name = "%d oscillators" % n_states
    n_samples = 250
    O_k = np.linspace(1, 5, n_states)
    k_k = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output

def load_exponentials(n_states):
    name = "%d exponentials" % n_states
    n_samples = 250
    rates = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output


solver_protocol = [dict(method="L-BFGS-B", fast=True), dict(method="L-BFGS-B", fast=True), dict(method="L-BFGS-B"), dict(method="adaptive")]
#solver_protocol = None
mbar_gens = {"new":lambda u_kn, N_k: pymbar.MBAR(u_kn, N_k, solver_protocol=solver_protocol)}
#mbar_gens = {"old":lambda u_kn, N_k: pymbar.old_mbar.MBAR(u_kn, N_k)}
#mbar_gens = {"new":lambda u_kn, N_k: pymbar.MBAR(u_kn, N_k), "old":lambda u_kn, N_k: pymbar.old_mbar.MBAR(u_kn, N_k)}
systems = [lambda : load_exponentials(50), lambda : load_exponentials(150), lambda : load_oscillators(50), lambda : load_oscillators(150), load_gas_data, load_8proteins_data]

timedata = []
for version, mbar_gen in mbar_gens.items():
    for sysgen in systems:
        name, u_kn, N_k = sysgen()
        time0 = time.time()
        mbar = mbar_gen(u_kn, N_k)
        wsum = np.linalg.norm(np.exp(mbar.Log_W_nk).sum(0) - 1.0)
        wdot = np.linalg.norm(np.exp(mbar.Log_W_nk).dot(N_k) - 1.0)
        grad_norm = np.linalg.norm(pymbar.mbar_solvers.mbar_gradient(u_kn, N_k, mbar.f_k))
        obj = pymbar.mbar_solvers.mbar_obj(u_kn, N_k, mbar.f_k)
        timedata.append([name, version, time.time() - time0, grad_norm, wsum, wdot])


timedata = pd.DataFrame(timedata, columns=["name", "version", "time", "grad", "|W.sum(0) - 1|", "|W.dot(N_k) - 1|"])
print timedata.to_string(float_format=lambda x: "%.3g" % x)
