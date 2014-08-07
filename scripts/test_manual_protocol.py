import pandas as pd
import numpy as np
import pymbar
from pymbar.testsystems.pymbar_datasets import load_gas_data, load_8proteins_data
import time

def load_oscillators(n_states, n_samples):
    name = "%dx%d oscillators" % (n_states, n_samples)
    O_k = np.linspace(1, 5, n_states)
    k_k = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output

def load_exponentials(n_states, n_samples):
    name = "%dx%d exponentials" % (n_states, n_samples)
    rates = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = pymbar.testsystems.exponential_distributions.ExponentialTestCase(rates)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output



name, u_kn, N_k = load_oscillators(250, 250)
#name, u_kn, N_k = load_gas_data()
#name, u_kn, N_k = load_8proteins_data()
u_kn = u_kn[N_k > 0]
N_k = N_k[N_k > 0]
n_states = N_k.shape



time0 = time.time()
last_time = time0
f_k, results = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, np.zeros(n_states), fast=True, method="L-BFGS-B")
print(time.time() - last_time)
last_time = time.time()
f_k, results = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, f_k, fast=True, method="L-BFGS-B")
print(time.time() - last_time)
last_time = time.time()
f_k, results = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, f_k, fast=True, method="L-BFGS-B", tol=1E-25, options=dict(gtol=1E-25, ftol=1E-25))
print(time.time() - last_time)
last_time = time.time()
f_k, results = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, f_k, method="L-BFGS-B")
print(time.time() - last_time)
last_time = time.time()
f_k, results = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, f_k, method="hybr", tol=1E-25)
print(time.time() - last_time)
dt = time.time() - time0
W = pymbar.mbar_solvers.mbar_W_nk(u_kn, N_k, f_k)
wsum = np.linalg.norm(W.sum(0) - 1.0)
wdot = np.linalg.norm(W.dot(N_k) - 1.0)
grad_norm = np.linalg.norm(pymbar.mbar_solvers.mbar_gradient(u_kn, N_k, f_k))
obj = pymbar.mbar_solvers.mbar_obj(u_kn, N_k, f_k)
timedata = [[name, "cur", time.time() - time0, grad_norm, wsum, wdot]]
timedata = pd.DataFrame(timedata, columns=["name", "version", "time", "grad", "|W.sum(0) - 1|", "|W.dot(N_k) - 1|"])
print timedata.to_string(float_format=lambda x: "%.3g" % x)
