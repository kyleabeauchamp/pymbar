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


#name, u_kn, N_k = load_oscillators(25)
name, u_kn, N_k = load_gas_data()
#name, u_kn, N_k = load_8proteins_data()
u_kn = u_kn[N_k > 0]
N_k = N_k[N_k > 0]
n_states = N_k.shape



time0 = time.time()
last_time = time0
f_k, results = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, np.zeros(n_states), fast=True, method="adaptive")
print(time.time() - last_time)
print(f_k)
last_time = time.time()
f_k, results = pymbar.mbar_solvers.solve_mbar(u_kn, N_k, f_k, method="adaptive")
print(time.time() - last_time)
print(f_k)
