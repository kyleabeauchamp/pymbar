import os
import numpy as np
import pymbar
from pymbar import mbar_solvers, mbar
from pymbar.testsystems import harmonic_oscillators, pymbar_datasets
from pymbar.utils import ensure_type
from pymbar.utils_for_testing import eq


def load_100_oscillators():
    name = "100 oscillators"
    n_states = 100
    n_samples = 250
    O_k = np.linspace(1, 5, n_states)
    k_k = np.linspace(1, 3, n_states)
    N_k = (np.ones(n_states) * n_samples).astype('int')
    test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
    x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')
    return name, u_kn, N_k_output

def _test(data_generator):
    name, U, N_k = data_generator()
    print(name)
    mbar = pymbar.MBAR(U, N_k)
    eq(mbar_solvers.mbar_gradient(U, N_k, mbar.f_k), np.zeros(N_k.shape), decimal=8)
    eq(np.exp(mbar.Log_W_nk).sum(0), np.ones(len(N_k)), decimal=10)
    eq(np.exp(mbar.Log_W_nk).dot(N_k), np.ones(U.shape[1]), decimal=10)
    eq(mbar_solvers.self_consistent_update(U, N_k, mbar.f_k), mbar.f_k, decimal=10)


def test_100_oscillators():
    data_generator = load_100_oscillators
    _test(data_generator)

def test_gas():
    data_generator = pymbar_datasets.load_gas_data
    _test(data_generator)

def test_8proteins():
    data_generator = pymbar_datasets.load_8proteins_data
    _test(data_generator)
