import numpy as np
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators
from pymbar import mbar_solvers
import scipy.optimize

n_states = 100
n_samples = 500
O_k = np.linspace(1, 5, n_states)
k_k = np.linspace(1, 3, n_states)
N_k = (np.ones(n_states) * n_samples).astype('int')

test = harmonic_oscillators.HarmonicOscillatorsTestCase(O_k, k_k)
x_n, u_kn, N_k_output = test.sample(N_k, mode='u_kn')

u_kn = np.load("/home/kyleb/src/choderalab/pymbar/gas_ukln.npz")["arr_0"]
N_k = np.load("/home/kyleb/src/choderalab/pymbar/gas_N_k.npz")["arr_0"]


mbar = MBAR(u_kn, N_k)  # Don't time first run because of weave compilation.

states_with_samples = mbar.states_with_samples
u_kn = mbar.u_kn[states_with_samples]
N_k = mbar.N_k[states_with_samples]
u_kn -= u_kn.max()

%time mbar = MBAR(u_kn, N_k)
n_states = len(N_k)

pad = lambda x: np.pad(x, (1, 0), mode='constant', constant_values=(np.log(N_k[0]),))  # Inserts zero before first element

obj = lambda x: mbar_solvers.r_obj(u_kn, N_k, pad(x))  # Objective function
grad = lambda x: mbar_solvers.r_grad(u_kn, N_k, pad(x))[1:]  # Objective function gradient

r_k = np.log(N_k)
%time results = scipy.optimize.minimize(obj, r_k[1:], jac=grad, method="L-BFGS-B", tol=1E-20)
r_k = pad(results["x"])
f_k = r_k - np.log(N_k)

mbar.f_k - f_k
