import numpy as np
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators
from pymbar import mbar_solvers
import scipy.optimize

n_states = 100
n_samples = 1000
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
f_k = np.zeros(n_states)

f_k_nonzero = np.zeros(n_states)
u_kn_nonzero = u_kn
N_k_nonzero = N_k

from pymbar.mbar_solvers import mbar_obj, mbar_grad, logsumexp,  mbar_W_kn
f_k_nonzero = f_k_nonzero - f_k_nonzero[0]  # Work with reduced dimensions with f_k[0] := 0

pad = lambda x: np.pad(x, (1, 0), mode='constant')
f = lambda x: -1.0 * mbar_obj(u_kn_nonzero, N_k_nonzero, pad(x))
df = lambda x: -1.0 * mbar_grad(u_kn_nonzero, N_k_nonzero, pad(x))[1:]

%time results = scipy.optimize.minimize(f, f_k_nonzero[1:], jac=df, method="L-BFGS-B", tol=1E-9)
print mbar.f_k[1:] - results["x"]


%prun results = scipy.optimize.minimize(f, f_k_nonzero[1:], jac=df, method="L-BFGS-B", tol=1E-9)
