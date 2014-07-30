import numpy as np
from pymbar import MBAR
from pymbar.testsystems import harmonic_oscillators
from pymbar import mbar_solvers
import scipy.optimize
import time

n_states = 5
n_samples = 50
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

time0 = time.time()
mbar = MBAR(u_kn, N_k)
mbar_time = time.time() - time0

n_states = len(N_k)
f_k = np.zeros(n_states)

f0 = mbar_solvers.mbar_obj(u_kn, N_k, mbar.f_k)
grad0 = np.linalg.norm(mbar_solvers.mbar_gradient(u_kn, N_k, mbar.f_k))

#methods1 = np.array(["L-BFGS-B", "dogleg", "trust-ncg", "hybr"])
methods1 = np.array(["L-BFGS-B", "hybr"])
timings = np.zeros((len(methods1)))
errors = np.zeros((len(methods1)))
gradients = np.zeros((len(methods1)))
fvals = np.zeros((len(methods1)))
raw_fvals = np.zeros((len(methods1)))

for i, method1 in enumerate(methods1):
    print(method1)
    time0 = time.time()
    f_k, results = mbar_solvers.solve_mbar(u_kn, N_k, np.zeros(n_states), method=method1)
    delta = time.time() - time0
    timings[i] = delta
    grad = mbar_solvers.mbar_gradient(u_kn, N_k, f_k)
    gradients[i] = np.linalg.norm(grad)
    fvals[i] = mbar_solvers.mbar_obj(u_kn, N_k, f_k)
    raw_fvals[i] = mbar_solvers.mbar_obj(u_kn, N_k, f_k, False)
    print(delta)
    print(fvals[i])
    print(gradients[i])


offset = np.array([0.05, 0.0])
what_to_plot = gradients
plot(timings, what_to_plot, 'o')
yscale('log')
plot([mbar_time] * 2, [what_to_plot.min(), what_to_plot.max()], 'k')
plot([timings.min(), timings.max()], [grad0] * 2, 'k')
title("%d" % n_states)
for i, method1 in enumerate(methods1):
    plt.annotate(method1, np.array([timings, what_to_plot])[:, i] + offset, fontsize='large')


