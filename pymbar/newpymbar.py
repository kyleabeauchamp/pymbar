# Copyright 2012 pymbar developers
#
# This file is part of pymbar
#
# pymbar is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymbar is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# mdtraj. If not, see http://www.gnu.org/licenses/.

"""

Notes
-----

If numerical precision / underflow is a problem, the following code could
be improved by replacing `logsumexp` with an implementation of `logsumexp`
that uses Kahan summation--as in `math.fsum()`.

"""

import numpy as np
from mdtraj.utils import ensure_type
from sklearn.utils.extmath import logsumexp

class MBAR(object):
    """Object for performing MBAR calculations.
    """
    def __init__(self, u_ki, N_k):
        """Create MBAR object from reduced energies.

        Parameters
        ----------
        u_ki : np.ndarray, shape=(n_states, n_samples)
            Reduced potential energies in states k for samples i.
        N_k : np.ndarray, shape=(n_states)
            Number of samples taken from each thermodynamic state
        """        
        self.n_states, self.n_samples = u_ki.shape

        self.u_ki = ensure_type(u_ki, np.float64, 2, 'u_ki')
        
        self.q_ki = np.exp(-self.u_ki)
        self.q_ki /= self.q_ki.max(0)  # Divide for overflow.
        
        self.N_k = ensure_type(N_k, np.float64, 1, 'N_k', (self.n_states))
        self.log_N_k = np.log(self.N_k)
        
        self.N = self.N_k.sum()
        
        self.states = self.u_ki

    def self_consistent_eqn_fast(self, f_i):
        c_i = np.exp(f_i)
        denom_n = self.q_ki.T.dot(self.N_k * c_i)
        
        num = self.q_ki.dot(denom_n ** -1.)
        
        new_f_i = 1.0 * np.log(num)

        return f_i + new_f_i

    def self_consistent_eqn(self, f_i):
      
        exp_args = self.log_N_k + f_i - self.u_ki.T
        L_n = logsumexp(exp_args, axis=1)
        
        exp_args = -L_n - self.u_ki
        q_i = logsumexp(exp_args, axis=1)
        
        return f_i + q_i

    def fixed_point_eqn(self, f_i):
        return self.self_consistent_eqn(f_i) + f_i

    def fixed_point_eqn_fast(self, f_i):
        return self.self_consistent_eqn_fast(f_i) + f_i

    def solve_iterative(self, max_iter=10000, tol=1E-12):
        """Iteratively solve the MBAR equations in 'f' space."""
        f = np.zeros(self.n_states)
        f_old = f.copy()
        for k in xrange(max_iter):
            f -= self.self_consistent_eqn(f)
            f -= f[0]
            epsilon = np.linalg.norm(f - f_old)
            if epsilon <= tol:
                break
            f_old = f.copy()
        return f


    def objective(self, f_i):
        F = self.N_k.dot(f_i)
        
        exp_arg = self.log_N_k + f_i - self.u_ki.T
        F -= logsumexp(exp_arg, axis=1).sum()
        return F * -1.
        

    def gradient(self, f_i):   
        exp_args = self.log_N_k + f_i - self.u_ki.T
        L_n = logsumexp(exp_args, axis=1)
        
        exp_args = -L_n - self.u_ki
        q_i = logsumexp(exp_args, axis=1)
        
        grad = -1.0 * self.N_k * (1 - np.exp(f_i + q_i))
        
        return grad
        
    def solve_minimize(self):
        start = np.zeros(self.n_states)
        f, final_objective, convergence_parms = scipy.optimize.fmin_l_bfgs_b(self.objective, start, self.gradient, factr=1E-2, pgtol=1E-8)
        return f


    def objective_fast(self, f_i):
        F = self.N_k.dot(f_i)

        c_i = np.exp(f_i)

        log_arg = self.q_ki.T.dot(self.N_k * c_i)
        F -= np.log(log_arg).sum()
        return F * -1.
        

    def gradient_fast(self, f_i):   
        c_i = np.exp(f_i)
        denom_n = self.q_ki.T.dot(self.N_k * c_i)
        
        num = self.q_ki.dot(denom_n ** -1.)

        grad = self.N_k * (1.0 - c_i * num)
        grad *= -1.

        return grad
