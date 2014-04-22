# Copyright 2013 mdtraj developers
#
# This file is part of mdtraj
#
# mdtraj is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# mdtraj is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# mdtraj. If not, see http://www.gnu.org/licenses/.

##############################################################################
# imports
##############################################################################

import itertools
import warnings

import numpy as np
import pandas as pd

##############################################################################
# functions / classes
##############################################################################


class TypeCastPerformanceWarning(RuntimeWarning):
    pass




def _logsum(a_n):
    """Compute the log of a sum of exponentiated terms exp(a_n) in a numerically-stable manner.

    Parameters
    ----------
    a_n : np.ndarray, shape=(n_samples)
        a_n[n] is the nth exponential argument
        
    Returns
    -------
    a_n : np.ndarray, shape=(n_samples)
        a_n[n] is the nth exponential argument
        
    Notes
    -----

    _logsum a_n = max_arg + \log \sum_{n=1}^N \exp[a_n - max_arg]

    where max_arg = max_n a_n.  This is mathematically (but not numerically) equivalent to

    _logsum a_n = \log \sum_{n=1}^N \exp[a_n]



    Example
    -------
    >>> a_n = np.array([0.0, 1.0, 1.2], np.float64)
    >>> print '%.3e' % _logsum(a_n)
    1.951e+00
    """

    # Compute the maximum argument.
    max_log_term = np.max(a_n)

    # Compute the reduced terms.
    terms = np.exp(a_n - max_log_term)

    # Compute the log sum.
    log_sum = np.log(np.sum(terms)) + max_log_term
        
    return log_sum

#=============================================================================================
# Exception classes
#=============================================================================================

class ParameterError(Exception):
    """
    An error in the input parameters has been detected.

    """
    pass

class ConvergenceError(Exception):
    """
    Convergence could not be achieved.

    """
    pass

class BoundsError(Exception):
    """
    Could not determine bounds on free energy

    """
    pass
