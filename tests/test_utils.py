import numpy as np
from pymbar.utils_for_testing import eq
import pymbar.utils
import pymbar.mbar_solvers
import scipy.misc

def test_logsumexp():
    a = np.random.normal(size=(200, 500, 5))

    for axis in range(a.ndim):
        ans_ne = pymbar.mbar_solvers.logsumexp(a, axis=axis)
        ans_no_ne = pymbar.mbar_solvers.logsumexp(a, axis=axis, use_numexpr=False)
        ans_scipy = scipy.misc.logsumexp(a, axis=axis)
        eq(ans_ne, ans_no_ne)
        eq(ans_ne, ans_scipy)

def test_logsumexp_b():
    a = np.random.normal(size=(200, 500, 5))
    b = np.random.normal(size=(200, 500, 5)) ** 2.

    for axis in range(a.ndim):
        ans_ne = pymbar.mbar_solvers.logsumexp(a, b=b, axis=axis)
        ans_no_ne = pymbar.mbar_solvers.logsumexp(a, b=b, axis=axis, use_numexpr=False)
        ans_scipy = scipy.misc.logsumexp(a, b=b, axis=axis)
        eq(ans_ne, ans_no_ne)
        eq(ans_ne, ans_scipy)
