import numpy as np
import math
import scipy.optimize


try:  # numexpr used in logsumexp when available.
    import numexpr
    HAVE_NUMEXPR = True
except ImportError:
    HAVE_NUMEXPR = False


def logsumexp(a, axis=None, b=None, use_numexpr=True):
    """Compute the log of the sum of exponentials of input elements.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : None or int or tuple of ints, optional
        Axis or axes over which the sum is taken. By default `axis` is None,
        and all elements are summed. 
    b : array-like, optional
        Scaling factor for exp(`a`) must be of the same shape as `a` or
        broadcastable to `a`.
    use_numexpr : bool, optional, default=True
        If True, use the numexpr library to speed up the calculation, which
        can give a 2-4X speedup when working with large arrays.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way. If `b` is given then ``np.log(np.sum(b*np.exp(a)))``
        is returned.

    See Also
    --------
    numpy.logaddexp, numpy.logaddexp2, scipy.misc.logsumexp

    Notes
    -----
    This is based on scipy.misc.logsumexp but with optional numexpr
    support for improved performance.
    """

    a = np.asarray(a)

    a_max = np.amax(a, axis=axis, keepdims=True)

    if a_max.ndim > 0:
        a_max[~np.isfinite(a_max)] = 0
    elif not np.isfinite(a_max):
        a_max = 0

    if b is not None:
        b = np.asarray(b)
        if use_numexpr and HAVE_NUMEXPR:
            out = np.log(numexpr.evaluate("b * exp(a - a_max)").sum(axis))
        else:
            out = np.log(np.sum(b * np.exp(a - a_max), axis=axis))
    else:
        if use_numexpr and HAVE_NUMEXPR:
            out = np.log(numexpr.evaluate("exp(a - a_max)").sum(axis))
        else:
            out = np.log(np.sum(np.exp(a - a_max), axis=axis))

    a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out


def self_consistent_update(u_kn, N_k, f_k):
    """Return an improved guess for the dimensionless free energies
    
    Parameters
    ----------
    u_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        The reduced potential energies, i.e. log unnormalized probabilities
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    f_k : np.ndarray, shape=(n_states), dtype='float'
        Updated estimate of f_k
    
    Notes
    -----
    Equation C3 in MBAR JCP paper.
    """
    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    return -1. * logsumexp(-log_denominator_n - u_kn, axis=1)


def mbar_obj_fast(Q_kn, N_k, f_k):
    """Objective function that, when minimized, solves MBAR problem.
    
    Parameters
    ----------
    Q_kn : np.ndarray, shape=(n_states, n_samples), dtype='float'
        Unnormalized probabilities.  Q_kn = exp(-u_kn)
    N_k : np.ndarray, shape=(n_states), dtype='int'
        The number of samples in each state
    f_k : np.ndarray, shape=(n_states), dtype='float'
        The reduced free energies of each state

    Returns
    -------
    obj : float
        Objective function.
    
    Notes
    -----
    This objective function is essentially a doubly-summed partition function and is
    quite sensitive to precision loss from both overflow and underflow.  For optimal
    results, u_kn should be preconditioned by subtracting out a `n` dependent
    vector.  
    
    This "fast" version works by performing matrix operations using Q_kn.
    This may have slightly reduced precision as compared to 
    """
    c_k_inv = np.exp(f_k)
    return -N_k.dot(f_k) + np.log(Q_kn.T.dot(c_k_inv * N_k)).sum()


def mbar_gradient_fast(Q_kn, N_k, f_k):
    """Gradient of mbar_obj()
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    grad : np.ndarray, dtype=float, shape=(n_states)
        Gradient of mbar_obj
    
    Notes
    -----
    This is equation C6 in the original MBAR paper.
    """

    c_k_inv = np.exp(f_k)
    denom_n = Q_kn.T.dot(N_k * c_k_inv)
    
    num = Q_kn.dot(denom_n ** -1.)

    grad = N_k * (1.0 - c_k_inv * num)
    grad *= -1.

    return grad



def mbar_gradient_and_obj_fast(Q_kn, N_k, f_k):
    """Gradient of mbar_obj()
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    grad : np.ndarray, dtype=float, shape=(n_states)
        Gradient of mbar_obj
    
    Notes
    -----
    This is equation C6 in the original MBAR paper.
    """

    c_k_inv = np.exp(f_k)
    denom_n = Q_kn.T.dot(N_k * c_k_inv)
    
    num = Q_kn.dot(denom_n ** -1.)

    grad = N_k * (1.0 - c_k_inv * num)
    grad *= -1.

    obj =  -N_k.dot(f_k) + np.log(denom_n).sum()

    return obj, grad



def mbar_obj(u_kn, N_k, f_k):
    """Objective function that, when minimized, solves MBAR problem.
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    obj : float
        Objective function.
    
    Notes
    -----
    This objective function is essentially a doubly-summed partition function and is
    quite sensitive to precision loss from both overflow and underflow.  For optimal
    results, u_kn should be preconditioned by subtracting out a `n` dependent
    vector.  Uses math.fsum for the outermost sum.
    """
    
    obj = math.fsum(logsumexp(f_k - u_kn.T, b=N_k, axis=1)) - N_k.dot(f_k)
    return obj


def mbar_gradient(u_kn, N_k, f_k):
    """Gradient of mbar_obj()
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    grad : np.ndarray, dtype=float, shape=(n_states)
        Gradient of mbar_obj
    
    Notes
    -----
    This is equation C6 in the original MBAR paper.
    """

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    W_logsum = logsumexp(-log_denominator_n - u_kn, axis=1)
    return -1 * N_k * (1.0 - np.exp(f_k + W_logsum))



def mbar_gradient_and_obj(u_kn, N_k, f_k):
    """Gradient of mbar_obj()
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    grad : np.ndarray, dtype=float, shape=(n_states)
        Gradient of mbar_obj
    
    Notes
    -----
    This is equation C6 in the original MBAR paper.
    """

    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    W_logsum = logsumexp(-log_denominator_n - u_kn, axis=1)
    grad = -1 * N_k * (1.0 - np.exp(f_k + W_logsum))
    
    obj = math.fsum(log_denominator_n) - N_k.dot(f_k)
    
    return obj, grad



def mbar_hessian(u_kn, N_k, f_k):
    """Hessian of mbar_obj.
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    H : np.ndarray, dtype=float, shape=(n_states, n_states)
        Hessian of mbar objective function.
    
    Notes
    -----
    Equation (C9) in the original MBAR paper.
    """

    W = mbar_W_nk(u_kn, N_k, f_k)
    
    H = W.T.dot(W)
    H *= N_k
    H *= N_k[:, np.newaxis]
    H -= np.diag(W.sum(0) * N_k)
    
    return -1.0 * H



def mbar_W_nk(u_kn, N_k, f_k):
    """Calculate the weight matrix.
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    W_nk : np.ndarray, dtype='float', shape=(n_samples, n_states)
        The normalized weights.
    
    Notes
    -----
    This implements equation (9) in the MBAR paper.
    """
    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    W = np.exp(f_k -u_kn.T - log_denominator_n[:, np.newaxis])
    return W



def mbar_hessian_fast(Q_kn, N_k, f_k):
    """Hessian of mbar_obj.
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    H : np.ndarray, dtype=float, shape=(n_states, n_states)
        Hessian of mbar objective function.
    
    Notes
    -----
    Equation (C9) in the original MBAR paper.
    """

    W = mbar_W_nk_fast(Q_kn, N_k, f_k)
    
    H = W.T.dot(W)
    H *= N_k
    H *= N_k[:, np.newaxis]
    H -= np.diag(W.sum(0) * N_k)
    
    return -1.0 * H



def mbar_W_nk_fast(Q_kn, N_k, f_k):
    """Calculate the weight matrix.
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.

    Returns
    -------
    W_nk : np.ndarray, dtype='float', shape=(n_samples, n_states)
        The normalized weights.
    
    Notes
    -----
    This implements equation (9) in the MBAR paper.
    """
    c_k_inv = np.exp(f_k)
    denom_n = Q_kn.T.dot(N_k * c_k_inv)
    return c_k_inv * Q_kn.T / denom_n[:, np.newaxis]


def solve_mbar(u_kn_nonzero, N_k_nonzero, f_k_nonzero, fast=False, method="hybr", tol=1E-20, options=None):
    """Solve MBAR self-consistent equations using some form of equation solver.
    
    Parameters
    ----------
    u_kn_nonzero : np.ndarray
        The reduced potential energies of the nonempty states.
    N_k_nonzero : np.ndarray
        The number of samples of the nonempty states
    f_k_Nozero : np.ndarray
        The reduced free energies of the nonempty states
    method : str, optional, default="hybr"
        The optimization routine to use.  This can be any of the methods
        available via scipy.optimize.minimize() or scipy.optimize.root().
        We find that 'hybr', which uses the MINPACK HYBR nonlinear equation solver,
        provides the fastest convergence to precise results.  'L-BFGS-B'
        is less precise and slower, but does not require calculation of
        an n^2 hessian or jacobian matrix.
    tol : float, optional, default=1E-20
        The convergance tolerance for minimize() or root()
    options: dict, optional, default=None
        Optional dictionary of algorithm-specific parameters.  See
        scipy.optimize.root or scipy.optimize.minimize for details.

    Returns
    -------
    f_k : np.ndarray
        The converged reduced free energies.
    results : dict
        Dictionary containing entire results of optimization routine, may
        be useful when debugging convergence.
    
    Notes
    -----
    This function requires that N_k_nonzero > 0--that is, you should have
    already dropped all the states for which you have no samples.
    Internally, this function works in a reduced coordinate system defined
    by subtracting off the first component of f_k and fixing that component
    to be zero.  The "hybr" method explicitly calculates the (n_states, n_states)
    Jacobian matrix, which means it could result in large memory usage
    when n_states is large.

    """    
    u_kn_nonzero = u_kn_nonzero - u_kn_nonzero.min(0)  # This should improve precision of the scalar objective function.
    # Subtract off a constant b_n from the 
    u_kn_nonzero += (logsumexp(f_k_nonzero - u_kn_nonzero.T, b=N_k_nonzero, axis=1)) - N_k_nonzero.dot(f_k_nonzero) / float(N_k_nonzero.sum())

    f_k_nonzero = f_k_nonzero - f_k_nonzero[0]  # Work with reduced dimensions with f_k[0] := 0

    pad = lambda x: np.pad(x, (1, 0), mode='constant')  # Helper function inserts zero before first element
    unpad_second_arg = lambda x, y: (x, y[1:])
    
    if not fast:
        obj = lambda x: mbar_obj(u_kn_nonzero, N_k_nonzero, pad(x))  # Objective function
        grad = lambda x: mbar_gradient(u_kn_nonzero, N_k_nonzero, pad(x))[1:]  # Objective function gradient
        grad_and_obj = lambda x: unpad_second_arg(*mbar_gradient_and_obj(u_kn_nonzero, N_k_nonzero, pad(x)))  # Objective function gradient
        hess = lambda x: mbar_hessian(u_kn_nonzero, N_k_nonzero, pad(x))[1:][:, 1:]  # Hessian of objective function        
        eqns = grad
        jac = hess
    else:
        Q_kn_nonzero = np.exp(-u_kn_nonzero)
        obj = lambda x: mbar_obj_fast(Q_kn_nonzero, N_k_nonzero, pad(x))  # Objective function
        grad = lambda x: mbar_gradient_fast(Q_kn_nonzero, N_k_nonzero, pad(x))[1:]  # Objective function gradient
        grad_and_obj = lambda x: unpad_second_arg(*mbar_gradient_and_obj_fast(Q_kn_nonzero, N_k_nonzero, pad(x)))  # Objective function gradient
        hess = lambda x: mbar_hessian_fast(Q_kn_nonzero, N_k_nonzero, pad(x))[1:][:, 1:]  # Hessian of objective function
        eqns = grad
        jac = hess


    if method in ["L-BFGS-B", "dogleg", "CG", "BFGS", "Newton-CG", "TNC", "trust-ncg", "SLSQP"]:        
        results = scipy.optimize.minimize(grad_and_obj, f_k_nonzero[1:], jac=True, hess=hess, method=method, tol=tol, options=options)
    else:
        results = scipy.optimize.root(eqns, f_k_nonzero[1:], jac=jac, method=method, tol=tol, options=options)

    f_k_nonzero = pad(results["x"])
    return f_k_nonzero, results
