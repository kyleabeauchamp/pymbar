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
        can give a 2X speedup when working with large arrays.

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
    This is based on scipy.misc.logsumexp but with optional numexpr support.
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
            out = np.log(numexpr.evaluate("sum(b * exp(a - a_max), axis=%d)" % axis))
        else:
            out = np.log(np.sum(b * np.exp(a - a_max), axis=axis))
    else:
        if use_numexpr and HAVE_NUMEXPR:
            out = np.log(numexpr.evaluate("sum(exp(a - a_max), axis=%d)" % axis))
        else:
            out = np.log(np.sum(np.exp(a - a_max), axis=axis))

    a_max = np.squeeze(a_max, axis=axis)
    out += a_max

    return out


def self_consistent_update(u_kn, N_k, f_k):
    """Return an improved guess for the dimensionless free energies
    
    Parameters
    ----------
    u_kn : np.ndarray, dtype=float, shape=(n_states, n_samples)
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.
    Returns
    -------
    f_k : np.ndarray
        Updated estimate of f_k
    """
    log_denominator_n = logsumexp(f_k - u_kn.T, b=N_k, axis=1)
    return -1. * logsumexp(-log_denominator_n - u_kn, axis=1)


def self_consistent_update_fast(R_kn, N_k, f_k):
    """Return an improved guess for the dimensionless free energies
    
    Parameters
    ----------
    u_kn : np.ndarray, dtype=float, shape=(n_states, n_samples)
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.
    Returns
    -------
    f_k : np.ndarray
        Updated estimate of f_k
    """

    c_k = np.exp(f_k)
    denom_n = R_kn.T.dot(N_k * c_k)
    
    num = R_kn.dot(denom_n ** -1.)
    return -np.log(num)


def logspace_eqns(u_kn, N_k, f_k):
    """Calculate nonlinear equations whose root is the MBAR solution.
    
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
    eqns : np.ndarray
        Nonlinear equations whose root to find.
    
    Notes
    -----
    This function works in logspace and is based on Eqn. 11 in
    the original MBAR paper.
    """
    return f_k - self_consistent_update(u_kn, N_k, f_k)


def logspace_jacobian(u_kn, N_k, f_k):
    """Calculate jacobian of the nonlinear equations whose root is the MBAR solution.
    
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
    J : np.ndarray, dtype=float, shape=(n_states, n_states)
        Jacobian of nonlinear equations.
    
    Notes
    -----
    This function works in logspace and is based on Eqn. 11 in
    the original MBAR paper.  This is NOT the same as the approach shown
    in Eqn. C9 in the MBAR paper.
    """    

    W = mbar_W_nk(u_kn, N_k, f_k)
    W_sum = W.sum(0)
    
    J = W.T.dot(W)
    J *= N_k
    J -= np.diag(W_sum)
    J /= W_sum
    
    return -1.0 * J


def mbar_obj_fast(R_kn, N_k, f_k):
    """Objective function that, when minimized, solves MBAR problem.
    
    Parameters
    ----------
    u_kn : np.ndarray
        The reduced potential energies.
    N_k : np.ndarray
        The number of samples.
    f_k : np.ndarray
        The reduced free energies.
    use_fsum : bool, optional, default=True
        If True, use math.fsum to perform a stable (sorted) sum.  Although
        slower, fsum reduces underflow error and allows minimizers to achieve
        tighter convergence by a factor of 10 to 100 fold.

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
    """
    c_k = np.exp(f_k)
    return -N_k.dot(f_k) + np.log(R_kn.T.dot(c_k * N_k)).sum()

def mbar_gradient_fast(R_kn, N_k, f_k):
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

    c_k = np.exp(f_k)
    denom_n = R_kn.T.dot(N_k * c_k)
    
    num = R_kn.dot(denom_n ** -1.)

    grad = N_k * (1.0 - c_k * num)
    grad *= -1.

    return grad

def mbar_gradient_and_obj_fast(R_kn, N_k, f_k):
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

    c_k = np.exp(f_k)
    denom_n = R_kn.T.dot(N_k * c_k)
    
    num = R_kn.dot(denom_n ** -1.)

    grad = N_k * (1.0 - c_k * num)
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

def mbar_hessian_fast(R_kn, N_k, f_k):
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

    W = mbar_W_nk_fast(R_kn, N_k, f_k)
    
    H = W.T.dot(W)
    H *= N_k
    H *= N_k[:, np.newaxis]
    H -= np.diag(W.sum(0) * N_k)
    
    return -1.0 * H


def mbar_W_nk_fast(R_kn, N_k, f_k):
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
    c_k = np.exp(f_k)
    denom_n = R_kn.T.dot(N_k * c_k)
    return c_k * R_kn.T / denom_n[:, np.newaxis]


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
    #roll_by = -np.argmin(abs(f_k_nonzero - f_k_nonzero.mean()))
    #f_k_nonzero = np.roll(f_k_nonzero, roll_by)
    #N_k_nonzero = np.roll(N_k_nonzero, roll_by)
    #u_kn_nonzero = np.roll(u_kn_nonzero, roll_by, axis=0)    
    
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
    else:
        R_kn_nonzero = np.exp(-u_kn_nonzero)
        obj = lambda x: mbar_obj_fast(R_kn_nonzero, N_k_nonzero, pad(x))  # Objective function
        grad = lambda x: mbar_gradient_fast(R_kn_nonzero, N_k_nonzero, pad(x))[1:]  # Objective function gradient
        grad_and_obj = lambda x: unpad_second_arg(*mbar_gradient_and_obj_fast(R_kn_nonzero, N_k_nonzero, pad(x)))  # Objective function gradient
        hess = lambda x: mbar_hessian_fast(R_kn_nonzero, N_k_nonzero, pad(x))[1:][:, 1:]  # Hessian of objective function        

    eqns = lambda x: logspace_eqns(u_kn_nonzero, N_k_nonzero, pad(x))[1:]  # Nonlinear equations to solve via root finder
    jac = lambda x: logspace_jacobian(u_kn_nonzero, N_k_nonzero, pad(x))[1:][:, 1:]  # Jacobian of nonlinear equations

    if method in ["L-BFGS-B", "dogleg", "CG", "BFGS", "Newton-CG", "TNC", "trust-ncg", "SLSQP"]:        
        #results = scipy.optimize.minimize(obj, f_k_nonzero[1:], jac=grad, hess=hess, method=method, tol=tol, options=options)
        results = scipy.optimize.minimize(grad_and_obj, f_k_nonzero[1:], jac=True, hess=hess, method=method, tol=tol, options=options)
        success = get_actual_success(results, method)
    elif method == "fixed-point":
        eqn_fixed_point = lambda x: logspace_eqns(u_kn_nonzero, N_k_nonzero, pad(x))[1:] + x  # Nonlinear equation for fixed point iteration
        results = {}  # Fixed point doesn't have a nice dictionary output wrapper, so we make one.
        results["x"] = scipy.optimize.fixed_point(eqn_fixed_point, f_k_nonzero[1:], xtol=tol, **options)
        success = True
    elif method == "adaptive":
        newton_lambda = lambda x: newton_step(grad, hess, x)
        grad_norm_lambda = lambda x: np.linalg.norm(grad(x))
        if not fast:
            eqn_sci = lambda x: self_consistent_update(u_kn_nonzero, N_k_nonzero, pad(x))[1:]
        else:
            eqn_sci = lambda x: self_consistent_update_fast(R_kn_nonzero, N_k_nonzero, pad(x))[1:]
        
        results = {}
        results["x"] = adaptive(eqn_sci, newton_lambda, grad_norm_lambda, f_k_nonzero[1:])
        success = True
    else:
        results = scipy.optimize.root(eqns, f_k_nonzero[1:], jac=jac, method=method, tol=tol, options=options)
        success = get_actual_success(results, method)
    
    if not success:
        raise(RuntimeError("MBAR algorithm %s did not converge; died with error %d. %s." % (method, results["status"], results["message"])))
    
    f_k_nonzero = pad(results["x"])
    #f_k_nonzero = np.roll(f_k_nonzero, -roll_by)
    return f_k_nonzero, results

def newton_step(eqn_function, jac_function, f_k):
    g = eqn_function(f_k)
    J = jac_function(f_k)
    Hinvg = np.linalg.lstsq(J, g)[0]
    return f_k - Hinvg

def adaptive(self_consistent_lambda, newton_lambda, objective_lambda, f_k, max_iter=1000, grad_norm_tol=1E-13):
    nrm0 = objective_lambda(f_k)
    for i in range(max_iter):
        print(nrm0)
        f_sci = self_consistent_lambda(f_k)
        f_nr = newton_lambda(f_k)
        
        nrm_sci = objective_lambda(f_sci)
        nrm_nr = objective_lambda(f_nr)
        print("%.3d:  %.3d %.3g %.3g %s" % (i, nrm0, nrm_sci, nrm_nr, {True:"SC", False:"NR"}[nrm_sci < nrm_nr]))
    
        if nrm_sci < nrm_nr or np.isnan(nrm_nr):
            f_k = f_sci
            nrm = nrm_sci
        else:
            f_k = f_nr
            nrm = nrm_nr
    
        if nrm <= grad_norm_tol:
            print("Break due to grad_norm_tol")
            break
            
        if nrm > nrm0:
            print("nrm_increase")
            break
        nrm0 = nrm
        
    return f_k    

def get_actual_success(results, method):
    """Hack to make scipy.optimize.minimize and scipy.optimize.root return consistent success flags."""
    if method == "hybr" and results["status"] == 3:  # Limited precision error--This probably means success for our objective.
        results["success"] = True
    if method == "lm" and results["status"] == 7:  # xtol=0.000000 is too small, no further improvement in the approximate
        results["success"] = True
    if method == "L-BFGS-B" and results["status"] == 2:  # ABNORMAL_TERMINATION_IN_LNSRCH.  This probably means success but precision issues.
        results["success"] = True
    if method == "trust-ncg" and results["status"] == 2:  # A bad approximation caused failure to predict improvement..
        results["success"] = True
    if method == "broyden2" and results["status"] == 2:  # The maximum number of iterations allowed has been reached.
        results["success"] = True
    return results["success"]
