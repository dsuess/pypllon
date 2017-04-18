# encoding: utf-8

from __future__ import division, print_function

from scipy.linalg import eigh
from scipy.optimize import minimize

try:
    import autograd.numpy as np
    from autograd import grad
except ImportError:
    import numpy as np
    from warnings import warn
    warn("Package autograd not found. Only limited functionality available.")

__all__ = ['rank1_approx', 'fix_phases']


def rank1_approx(mat, reterr=False):
    """Returns the best rank 1 approximation to `mat`, i.e. the singular
    vector corresponding to the largest singular value.

    :param mat: Hermitian matrix as `numpy.ndarray` with shape `(dim, dim)`.
    :param reterr: Return the sum of the truncated singular values
    :returns: Largest singular vector. If reterr==True, we also return the
        error given by the sum of the discarded singular values

    """
    vals, vecs = eigh(mat)
    if reterr:
        return np.sqrt(vals[-1]) * vecs[:, -1], np.sum(vals[:-1]) / vals[-1]
    else:
        return np.sqrt(vals[-1]) * vecs[:, -1]


def fix_phases(mode, *args):
    """Returns the row (or column) phase fixed versions of tmat depending on
    the mode.

    :param mode:
        'rows_by_first': the first column is made real
        'cols_by_first': the first row is made real
        'rows_by_max': the maximum-modulus element of each row is made real
        'cols_by_max': the maximum-modulus element of each column is made real
    :param *args: Transfer matrices to be phase-fixed. For the max.
        element-modes the first one is taken for choosing the max. elements
        and the other ones are phase-fixed to the same elements.
    :returns: List of normalized version of tmats

    TODO More pythonic
    """
    if mode == 'cols_by_first':
        sel = np.zeros(args[0].shape[0], dtype=int)
    elif mode == 'rows_by_first':
        return [x.T for x in fix_phases('cols_by_first', *(x.T for x in args))]

    elif mode == 'cols_by_max':
        sel = np.argmax(np.abs(args[0]), axis=0)
    elif mode == 'rows_by_max':
        return [x.T for x in fix_phases('cols_by_max', *(x.T for x in args))]

    else:
        raise ValueError("{} is not a valid mode.".format(mode))

    phase = lambda x: x / np.abs(x)
    return [tmat / phase(sel.choose(tmat))[None, :] for tmat in args]


def rand_angles(*args, rgen=np.random):
    """Returns uniform sampled random angles in [0, 2pi] of given shape"""
    return 2 * np.pi * rgen.uniform(size=args)


def best_normalization(A, B, **kwargs):
    """Finds the angles `phi` and `psi` that minimize the Frobenius distance
    between A and B', where

        B' = diag(exp(i phi)) @ B @ diag(exp(i psi))

    :param A: @todo
    :param B: @todo
    :returns: @todo

    """
    d = len(A)
    diagp = lambda phi: np.diag(np.exp(1.j * phi))
    B_ = lambda phi, psi: np.dot(diagp(phi), np.dot(B, diagp(psi)))
    norm_sq = lambda x: np.real(np.dot(np.conj(x.ravel()), x.ravel()))
    cost = lambda x: norm_sq(A - B_(x[:d], x[d:]))

    init_angles = np.random.uniform(0.0, 2 * np.pi, 2 * d)
    result = minimize(cost, init_angles, jac=grad(cost), **kwargs)
    phi, psi = result['x'][:d], result['x'][d:]

    # Normalization 1 / np.sqrt(2) due to real embedding
    return B_(phi, psi), result['fun'] / np.sqrt(2)


def phasefree_dist(x, y, rety=False):
    norm_sq = lambda x: np.real(np.dot(np.conj(x.ravel()), x.ravel()))
    cost = lambda phi: norm_sq(x - np.exp(1.j * phi) * y)
    result = minimize(cost, np.random.uniform(0, 2 * np.pi), jac=grad(cost))
    y_ = np.exp(1.j * result['x']) * y
    return result['fun'] if not rety else result['fun'], y_
