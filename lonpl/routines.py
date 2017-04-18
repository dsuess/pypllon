#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

import cvxpy as cvx
import numpy as np

from .postprocessing import rank1_approx


__all__ = ['lr_recover_nucular', 'lr_recover_l1',
           'lr_recover_l2', 'recover']


###############################################################################
#                            Optimization routines                            #
###############################################################################

def lr_recover_l1(invecs, intensities, nonneg=True, **kwargs):
    """Computes the low-rank matrix reconstruction using l1-minimisation

    .. math::

            \min_Z \sum_i \vert \langle a_i| Z | a_i \rangle - y_i \vert \\
            \mathrm{s.t.}  Z \ge 0

    where :math:`a_i` are the input vectors and :math:`y_i` are the measured
    intensities.

    For the arguments not listed see :func:`recover`

    :param bool nonneg: Enfornce the constraint Z >= 0 (default True)
    :param kwargs: Additional arguemnts passed to `cvx.Problem.solve`
    :returns: array of shape (dim, dim); Low-rank matrix approximation for
        given measurements

    """
    dim = invecs.shape[1]

    # we have to manually convert convex programm to real form since cvxpy
    # does not support complex programms
    z, mat_cons = _semidef_complex_as_real(dim) if nonneg else \
        _hermitian_as_real(dim)
    invecs_real = np.concatenate((invecs.real, invecs.imag), axis=1)

    obj = cvx.Minimize(sum(cvx.abs(cvx.quad_form(a, z) - y)
                           for a, y in zip(invecs_real, intensities)))

    prob = cvx.Problem(obj, mat_cons)
    prob.solve(**kwargs)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError("Optimization did not converge: " + prob.status)

    return z.value[:dim, :dim] + 1.j * z.value[dim:, :dim]


def lr_recover_l2(invecs, intensities, nonneg=True, **kwargs):
    """Same as :func:`lr_recover_l1`, but using l2-minimisation

    .. math::

            \min_Z \sum_i \vert \langle a_i| Z | a_i \rangle - y_i \vert^2 \\
            \mathrm{s.t.}  Z \ge 0

    """
    dim = invecs.shape[1]
    z, mat_cons = _semidef_complex_as_real(dim) if nonneg else \
        _hermitian_as_real(dim)
    invecs_real = np.concatenate((invecs.real, invecs.imag), axis=1)

    obj = cvx.Minimize(sum((cvx.quad_form(a, z) - y)**2
                           for a, y in zip(invecs_real, intensities)))

    prob = cvx.Problem(obj, mat_cons)
    prob.solve(**kwargs)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError("Optimization did not converge: " + prob.status)

    return z.value[:dim, :dim] + 1.j * z.value[dim:, :dim]


def lr_recover_nucular(invecs, intensities, noise_bound=None, nonneg=True,
                       **kwargs):
    """Same as :func:`lr_recover_l1`, but using nuclear norm-minimisation

    .. math::

            \min_Z \Vert Z \Vert_1
            \mathrm{s.t.} \sum_i \vert \langle a_i| Z | a_i \rangle - y_i \vert^2 \le \eta^2
                           Z \ge 0

    Here, :math:`\eta` denotes an additional a-priori noise bound. If
    `noise_bound` is set to `None`, we assume noiseless measurements and
    solve the following equality constrained programm instead:

    .. math::

            \min_Z \Vert Z \Vert_1
            \mathrm{s.t.} \langle a_i| Z | a_i \rangle = y_i
                           Z \ge 0

    Note that :func:`lr_recover_l1` and :func:`lr_recover_l2` do not require
    an a priori noise bound and usually perform better.

    :param float noise_bound: Bound for the column-wise noise :math:`\epsilon`:

    .. math::

            \Vert \epsilon \Vert_2 \le \eta
    """
    dim = invecs.shape[1]
    z, mat_cons = _semidef_complex_as_real(dim) if nonneg else \
        _hermitian_as_real(dim)
    invecs_real = np.concatenate((invecs.real, invecs.imag), axis=1)

    obj = cvx.Minimize(cvx.norm(z, 'nuc'))
    if noise_bound is None:
        cons = [cvx.quad_form(a, z) == y for a, y in
                zip(invecs_real, intensities)]
    else:
        cons = [sum(cvx.square(cvx.quad_form(a, z) - y)
                    for a, y in zip(invecs_real, intensities)) <= noise_bound**2]

    prob = cvx.Problem(obj, cons + mat_cons)
    prob.solve(**kwargs)

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        raise RuntimeError("Optimization did not converge: " + prob.status)

    return z.value[:dim, :dim] + 1.j * z.value[dim:, :dim]


###############################################################################
#                                   Helpers                                   #
###############################################################################

def _hermitian_as_real(dim):
    """Returns a cvxpy symmetric variable and constraints for representing
    a complex hermitian matrix represented as

    .. math::

            Z = X + iY  \iff   Z' = [X -Y; Y X]    (*)

    :param dim: Dimension of the complex matrix
    :returns: cvxpy.Variable(2 * dim, 2 * dim), List of constaints ensuring the
        structure in (*)

    """
    z = cvx.Variable(2 * dim, 2 * dim)
    return z, [z[:dim, :dim] == z[dim:, dim:], z[:dim, dim:] == -z[dim:, :dim],
               z.T == z]


def _semidef_complex_as_real(dim):
    """Returns a cvxpy Semidefinite matrix and constraints for representing
    a complex hermitian non-neg. matrix represented as

                            Z = X + iY   <==> Z' = [X -Y; Y X]    (*)

    :param dim: Dimension of the complex matrix
    :returns: cvxpy.Semidef(2 * dim), List of constaints ensuring the structure
        in (*)

    """
    z = cvx.Semidef(2 * dim)
    return z, [z[:dim, :dim] == z[dim:, dim:], z[:dim, dim:] == -z[dim:, :dim]]


###############################################################################
#                          Main recovery function                             #
###############################################################################

def recover(invecs, intensities, optim_func=lr_recover_l2, reterr=False):
    """Performs the PhaseLift reconstruction on the given data.

    :param invecs: 2D array of shape `(nr_measurements, dim)`; input vectors
    :param intensities: 2D array of shape (nr_measurements, dim); measured
        intensities. Related to the transfer matrix `M` of the linear optical
        network and the input vectors `invecs` by

        `intensities[i, j] = | <invecs^(i)| M^+ |j> |^2`

        where :math:`j` denotes the `j`-th cannonical unit vector.
    :param optim_func: Optimization function to use to recover the rows of
        the matrix. The function should take (invecs, intensities[:, j]) as
        arguments and return the (approximate) rank-1 matrix approximation
        for |M^* j><M^* j| (default :func:`optim_l2`)
    :param reterr: Should we also return error due to the rank-1 approximation
        (default False)
    :returns: 2D array of shape (dim, dim); reconstruction of the linear
        optical netork's transfer matrix from the given inputs.
        Also, if `reterr` is `True`, additionally returns array of shape (dim,)
        containing the vectorisation errors

    """
    assert len(invecs) == len(intensities)

    result = [rank1_approx(optim_func(invecs, intensity), reterr=True)
              for intensity in intensities.T]

    # conj() since we are reconstucting the conjugates of each row (see paper)
    reconstruction = np.conj([col for col, _ in result])
    errors = np.asarray([err for _, err in result])

    return (reconstruction, errors) if reterr else reconstruction
