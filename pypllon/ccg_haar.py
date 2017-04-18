# encoding: utf-8
"""Routines for sampling from the Haar measures of the classical compact
groups. Algorithms taken from http://arxiv.org/abs/math-ph/0609050.

"""

from __future__ import division, print_function
import numpy as np
from scipy.linalg import qr


def orthogonal_haar(dim, rgen=np.random):
    """Returns a sample from the haar measure on O(dim)

    :param int dim: Dimension
    :param rgen: Instance of `numpy.random.RandomState` (default `np.random`)
    """
    z = rgen.randn(dim, dim)
    q, r = qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph


def unitary_haar(dim, rgen=np.random):
    """Returns a sample from the haar measure on U(dim)

    :param int dim: Dimension
    :param rgen: Instance of `numpy.random.RandomState` (default `np.random`)
    """
    z = (rgen.randn(dim, dim) + 1j * rgen.randn(dim, dim)) / np.sqrt(2.0)
    q, r = qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph
