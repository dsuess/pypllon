#!/usr/bin/env python
# encoding: utf-8
from __future__ import absolute_import, division, print_function

from functools import wraps

import numpy as np
from numpy.linalg import norm


__all__ = ['invecs_gaussian', 'invecs_recr']


def normalizeable(genfunc, default=True):
    """Adds a keyword function to genfunc which allows normalization of the
    result (abs-sum over last axis yields 1)
    """

    @wraps(genfunc)
    def inner(*args, **kwargs):
        try:
            normalized = kwargs.pop('normalized')
        except KeyError:
            normalized = default

        result = genfunc(*args, **kwargs)
        return result / norm(result, axis=-1, keepdims=True) if normalized \
            else result

    return inner


@normalizeable
def invecs_gaussian(dim, length, rgen=np.random):
    """Generates complex Gaussian random vectors with iid, N(0,1) + iN(0,1)
    components

    :param dim: Dimension of the vectors (# of components)
    :param length: Number of vectors
    :param rgen: Instance of `numpy.random.RandomState` (default: `np.random`)
    :returns: (length,dim) numpy array

    """
    return rgen.randn(length, dim) + 1.j * rgen.randn(length, dim)


@normalizeable
def invecs_recr(dim, nr_mm, p_erasure=1 / 2, rgen=np.random):
    """Generates random preparation vectors with iid components distributed
    accorgening to a random-erased complex Rademacher (RECR) distribution,
    that is x_i = a_i * phi_i, where

        a_i ~ Bernoulli(p_erasure), phi_i ~ Uniform({1, -1, i, -i})

    :param dim: Dimension of the vectors (# of components)
    :param length: Number of vectors
    :param p_erasure: Erasure probability
    :returns: (length,dim) numpy array

    """
    pvecs = []
    while len(pvecs) < nr_mm:
        phase = rgen.choice([1, -1, 1j, -1j], size=dim)
        amplitude = rgen.choice([0, 1], size=dim, p=[p_erasure, 1 - p_erasure])
        # this penalizes the choice of zero for small values of dim
        if np.sum(amplitude) > 0:
            pvecs.append(phase * amplitude)
    return np.reshape(pvecs, (nr_mm, dim))
