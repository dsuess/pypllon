# encoding: utf-8

"""Specific functions for the Bristol experiment"""

import functools as ft

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.linalg import block_diag
from scipy.optimize import minimize


def phase_shifter(dim, alpha, mode):
    diag_elems = mode * [1] + [np.exp(1.j * alpha)] + (dim - mode - 1) * [1]
    return np.diag(diag_elems)


def single_mz(phi):
    """Returns the transfer matrix of a MZ on a 2 mode device"""
    a = np.exp(1.j * phi)
    return .5 * np.array([[-1 + a,        1.j * (1 + a)],
                          [1.j * (1 + a), 1 - a]])


def mach_zehnder(dim, phi, m):
    mz = single_mz(phi)
    return block_diag(np.eye(m), mz, np.eye(dim - m - 2))


def invec_to_phases(invec):
    """Computes the phase shifter settings for a 5-mode initization vector"""
    assert_almost_equal(1, np.linalg.norm(invec))

    # 5th mode is without a phase shifter
    x = invec / np.exp(1.j * np.angle(invec[4]))

    eta = np.zeros(4)
    eta[3] = (1 - np.abs(x[4])**2)
    eta[2] = (1 if eta[3] == 0
              else np.abs(x[3])**2 / eta[3])
    eta[1] = (1 if eta[2] == 1
              else np.abs(x[2])**2 / (eta[3] * (1 - eta[2])))
    eta[0] = (1 if eta[1] == 1
              else np.abs(x[1])**2 / (eta[3] * (1 - eta[2]) * (1 - eta[1])))

    y = 1 - 2 * eta
    y[y < -1.0] = -1.0
    y[y > 1.0] = 1.0
    phi = np.arccos(y)

    alpha = np.zeros(4, dtype=float)
    f = lambda phi: np.angle(1.j * (1 + np.exp(1.j * phi)))
    alpha[3] = np.angle(x[3]) - f(phi[2]) - np.pi
    alpha[2] = np.angle(x[2]) - np.angle(x[3]) - f(phi[1])
    alpha[1] = np.angle(x[1]) - np.angle(x[2]) - f(phi[0])
    alpha[0] = np.angle(x[0]) - np.angle(x[1]) + np.pi

    return np.mod((alpha, phi), 2 * np.pi)


def phases_to_invec(alpha, phi):
    assert len(alpha) == 4
    assert len(phi) == 4

    ps = ft.partial(phase_shifter, 6)
    mz = lambda phi, m: mach_zehnder(6, phi, m)
    elem = lambda i: np.dot(ps(alpha[i], i + 1), mz(phi[i], i + 1))
    res = np.array([0, 0, 0, 0, 1, 0])
    for i in [3, 2, 1, 0]:
        res = np.dot(elem(i), res)
    res = np.dot(np.dot(ps(0, 0), mz(np.pi, 0)), res)
    return res[1:]
