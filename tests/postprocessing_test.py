import numpy as np
import pytest as pt
from numpy.testing import assert_allclose

import pypllon as lp
from pypllon.postprocessing import rand_angles
from pypllon.ccg_haar import unitary_haar


@pt.mark.parametrize('mode', ['cols_by_first', 'cols_by_max',
                              'rows_by_first', 'rows_by_max'])
def test_fix_phases(mode, rgen):
    target = unitary_haar(10)
    phases = np.exp(1.j * rand_angles(len(target), rgen=rgen))
    new = np.dot(target, np.diag(phases)) if mode.startswith('cols') \
        else np.dot(np.diag(phases), target)
    target_, new_ = lp.fix_phases(mode, target, new)
    assert_allclose(target_, new_)

    if mode == 'cols_by_first':
        assert_allclose(new_[0].imag, 0, atol=1e-15)
    elif mode == 'rows_by_first':
        assert_allclose(new_[:, 0].imag, 0, atol=1e-15)


def test_fix_phases_identity(rgen):
    target = np.eye(10)
    phases = np.exp(1.j * rand_angles(len(target), rgen=rgen))
    new = np.dot(np.diag(phases), target)
    target_, new_ = lp.fix_phases('rows_by_max', target, new)
    assert_allclose(target_, new_)


@pt.mark.parametrize('dim', [5, 10, 15])
@pt.mark.parametrize('ensemble', [lp.invecs_gaussian, lp.invecs_recr])
def test_best_invec_phase(dim, ensemble, rgen):
    x, *_ = ensemble(dim, 1, normalized=True, rgen=rgen)
    y = np.exp(1.j * rand_angles()) * x
    y_, dist = lp.postprocessing.best_invec_phase(x, y)
    assert_allclose(dist, 0, atol=1e-5)
    assert_allclose(y_, x, atol=1e-5)
