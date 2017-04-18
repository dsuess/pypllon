import numpy as np
import pytest as pt
from numpy.testing import assert_allclose
from scipy.linalg import dft

import lonpl as lp
from lonpl.ccg_haar import unitary_haar


def target_unitaries():
    targets = [np.eye(5), dft(6, scale='sqrtn'), np.eye(3)[::-1]]
    targets += [unitary_haar(4, rgen=np.random.RandomState(12345))]
    return targets


@pt.mark.parametrize('target', target_unitaries())
@pt.mark.parametrize('optim_func', [lp.lr_recover_l1,
                                    lp.lr_recover_l2,
                                    lp.lr_recover_nucular])
@pt.mark.parametrize('ensemble', [lp.invecs_gaussian, lp.invecs_recr])
def test_recover_noiseless(target, optim_func, ensemble, rgen):
    dim = len(target)
    invecs = ensemble(dim, 6 * dim, normalized=True, rgen=rgen)
    intensities = np.abs(np.tensordot(invecs, target, axes=(1, 1)))**2
    recovery = lp.recover(invecs, intensities, optim_func=optim_func)
    target, recovery = lp.fix_phases('rows_by_max', target, recovery)
    assert_allclose(target, recovery, atol=1e-3)
