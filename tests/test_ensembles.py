import numpy as np
import lonpl.ensembles as ens
import pytest as pt

from test_fixtures import rgen
from numpy.testing import assert_allclose


def test_normalization():
    x = ens.invecs_gaussian(10, 1, normalized=True)[0]
    assert_allclose(np.linalg.norm(x), 1.0)
    x = ens.invecs_gaussian(20, 1, normalized=False)[0]
    assert np.linalg.norm(x) > 2.0

    x = ens.invecs_recr(10, 1, normalized=True)[0]
    assert_allclose(np.linalg.norm(x), 1.0)
    x = ens.invecs_recr(20, 1, normalized=False)[0]
    assert np.linalg.norm(x) > 2.0


@pt.mark.parametrize('dim', [20])
@pt.mark.parametrize('nr_measurements', [10000])
@pt.mark.parametrize('p_erasure', [0.0, 0.5, 0.9])
def test_recr(dim, nr_measurements, p_erasure, rgen):
    invecs = ens.invecs_recr(dim, nr_measurements, p_erasure=p_erasure,
                             normalized=False)

    assert_allclose(np.mean(np.abs(invecs)), 1 - p_erasure, atol=0.05)
