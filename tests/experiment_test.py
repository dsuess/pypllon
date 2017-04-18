import pytest as pt

import pypllon as lp
import pypllon.experiment as exp


@pt.mark.parametrize('ensemble', [lp.invecs_gaussian, lp.invecs_recr])
@pt.mark.parametrize('samples', [100])
def test_conversion(ensemble, samples, rgen):
    invecs = ensemble(5, samples, rgen=rgen)
    for invec in invecs:
        invec_ = exp.phases_to_invec(*exp.invec_to_phases(invec))
        assert lp.best_invec_phase(invec, invec_)[1] < 1e-10
