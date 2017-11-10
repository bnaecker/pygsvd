import itertools

import gsvd
import numpy as np
import pytest

def test_gsvd_computation():
    '''Test the actual computation performed by the GSVD.
    This really should not fail, as the numbers are directly returned
    by the LAPACK routine. The possible exception is `R`, which is
    stored in the input arrays when the routine returns.
    '''
    matrices = tuple(map(np.loadtxt, 
            ('{}.txt'.format(x) for x in 
            ('a', 'b', 'c', 's', 'r', 'u', 'v', 'q'))))
    outputs = gsvd.gsvd(matrices[0], matrices[1], extras='uvq')
    for xin, xout in zip(matrices[2:], outputs):
        assert np.allclose(xin, xout)

def test_same_columns():
    '''Verify that the method raises a ValueError when the inputs
    have a different number of columns.
    '''
    with pytest.raises(ValueError):
        gsvd.gsvd(np.arange(10).reshape(5, 2), np.arange(10).reshape(2, 5))

def test_dimensions():
    '''Verify that the input succeeds with 1D arrays (which are promoted
    to 2D), and raises a ValueError for 3+D arrays.
    '''
    x = np.arange(10)
    gsvd.gsvd(x, x)
    with pytest.raises(ValueError):
        gsvd.gsvd(x.reshape(5, 2, 1), x.reshape(5, 2, 1))

def test_return_extras():
    '''Verify that the extra arrays are returned as expected.'''
    names = ('a', 'b', 'c', 's', 'r', 'u', 'v', 'q')
    matrices = dict(zip(names, 
            map(np.loadtxt, ('{}.txt'.format(name) for name in names))))
    extras = 'uvq'

    # Make all combinations of 'uvq' of length 0 through 3, inclusive
    for combo_length in range(len(extras) + 1):
        combinations = list(itertools.combinations(extras, combo_length))

        for combo in combinations:
            # Join the combination letters to make the `extra` kwarg
            ex = ''.join(combo)

            # Compute GSVD including the extras, assigned to a dict
            out = dict(zip(('c', 's', 'r') + combo, 
                    gsvd.gsvd(matrices['a'], matrices['b'], extras=ex)))

            # Compare each extra output to the expected
            for each in combo:
                assert np.allclose(out[each], matrices[each])
