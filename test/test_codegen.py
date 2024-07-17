__copyright__ = "Copyright (C) 2017 Matt Wala"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import logging
import sys

import pytest


logger = logging.getLogger(__name__)


# {{{ test_symbolic_assignment_name_uniqueness

def test_symbolic_assignment_name_uniqueness():
    # https://gitlab.tiker.net/inducer/sumpy/issues/13
    from sumpy.assignment_collection import SymbolicAssignmentCollection

    sac = SymbolicAssignmentCollection({"s_0": 1})
    sac.assign_unique("s_", 1)
    sac.assign_unique("s_", 1)
    assert len(sac.assignments) == 3

    sac = SymbolicAssignmentCollection()
    sac.assign_unique("s_0", 1)
    sac.assign_unique("s_", 1)
    sac.assign_unique("s_", 1)

    assert len(sac.assignments) == 3

# }}}


# {{{ test_line_taylor_coeff_growth

def test_line_taylor_coeff_growth():
    # Regression test for LineTaylorLocalExpansion.
    # See https://gitlab.tiker.net/inducer/pytential/merge_requests/12
    import numpy as np

    from sumpy.expansion.local import LineTaylorLocalExpansion
    from sumpy.kernel import LaplaceKernel
    from sumpy.symbolic import SympyToPymbolicMapper, make_sym_vector

    order = 10
    expn = LineTaylorLocalExpansion(LaplaceKernel(2), order)
    avec = make_sym_vector("a", 2)
    bvec = make_sym_vector("b", 2)
    coeffs = expn.coefficients_from_source(expn.kernel, avec, bvec, rscale=1)

    sym2pymbolic = SympyToPymbolicMapper()
    coeffs_pymbolic = [sym2pymbolic(c) for c in coeffs]

    from pymbolic.mapper.flop_counter import FlopCounter
    flop_counter = FlopCounter()
    counts = [flop_counter(c) for c in coeffs_pymbolic]

    indices = np.arange(1, order + 2)
    max_order = 2
    assert np.polyfit(np.log(indices), np.log(counts), deg=1)[0] < max_order

# }}}


# You can test individual routines by typing
# $ python test_codegen.py 'test_line_taylor_coeff_growth()'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
