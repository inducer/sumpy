from __future__ import division, absolute_import, print_function

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


import sys

import pyopencl as cl
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import logging
logger = logging.getLogger(__name__)


def test_kill_trivial_assignments():
    from pymbolic import var
    x, y, t0, t1, t2 = [var(s) for s in "x y t0 t1 t2".split()]

    assignments = (
        ("t0", 6),
        ("t1", -t0),
        ("t2", 6*x),
        ("nt", x**y),
        # users of trivial assignments
        ("u0", t0 + 1),
        ("u1", t1 + 1),
        ("u2", t2 + 1),
    )

    from sumpy.codegen import kill_trivial_assignments
    result = kill_trivial_assignments(
        assignments,
        retain_names=("u0", "u1", "u2"))

    from pymbolic.primitives import Sum

    def _s(*vals):
        return Sum(vals)

    assert result == [
        ('nt', x**y),
        ('u0', _s(6, 1)),
        ('u1', _s(-6, 1)),
        ('u2', _s(6*x, 1))]


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


def test_line_taylor_coeff_growth():
    # Regression test for LineTaylorLocalExpansion.
    # See https://gitlab.tiker.net/inducer/pytential/merge_requests/12
    from sumpy.kernel import LaplaceKernel
    from sumpy.expansion.local import LineTaylorLocalExpansion
    from sumpy.symbolic import make_sym_vector, SympyToPymbolicMapper

    import numpy as np

    order = 10
    expn = LineTaylorLocalExpansion(LaplaceKernel(2), order)
    avec = make_sym_vector("a", 2)
    bvec = make_sym_vector("b", 2)
    coeffs = expn.coefficients_from_source(avec, bvec, rscale=1)

    sym2pymbolic = SympyToPymbolicMapper()
    coeffs_pymbolic = [sym2pymbolic(c) for c in coeffs]

    from pymbolic.mapper.flop_counter import FlopCounter
    flop_counter = FlopCounter()
    counts = [flop_counter(c) for c in coeffs_pymbolic]

    indices = np.arange(1, order + 2)
    max_order = 2
    assert np.polyfit(np.log(indices), np.log(counts), deg=1)[0] < max_order


def test_sym_sum(ctx_getter):
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    import six
    from sumpy.assignment_collection import SymbolicAssignmentCollection
    sac = SymbolicAssignmentCollection()

    from sympy.abc import j
    from sympy import Sum
    sac.add_assignment("tmp", Sum(j, (j, 1, 10)))

    from sumpy.codegen import to_loopy_insns
    insn, additional_loop_domain = to_loopy_insns(
        six.iteritems(sac.assignments),
        retain_names=["tmp"]
    )

    from sumpy.tools import get_loopy_domain
    domain = get_loopy_domain(
        [("i", 0, 5)]
        + additional_loop_domain
    )

    import loopy as lp
    knl = lp.make_kernel(
        domain,
        insn + [lp.Assignment("a[i]", "tmp")],
        lang_version=(2018, 2)
    )

    _, result = knl(queue)
    result = result[0].get()

    import numpy as np
    ref_sol = np.ones(5, dtype=np.int32)
    ref_sol = ref_sol * 45

    assert result.shape == (5,)
    assert np.allclose(result, ref_sol)


# You can test individual routines by typing
# $ python test_fmm.py 'test_sumpy_fmm(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
