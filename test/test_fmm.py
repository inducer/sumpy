from __future__ import division

__copyright__ = "Copyright (C) 2013 Andreas Kloeckner"

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
import numpy as np
import numpy.linalg as la
import pyopencl as cl
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
from sumpy.kernel import LaplaceKernel, HelmholtzKernel
import pytest

import logging
logger = logging.getLogger(__name__)


try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


@pytest.mark.parametrize("knl", [
    LaplaceKernel(2),
    LaplaceKernel(3),
    HelmholtzKernel(2),
    HelmholtzKernel(3),
    ])
def test_sumpy_fmm(ctx_getter, knl):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    nsources = 500
    ntargets = 50
    dtype = np.float64

    from boxtree.tools import (
            make_normal_particle_array as p_normal)

    sources = p_normal(queue, nsources, knl.dim, dtype, seed=15)
    if 1:
        offset = np.zeros(knl.dim)
        offset[0] = 0.1

        targets = (
                p_normal(queue, ntargets, knl.dim, dtype, seed=18)
                + offset)

        del offset
    else:
        from sumpy.visualization import FieldPlotter
        fp = FieldPlotter(np.array([0.5, 0]), extent=3, npoints=200)
        from pytools.obj_array import make_obj_array
        targets = make_obj_array(
                [fp.points[i] for i in xrange(knl.dim)])

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    # {{{ plot tree

    if 0:
        host_tree = tree.get()
        host_trav = trav.get()

        if 1:
            print "src_box", host_tree.find_box_nr_for_source(403)
            print "tgt_box", host_tree.find_box_nr_for_target(28)
            print list(host_trav.target_or_target_parent_boxes).index(37)
            print host_trav.get_box_list("sep_bigger", 22)

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(host_tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.set_bounding_box()
        plotter.draw_box_numbers()

        import matplotlib.pyplot as pt
        pt.show()

    # }}}

    from pyopencl.clrandom import RanluxGenerator
    rng = RanluxGenerator(queue, seed=20)
    weights = rng.uniform(queue, nsources, dtype=np.float64)

    logger.info("computing direct (reference) result")

    from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
    from sumpy.expansion.local import VolumeTaylorLocalExpansion

    from pytools.convergence import PConvergenceVerifier

    pconv_verifier = PConvergenceVerifier()

    extra_kwargs = {}
    dtype = np.float64
    order_values = [1, 2, 3]
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05
        dtype = np.complex128

        if knl.dim == 3:
            order_values = [1, 2]

    for order in order_values:
        mpole_expn = VolumeTaylorMultipoleExpansion(knl, order)
        local_expn = VolumeTaylorLocalExpansion(knl, order)
        out_kernels = [knl]

        from sumpy.fmm import SumpyExpansionWranglerCodeContainer
        wcc = SumpyExpansionWranglerCodeContainer(
                ctx, mpole_expn, local_expn, out_kernels)
        wrangler = wcc.get_wrangler(queue, tree, dtype,
                kernel_extra_kwargs=extra_kwargs)

        from boxtree.fmm import drive_fmm

        pot, = drive_fmm(trav, wrangler, weights)

        from sumpy import P2P
        p2p = P2P(ctx, out_kernels, exclude_self=False)
        evt, (ref_pot,) = p2p(queue, targets, sources, (weights,),
                **extra_kwargs)

        pot = pot.get()
        ref_pot = ref_pot.get()

        rel_err = la.norm(pot - ref_pot) / la.norm(ref_pot)
        logger.info("order %d -> relative l2 error: %g" % (order, rel_err))

        pconv_verifier.add_data_point(order, rel_err)

    print pconv_verifier
    pconv_verifier()


# You can test individual routines by typing
# $ python test_fmm.py 'test_sumpy_fmm(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
