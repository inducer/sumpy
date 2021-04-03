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
from sumpy.kernel import LaplaceKernel, HelmholtzKernel, YukawaKernel
from sumpy.expansion.multipole import (
    VolumeTaylorMultipoleExpansion,
    H2DMultipoleExpansion, Y2DMultipoleExpansion,
    LaplaceConformingVolumeTaylorMultipoleExpansion,
    HelmholtzConformingVolumeTaylorMultipoleExpansion)
from sumpy.expansion.local import (
    VolumeTaylorLocalExpansion,
    H2DLocalExpansion, Y2DLocalExpansion,
    LaplaceConformingVolumeTaylorLocalExpansion,
    HelmholtzConformingVolumeTaylorLocalExpansion)

import pytest

import logging
logger = logging.getLogger(__name__)


try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


@pytest.mark.parametrize(
    "knl, local_expn_class, mpole_expn_class, optimized_m2l, use_fft",
    [
        (LaplaceKernel(2), VolumeTaylorLocalExpansion,
            VolumeTaylorMultipoleExpansion, False, False),
        (LaplaceKernel(2), VolumeTaylorLocalExpansion,
            VolumeTaylorMultipoleExpansion, True, False),
        (LaplaceKernel(2), LaplaceConformingVolumeTaylorLocalExpansion,
            LaplaceConformingVolumeTaylorMultipoleExpansion, True, False),
        (LaplaceKernel(2), LaplaceConformingVolumeTaylorLocalExpansion,
            LaplaceConformingVolumeTaylorMultipoleExpansion, True, True),
        (LaplaceKernel(2), LaplaceConformingVolumeTaylorLocalExpansion,
            LaplaceConformingVolumeTaylorMultipoleExpansion, False, False),
        (LaplaceKernel(3), VolumeTaylorLocalExpansion,
            VolumeTaylorMultipoleExpansion, False, False),
        (LaplaceKernel(3), VolumeTaylorLocalExpansion,
            VolumeTaylorMultipoleExpansion, True, False),
        (LaplaceKernel(3), LaplaceConformingVolumeTaylorLocalExpansion,
            LaplaceConformingVolumeTaylorMultipoleExpansion, True, False),
        (LaplaceKernel(3), LaplaceConformingVolumeTaylorLocalExpansion,
            LaplaceConformingVolumeTaylorMultipoleExpansion, False, False),
        (HelmholtzKernel(2), VolumeTaylorLocalExpansion,
            VolumeTaylorMultipoleExpansion, False, False),
        (HelmholtzKernel(2), HelmholtzConformingVolumeTaylorLocalExpansion,
            HelmholtzConformingVolumeTaylorMultipoleExpansion, False, False),
        (HelmholtzKernel(2), H2DLocalExpansion,
            H2DMultipoleExpansion, False, False),
        (HelmholtzKernel(2), H2DLocalExpansion,
            H2DMultipoleExpansion, True, False),
        #(HelmholtzKernel(2), H2DLocalExpansion,
        #    H2DMultipoleExpansion, True, True),
        (HelmholtzKernel(3), VolumeTaylorLocalExpansion,
            VolumeTaylorMultipoleExpansion, False, False),
        (HelmholtzKernel(3), HelmholtzConformingVolumeTaylorLocalExpansion,
            HelmholtzConformingVolumeTaylorMultipoleExpansion, False, False),
        (YukawaKernel(2), Y2DLocalExpansion,
            Y2DMultipoleExpansion, False, False),
    ])
def test_sumpy_fmm(ctx_factory, knl, local_expn_class, mpole_expn_class,
        optimized_m2l, use_fft):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nsources = 1000
    ntargets = 300
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
                [fp.points[i] for i in range(knl.dim)])

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    # {{{ plot tree

    if 0:
        host_tree = tree.get(queue)
        host_trav = trav.get(queue)

        if 0:
            print("src_box", host_tree.find_box_nr_for_source(403))
            print("tgt_box", host_tree.find_box_nr_for_target(28))
            print(list(host_trav.target_or_target_parent_boxes).index(37))
            print(host_trav.get_box_list("sep_bigger", 22))

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(host_tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.set_bounding_box()
        plotter.draw_box_numbers()

        import matplotlib.pyplot as pt
        pt.show()

    # }}}

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(ctx, seed=44)
    weights = rng.uniform(queue, nsources, dtype=np.float64)

    logger.info("computing direct (reference) result")

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
        elif knl.dim == 2 and issubclass(local_expn_class, H2DLocalExpansion):
            order_values = [10, 12]

    elif isinstance(knl, YukawaKernel):
        extra_kwargs["lam"] = 2
        dtype = np.complex128

        if knl.dim == 3:
            order_values = [1, 2]
        elif knl.dim == 2 and issubclass(local_expn_class, Y2DLocalExpansion):
            order_values = [10, 12]

    from functools import partial
    for order in order_values:
        target_kernels = [knl]

        from sumpy.fmm import (SumpyExpansionWranglerCodeContainer,
            SumpyTranslationClassesData)

        if optimized_m2l:
            translation_classes_data = SumpyTranslationClassesData(queue, trav)
        else:
            translation_classes_data = None

        wcc = SumpyExpansionWranglerCodeContainer(
                ctx,
                partial(mpole_expn_class, knl),
                partial(local_expn_class, knl),
                target_kernels, use_fft=use_fft)

        wrangler = wcc.get_wrangler(queue, tree, dtype,
                fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order,
                kernel_extra_kwargs=extra_kwargs,
                translation_classes_data=translation_classes_data)

        from boxtree.fmm import drive_fmm

        pot, = drive_fmm(trav, wrangler, (weights,))

        from sumpy import P2P
        p2p = P2P(ctx, target_kernels, exclude_self=False)
        evt, (ref_pot,) = p2p(queue, targets, sources, (weights,),
                **extra_kwargs)

        pot = pot.get()
        ref_pot = ref_pot.get()

        rel_err = la.norm(pot - ref_pot, np.inf) / la.norm(ref_pot, np.inf)
        logger.info("order %d -> relative l2 error: %g" % (order, rel_err))

        pconv_verifier.add_data_point(order, rel_err)

    print(pconv_verifier)
    pconv_verifier()


def test_unified_single_and_double(ctx_factory):
    """
    Test that running one FMM for single layer + double layer gives the
    same result as running one FMM for each and adding the results together
    at the end
    """
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    knl = LaplaceKernel(2)
    local_expn_class = LaplaceConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LaplaceConformingVolumeTaylorMultipoleExpansion

    nsources = 1000
    ntargets = 300
    dtype = np.float64

    from boxtree.tools import (
            make_normal_particle_array as p_normal)

    sources = p_normal(queue, nsources, knl.dim, dtype, seed=15)
    offset = np.zeros(knl.dim)
    offset[0] = 0.1

    targets = (
                p_normal(queue, ntargets, knl.dim, dtype, seed=18)
                + offset)

    del offset

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(ctx, seed=44)
    weights = (
        rng.uniform(queue, nsources, dtype=np.float64),
        rng.uniform(queue, nsources, dtype=np.float64),
    )

    logger.info("computing direct (reference) result")

    dtype = np.float64
    order = 3

    from functools import partial
    from sumpy.kernel import DirectionalSourceDerivative, AxisTargetDerivative

    deriv_knl = DirectionalSourceDerivative(knl, "dir_vec")

    target_kernels = [knl, AxisTargetDerivative(0, knl)]
    source_kernel_vecs = [[knl], [deriv_knl], [knl, deriv_knl]]
    strength_usages = [[0], [1], [0, 1]]

    alpha = np.linspace(0, 2*np.pi, nsources, np.float64)
    dir_vec = np.vstack([np.cos(alpha), np.sin(alpha)])

    results = []
    for source_kernels, strength_usage in zip(source_kernel_vecs, strength_usages):
        source_extra_kwargs = {}
        if deriv_knl in source_kernels:
            source_extra_kwargs["dir_vec"] = dir_vec
        from sumpy.fmm import SumpyExpansionWranglerCodeContainer
        wcc = SumpyExpansionWranglerCodeContainer(
                ctx,
                partial(mpole_expn_class, knl),
                partial(local_expn_class, knl),
                target_kernels=target_kernels, source_kernels=source_kernels,
                strength_usage=strength_usage)
        wrangler = wcc.get_wrangler(queue, tree, dtype,
                fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order,
                source_extra_kwargs=source_extra_kwargs)

        from boxtree.fmm import drive_fmm

        pot = drive_fmm(trav, wrangler, weights)
        results.append(np.array([pot[0].get(), pot[1].get()]))

    ref_pot = results[0] + results[1]
    pot = results[2]
    rel_err = la.norm(pot - ref_pot, np.inf) / la.norm(ref_pot, np.inf)

    assert rel_err < 1e-12


def test_sumpy_fmm_timing_data_collection(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(
            ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)

    nsources = 500
    dtype = np.float64

    from boxtree.tools import (
            make_normal_particle_array as p_normal)

    knl = LaplaceKernel(2)
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    order = 1

    sources = p_normal(queue, nsources, knl.dim, dtype, seed=15)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(ctx)
    weights = rng.uniform(queue, nsources, dtype=np.float64)

    target_kernels = [knl]

    from functools import partial

    from sumpy.fmm import SumpyExpansionWranglerCodeContainer
    wcc = SumpyExpansionWranglerCodeContainer(
            ctx,
            partial(mpole_expn_class, knl),
            partial(local_expn_class, knl),
            target_kernels)

    wrangler = wcc.get_wrangler(queue, tree, dtype,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order)
    from boxtree.fmm import drive_fmm

    timing_data = {}
    pot, = drive_fmm(trav, wrangler, (weights,), timing_data=timing_data)
    print(timing_data)
    assert timing_data


def test_sumpy_fmm_exclude_self(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nsources = 500
    dtype = np.float64

    from boxtree.tools import (
            make_normal_particle_array as p_normal)

    knl = LaplaceKernel(2)
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    order = 10

    sources = p_normal(queue, nsources, knl.dim, dtype, seed=15)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(ctx)
    weights = rng.uniform(queue, nsources, dtype=np.float64)

    target_to_source = np.arange(tree.ntargets, dtype=np.int32)
    self_extra_kwargs = {"target_to_source": target_to_source}

    target_kernels = [knl]

    from functools import partial

    from sumpy.fmm import SumpyExpansionWranglerCodeContainer
    wcc = SumpyExpansionWranglerCodeContainer(
            ctx,
            partial(mpole_expn_class, knl),
            partial(local_expn_class, knl),
            target_kernels,
            exclude_self=True)

    wrangler = wcc.get_wrangler(queue, tree, dtype,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order,
            self_extra_kwargs=self_extra_kwargs)

    from boxtree.fmm import drive_fmm

    pot, = drive_fmm(trav, wrangler, (weights,))

    from sumpy import P2P
    p2p = P2P(ctx, target_kernels, exclude_self=True)
    evt, (ref_pot,) = p2p(queue, sources, sources, (weights,),
            **self_extra_kwargs)

    pot = pot.get()
    ref_pot = ref_pot.get()

    rel_err = la.norm(pot - ref_pot) / la.norm(ref_pot)
    logger.info("order %d -> relative l2 error: %g" % (order, rel_err))

    assert np.isclose(rel_err, 0, atol=1e-7)


def test_sumpy_axis_source_derivative(ctx_factory):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nsources = 500
    dtype = np.float64

    from boxtree.tools import (
            make_normal_particle_array as p_normal)

    knl = LaplaceKernel(2)
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    order = 10

    sources = p_normal(queue, nsources, knl.dim, dtype, seed=15)

    from boxtree import TreeBuilder
    tb = TreeBuilder(ctx)

    tree, _ = tb(queue, sources,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(ctx)
    trav, _ = tbuild(queue, tree, debug=True)

    from pyopencl.clrandom import PhiloxGenerator
    rng = PhiloxGenerator(ctx, seed=12)
    weights = rng.uniform(queue, nsources, dtype=np.float64)

    target_to_source = np.arange(tree.ntargets, dtype=np.int32)
    self_extra_kwargs = {"target_to_source": target_to_source}

    from functools import partial

    from sumpy.fmm import SumpyExpansionWranglerCodeContainer
    from sumpy.kernel import AxisTargetDerivative, AxisSourceDerivative

    pots = []
    for tgt_knl, src_knl in [(AxisTargetDerivative(0, knl), knl),
            (knl, AxisSourceDerivative(0, knl))]:

        wcc = SumpyExpansionWranglerCodeContainer(
                ctx,
                partial(mpole_expn_class, knl),
                partial(local_expn_class, knl),
                target_kernels=[tgt_knl],
                source_kernels=[src_knl],
                exclude_self=True)

        wrangler = wcc.get_wrangler(queue, tree, dtype,
                fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order,
                self_extra_kwargs=self_extra_kwargs)

        from boxtree.fmm import drive_fmm

        pot, = drive_fmm(trav, wrangler, (weights,))
        pots.append(pot.get())

    rel_err = la.norm(pots[0] + pots[1]) / la.norm(pots[0])
    logger.info("order %d -> relative l2 error: %g" % (order, rel_err))

    assert np.isclose(rel_err, 0, atol=1e-5)


# You can test individual routines by typing
# $ python test_fmm.py 'test_sumpy_fmm(cl.create_some_context, LaplaceKernel(2),
#       VolumeTaylorLocalExpansion, VolumeTaylorMultipoleExpansion, False, False)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
