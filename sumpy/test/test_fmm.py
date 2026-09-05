from __future__ import annotations


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

import logging
import sys
from dataclasses import fields
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la
import pytest

from arraycontext import (
    ArrayContextFactory,
    PyOpenCLArrayContext,
    pytest_generate_tests_for_array_contexts,
)
from pytools import obj_array

from sumpy.array_context import (  # ruff:ignore[unused-import]
    PytestPyOpenCLArrayContextFactory,
    _acf,
)
from sumpy.expansion.local import (
    H2DLocalExpansion,
    LinearPDEConformingVolumeTaylorLocalExpansion,
    LocalExpansionBase,
    VolumeTaylorLocalExpansion,
    Y2DLocalExpansion,
)
from sumpy.expansion.multipole import (
    H2DMultipoleExpansion,
    LinearPDEConformingVolumeTaylorMultipoleExpansion,
    MultipoleExpansionBase,
    VolumeTaylorMultipoleExpansion,
    Y2DMultipoleExpansion,
)
from sumpy.fmm import SumpyExpansionWrangler, SumpyTreeIndependentDataForWrangler
from sumpy.kernel import (
    BiharmonicKernel,
    HelmholtzKernel,
    LaplaceKernel,
    ScalarKernel,
    YukawaKernel,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import pyopencl as cl

logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_sumpy_fmm

@pytest.mark.parametrize(
    ("use_translation_classes", "use_fft", "fft_backend_name"), [
        (False, False, None),
        (True, False, None),
        (True, True, "loopy"),
        (True, True, "pyvkfft"),
    ])
@pytest.mark.parametrize(
    ("knl", "local_expn_class", "mpole_expn_class", "order_varies_with_level"), [
            (LaplaceKernel(2), VolumeTaylorLocalExpansion,
                VolumeTaylorMultipoleExpansion, False),
            (LaplaceKernel(2), LinearPDEConformingVolumeTaylorLocalExpansion,
                LinearPDEConformingVolumeTaylorMultipoleExpansion, False),
            (LaplaceKernel(3), VolumeTaylorLocalExpansion,
                VolumeTaylorMultipoleExpansion, False),
            (LaplaceKernel(3), LinearPDEConformingVolumeTaylorLocalExpansion,
                LinearPDEConformingVolumeTaylorMultipoleExpansion, False),
            (HelmholtzKernel(2), VolumeTaylorLocalExpansion,
                VolumeTaylorMultipoleExpansion, False),
            (HelmholtzKernel(2), LinearPDEConformingVolumeTaylorLocalExpansion,
                LinearPDEConformingVolumeTaylorMultipoleExpansion, False),
            (HelmholtzKernel(2), H2DLocalExpansion, H2DMultipoleExpansion, False),
            (HelmholtzKernel(2), H2DLocalExpansion, H2DMultipoleExpansion, True),
            (HelmholtzKernel(3), VolumeTaylorLocalExpansion,
                VolumeTaylorMultipoleExpansion, False),
            (HelmholtzKernel(3), LinearPDEConformingVolumeTaylorLocalExpansion,
                LinearPDEConformingVolumeTaylorMultipoleExpansion, False),
            (YukawaKernel(2), Y2DLocalExpansion, Y2DMultipoleExpansion, False),
    ])
def test_sumpy_fmm(
            actx_factory: ArrayContextFactory,
            knl: ScalarKernel,
            local_expn_class: type[LocalExpansionBase],
            mpole_expn_class: type[MultipoleExpansionBase],
            order_varies_with_level: bool,
            use_translation_classes: bool,
            use_fft: bool,
            fft_backend_name: str | None,
            visualize: bool = False) -> None:
    if fft_backend_name == "pyvkfft":
        pytest.importorskip("pyvkfft")

    if visualize:
        logging.basicConfig(level=logging.INFO)

    if local_expn_class == VolumeTaylorLocalExpansion and use_fft:
        pytest.skip("VolumeTaylorExpansion with FFT takes a lot of resources.")

    if local_expn_class in [H2DLocalExpansion, Y2DLocalExpansion] and use_fft:
        pytest.skip("Fourier/Bessel based expansions with FFT is not supported yet.")

    _test_sumpy_fmm(
        actx_factory, knl,
        local_expn_class, mpole_expn_class,
        order_varies_with_level=order_varies_with_level,
        use_translation_classes=use_translation_classes,
        use_fft=use_fft,
        fft_backend_name=fft_backend_name,
        visualize=visualize,
   )


def _test_sumpy_fmm(
            actx_factory: ArrayContextFactory,
            knl: ScalarKernel,
            local_expn_class: type[LocalExpansionBase],
            mpole_expn_class: type[MultipoleExpansionBase], *,
            order_varies_with_level: bool,
            use_translation_classes: bool,
            use_fft: bool,
            fft_backend_name: str | None,
            visualize: bool = False) -> None:
    from sumpy.tools import FFTBackend

    if fft_backend_name is None:
        fft_backend = None
    else:
        fft_backend = FFTBackend[fft_backend_name.upper()]

    actx = actx_factory()
    if fft_backend == FFTBackend.PYVKFFT and isinstance(actx, PyOpenCLArrayContext):
        from pyopencl.characterize import get_pocl_version
        pocl_version = get_pocl_version(actx.queue.device.platform)

        if pocl_version and pocl_version >= (7,):
            pytest.skip("pocl 7 and pyvkfft don't get along: "
                        "https://github.com/pocl/pocl/issues/2069")

    rng = np.random.default_rng(44)

    nsources = 1000
    ntargets = 300
    dtype = np.dtype(np.float64)

    from boxtree.tools import make_normal_particle_array as p_normal
    sources = p_normal(actx, nsources, knl.dim, dtype, rng=rng)

    if not visualize:
        targets = obj_array.new_1d([
            x + offset for offset, x in zip(
                [0.1, 0.0, 0.0, 0.0][:knl.dim],
                p_normal(actx, ntargets, knl.dim, dtype, rng=rng),
                strict=True)
        ])
    else:
        from sumpy.visualization import FieldPlotter
        fp = FieldPlotter(np.array([0.5, 0]), extent=3, npoints=200)

        targets = obj_array.new_1d([fp.points[i] for i in range(knl.dim)])

    from boxtree import Tree, TreeBuilder
    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(actx)
    trav, _ = tbuild(actx, tree, debug=True)

    # {{{ plot tree

    if visualize:
        host_tree = actx.to_numpy(tree)
        host_trav = actx.to_numpy(trav)

        logger.info("src_box: %s", host_tree.find_box_nr_for_source(403))
        logger.info("tgt_box: %s", host_tree.find_box_nr_for_target(28))
        logger.info("%s",
            list(host_trav.target_or_target_parent_boxes).index(37))
        logger.info("%s", host_trav.get_box_list("sep_bigger", 22))

        from boxtree.visualization import TreePlotter
        plotter = TreePlotter(host_tree)
        plotter.draw_tree(fill=False, edgecolor="black", zorder=10)
        plotter.set_bounding_box()
        plotter.draw_box_numbers()

        import matplotlib.pyplot as pt
        pt.show()

    # }}}

    weights = actx.from_numpy(rng.random(nsources, dtype=np.float64))
    logger.info("computing direct (reference) result")

    from pytools.convergence import PConvergenceVerifier
    pconv_verifier = PConvergenceVerifier()

    extra_kwargs: dict[str, object] = {}
    dtype = np.float64
    order_values = [1, 2, 3]
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05
        dtype = np.complex128

        if knl.dim == 3:
            order_values = [1, 2]
        elif knl.dim == 2 and issubclass(local_expn_class, H2DLocalExpansion):
            order_values = [4, 5]

    elif isinstance(knl, YukawaKernel):
        extra_kwargs["lam"] = 2
        dtype = np.complex128

        if knl.dim == 3:
            order_values = [1, 2]
        elif knl.dim == 2 and issubclass(local_expn_class, Y2DLocalExpansion):
            order_values = [10, 12]

    for order in order_values:
        target_kernels = [knl]

        if use_fft:
            from sumpy.expansion.m2l import FFTM2LTranslationClassFactory
            m2l_translation_factory = FFTM2LTranslationClassFactory()
        else:
            from sumpy.expansion.m2l import NonFFTM2LTranslationClassFactory
            m2l_translation_factory = NonFFTM2LTranslationClassFactory()

        m2l_translation = m2l_translation_factory.get_m2l_translation_class(
                knl, local_expn_class)()

        if any(f.name == "m2l_translation_override" for f in fields(local_expn_class)):
            local_expansion_factory = partial(local_expn_class, knl,
                                              m2l_translation_override=m2l_translation)
        else:
            local_expansion_factory = partial(local_expn_class, knl)

        tree_indep = SumpyTreeIndependentDataForWrangler(
                actx,
                partial(mpole_expn_class, knl),
                local_expansion_factory,
                target_kernels,
                fft_backend=fft_backend,
        )

        if order_varies_with_level:
            def fmm_level_to_order(  # pyright: ignore[reportRedeclaration]
                    _kernel: ScalarKernel,
                    _kernel_args: frozenset[tuple[str, object]],
                    _tree: Tree,
                    lev: int
                ) -> int:
                return order + lev % 2  # ruff:ignore[function-uses-loop-variable]
        else:
            def fmm_level_to_order(
                    _kernel: ScalarKernel,
                    _kernel_args: frozenset[tuple[str, object]],
                    _tree: Tree,
                    _lev: int
                ) -> int:
                return order  # ruff:ignore[function-uses-loop-variable]

        wrangler = SumpyExpansionWrangler(tree_indep, trav, dtype,
            fmm_level_to_order=fmm_level_to_order,
            kernel_extra_kwargs=extra_kwargs,
            _disable_translation_classes=not use_translation_classes)

        from boxtree.fmm import drive_fmm

        pot, = drive_fmm(actx, wrangler, (weights,))

        from sumpy import P2P
        p2p = P2P(target_kernels, exclude_self=False)
        ref_pot, = p2p(actx, targets, sources, (weights,), **extra_kwargs)

        pot = actx.to_numpy(pot)
        ref_pot = actx.to_numpy(ref_pot)

        rel_err = la.norm(pot - ref_pot, np.inf) / la.norm(ref_pot, np.inf)
        logger.info("order %d -> relative l2 error: %g", order, rel_err)

        pconv_verifier.add_data_point(order, rel_err)

    logger.info("\n%s", pconv_verifier)
    pconv_verifier()

# }}}


# {{{ test_coeff_magnitude_rscale

@pytest.mark.parametrize("knl", [LaplaceKernel(2), BiharmonicKernel(2)])
def test_coeff_magnitude_rscale(
        actx_factory: ArrayContextFactory, knl: ScalarKernel) -> None:
    """Checks that the rscale used keeps the coefficient magnitude
    difference small
    """
    local_expn_class = LinearPDEConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LinearPDEConformingVolumeTaylorMultipoleExpansion

    actx = actx_factory()
    rng = np.random.default_rng(31)

    nsources = 1000
    ntargets = 300
    dtype = np.dtype(np.float64)

    from boxtree.tools import make_normal_particle_array as p_normal

    sources = p_normal(actx, nsources, knl.dim, dtype, rng=rng)
    targets = obj_array.new_1d([
        x + offset for offset, x in zip(
            [0.1, 0.0, 0.0, 0.0][:knl.dim],
            p_normal(actx, ntargets, knl.dim, dtype, rng=rng),
            strict=True)
    ])

    from boxtree import Tree, TreeBuilder
    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets, max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(actx)
    trav, _ = tbuild(actx, tree, debug=True)

    weights = actx.from_numpy(rng.random(nsources, dtype=np.float64))

    extra_kwargs: dict[str, object] = {}
    dtype = np.float64
    order = 10
    if isinstance(knl, HelmholtzKernel):
        extra_kwargs["k"] = 0.05
        dtype = np.complex128

    elif isinstance(knl, YukawaKernel):
        extra_kwargs["lam"] = 2
        dtype = np.complex128

    target_kernels = [knl]

    tree_indep = SumpyTreeIndependentDataForWrangler(
        actx,
        partial(mpole_expn_class, knl),
        partial(local_expn_class, knl),
        target_kernels)

    def fmm_level_to_order(
            _kernel: ScalarKernel,
            _kernel_args: frozenset[tuple[str, object]],
            _tree: Tree,
            _lev: int
        ) -> int:
        return order

    wrangler = SumpyExpansionWrangler(tree_indep, trav, dtype,
        fmm_level_to_order=fmm_level_to_order,
        kernel_extra_kwargs=extra_kwargs)

    weights = wrangler.reorder_sources(weights)
    (weights,) = wrangler.distribute_source_weights(actx, (weights,), None)

    assert trav.level_start_target_or_target_parent_box_nrs is not None
    local_result = wrangler.form_locals(
        actx,
        trav.level_start_target_or_target_parent_box_nrs,
        trav.target_or_target_parent_boxes,
        trav.from_sep_bigger_starts,
        trav.from_sep_bigger_lists,
        (weights,))

    result = actx.to_numpy(
        actx.np.abs(wrangler.local_expansions_view(local_result, 5)[1][0])
        )

    result_ratio = np.max(result) / np.min(result)
    assert result_ratio < 10**6, result_ratio

# }}}


# {{{ test_unified_single_and_double

def test_unified_single_and_double(
        actx_factory: ArrayContextFactory, visualize: bool = False) -> None:
    """
    Test that running one FMM for single layer + double layer gives the
    same result as running one FMM for each and adding the results together
    at the end
    """
    if visualize:
        logging.basicConfig(level=logging.INFO)

    actx = actx_factory()
    rng = np.random.default_rng(44)

    knl = LaplaceKernel(2)
    local_expn_class = LinearPDEConformingVolumeTaylorLocalExpansion
    mpole_expn_class = LinearPDEConformingVolumeTaylorMultipoleExpansion

    nsources = 1000
    ntargets = 300
    dtype = np.float64

    from boxtree.tools import make_normal_particle_array as p_normal

    sources = p_normal(actx, nsources, knl.dim, dtype, rng=rng)
    targets = obj_array.new_1d([
        x + offset for offset, x in zip(
            [0.1, 0.0, 0.0, 0.0][:knl.dim],
            p_normal(actx, ntargets, knl.dim, dtype, rng=rng),
            strict=True)
    ])

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, targets=targets, max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(actx)
    trav, _ = tbuild(actx, tree, debug=True)

    weights = (
        actx.from_numpy(rng.random(nsources, dtype=np.float64)),
        actx.from_numpy(rng.random(nsources, dtype=np.float64))
        )

    logger.info("computing direct (reference) result")

    dtype = np.float64
    order = 3

    from sumpy.kernel import AxisTargetDerivative, DirectionalSourceDerivative

    deriv_knl = DirectionalSourceDerivative(knl, "dir_vec")

    target_kernels = [knl, AxisTargetDerivative(0, knl)]
    source_kernel_vecs = [[knl], [deriv_knl], [knl, deriv_knl]]
    strength_usages = [[0], [1], [0, 1]]

    alpha = np.linspace(0, 2*np.pi, nsources, dtype=np.float64)
    dir_vec = actx.from_numpy(np.vstack([np.cos(alpha), np.sin(alpha)]))

    results: list[np.ndarray[tuple[int, int], np.dtype[np.float64]]] = []
    for source_kernels, strength_usage in zip(
            source_kernel_vecs, strength_usages, strict=True):
        source_extra_kwargs: dict[str, object] = {}
        if deriv_knl in source_kernels:
            source_extra_kwargs["dir_vec"] = dir_vec
        tree_indep = SumpyTreeIndependentDataForWrangler(
                actx,
                partial(mpole_expn_class, knl),
                partial(local_expn_class, knl),
                target_kernels=target_kernels, source_kernels=source_kernels,
                strength_usage=strength_usage)
        wrangler = SumpyExpansionWrangler(tree_indep, trav, dtype,
                fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order,
                source_extra_kwargs=source_extra_kwargs)

        from boxtree.fmm import drive_fmm

        pot = drive_fmm(actx, wrangler, weights)
        results.append(np.array([actx.to_numpy(pot[0]), actx.to_numpy(pot[1])]))

    ref_pot = results[0] + results[1]
    pot = results[2]
    rel_err = la.norm(pot - ref_pot, np.inf) / la.norm(ref_pot, np.inf)

    assert rel_err < 1e-12

# }}}


# {{{ test_sumpy_fmm_timing_data_collection

@pytest.mark.parametrize("use_fft", [True, False])
def test_sumpy_fmm_timing_data_collection(
        ctx_factory: Callable[[], cl.Context],
        use_fft: bool,
        visualize: bool = False) -> None:
    if visualize:
        logging.basicConfig(level=logging.INFO)

    import pyopencl as cl

    from sumpy.array_context import PyOpenCLArrayContext

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx,
        properties=cl.command_queue_properties.PROFILING_ENABLE)
    actx = PyOpenCLArrayContext(queue)

    rng = np.random.default_rng(44)
    nsources = 500
    dtype = np.float64

    from boxtree.tools import make_normal_particle_array as p_normal

    knl = LaplaceKernel(2)
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    order = 1

    sources = p_normal(actx, nsources, knl.dim, dtype, rng=rng)

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(actx)
    trav, _ = tbuild(actx, tree, debug=True)

    weights = actx.from_numpy(rng.random(nsources, dtype=np.float64))

    target_kernels = [knl]

    if use_fft:
        from sumpy.expansion.m2l import FFTM2LTranslationClassFactory
        m2l_translation_factory = FFTM2LTranslationClassFactory()
    else:
        from sumpy.expansion.m2l import NonFFTM2LTranslationClassFactory
        m2l_translation_factory = NonFFTM2LTranslationClassFactory()

    m2l_translation = m2l_translation_factory.get_m2l_translation_class(
                knl, local_expn_class)()

    tree_indep = SumpyTreeIndependentDataForWrangler(
            actx,
            partial(mpole_expn_class, knl),
            partial(local_expn_class, knl, m2l_translation_override=m2l_translation),
            target_kernels)

    wrangler = SumpyExpansionWrangler(tree_indep, trav, dtype,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order)
    from boxtree.fmm import drive_fmm

    _pot, = drive_fmm(actx, wrangler, (weights,))


def test_sumpy_fmm_exclude_self(
        actx_factory: ArrayContextFactory, visualize: bool = False) -> None:
    if visualize:
        logging.basicConfig(level=logging.INFO)

    actx = actx_factory()
    rng = np.random.default_rng(44)

    nsources = 500
    dtype = np.float64

    from boxtree.tools import make_normal_particle_array as p_normal

    knl = LaplaceKernel(2)
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    order = 10

    sources = p_normal(actx, nsources, knl.dim, dtype, rng=rng)

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx)

    tree, _ = tb(actx, sources, max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(actx)
    trav, _ = tbuild(actx, tree, debug=True)

    weights = actx.from_numpy(rng.random(nsources, dtype=np.float64))

    target_to_source = actx.from_numpy(np.arange(tree.ntargets, dtype=np.int32))
    self_extra_kwargs = {"target_to_source": target_to_source}

    target_kernels = [knl]

    tree_indep = SumpyTreeIndependentDataForWrangler(
            actx,
            partial(mpole_expn_class, knl),
            partial(local_expn_class, knl),
            target_kernels,
            exclude_self=True)

    wrangler = SumpyExpansionWrangler(tree_indep, trav, dtype,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order,
            self_extra_kwargs=self_extra_kwargs)

    from boxtree.fmm import drive_fmm

    pot, = drive_fmm(actx, wrangler, (weights,))

    from sumpy import P2P
    p2p = P2P(target_kernels, exclude_self=True)
    ref_pot, = p2p(actx, sources, sources, (weights,), **self_extra_kwargs)

    pot = actx.to_numpy(pot)
    ref_pot = actx.to_numpy(ref_pot)

    rel_err = la.norm(pot - ref_pot) / la.norm(ref_pot)
    logger.info("order %d -> relative l2 error: %g", order, rel_err)

    assert np.isclose(rel_err, 0, atol=1e-7)

# }}}


# {{{ test_sumpy_axis_source_derivative

def test_sumpy_axis_source_derivative(
            actx_factory: ArrayContextFactory,
            visualize: bool = False) -> None:
    if visualize:
        logging.basicConfig(level=logging.INFO)

    actx = actx_factory()
    rng = np.random.default_rng(12)

    nsources = 500
    dtype = np.float64

    from boxtree.tools import make_normal_particle_array as p_normal

    knl = LaplaceKernel(2)
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    order = 10

    sources = p_normal(actx, nsources, knl.dim, dtype, rng=rng)

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx)
    tree, _ = tb(actx, sources, max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(actx)
    trav, _ = tbuild(actx, tree, debug=True)

    weights = actx.from_numpy(rng.random(nsources, dtype=np.float64))

    target_to_source = actx.from_numpy(np.arange(tree.ntargets, dtype=np.int32))
    self_extra_kwargs = {"target_to_source": target_to_source}

    from sumpy.kernel import AxisSourceDerivative, AxisTargetDerivative

    pots: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
    for tgt_knl, src_knl in [
            (AxisTargetDerivative(0, knl), knl),
            (knl, AxisSourceDerivative(0, knl))]:
        tree_indep = SumpyTreeIndependentDataForWrangler(
                actx,
                partial(mpole_expn_class, knl),
                partial(local_expn_class, knl),
                target_kernels=[tgt_knl],
                source_kernels=[src_knl],
                exclude_self=True)

        wrangler = SumpyExpansionWrangler(tree_indep, trav, dtype,
                fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order,
                self_extra_kwargs=self_extra_kwargs)

        from boxtree.fmm import drive_fmm

        pot, = drive_fmm(actx, wrangler, (weights,))
        pots.append(actx.to_numpy(pot))

    rel_err = la.norm(pots[0] + pots[1]) / la.norm(pots[0])
    logger.info("order %d -> relative l2 error: %g", order, rel_err)

    assert np.isclose(rel_err, 0, atol=1e-5)

# }}}


# {{{ test_sumpy_target_point_multiplier

@pytest.mark.parametrize("deriv_axes", [(), (0,), (1,)])
def test_sumpy_target_point_multiplier(
            actx_factory: ArrayContextFactory,
            deriv_axes: tuple[int, ...],
            visualize: bool = False) -> None:
    if visualize:
        logging.basicConfig(level=logging.INFO)

    actx = actx_factory()
    rng = np.random.default_rng(12)

    nsources = 500
    dtype = np.float64

    from boxtree.tools import make_normal_particle_array as p_normal

    knl = LaplaceKernel(2)
    local_expn_class = VolumeTaylorLocalExpansion
    mpole_expn_class = VolumeTaylorMultipoleExpansion
    order = 5

    sources = p_normal(actx, nsources, knl.dim, dtype, rng=rng)

    from boxtree import TreeBuilder
    tb = TreeBuilder(actx)

    tree, _ = tb(actx, sources,
            max_particles_in_box=30, debug=True)

    from boxtree.traversal import FMMTraversalBuilder
    tbuild = FMMTraversalBuilder(actx)
    trav, _ = tbuild(actx, tree, debug=True)

    weights = actx.from_numpy(rng.random(nsources, dtype=np.float64))

    target_to_source = actx.from_numpy(np.arange(tree.ntargets, dtype=np.int32))
    self_extra_kwargs = {"target_to_source": target_to_source}

    from sumpy.kernel import AxisTargetDerivative, TargetPointMultiplier

    tgt_knls: list[ScalarKernel] = [TargetPointMultiplier(0, knl), knl, knl]
    for axis in deriv_axes:
        tgt_knls[0] = AxisTargetDerivative(axis, tgt_knls[0])
        tgt_knls[1] = AxisTargetDerivative(axis, tgt_knls[1])

    tree_indep = SumpyTreeIndependentDataForWrangler(
            actx,
            partial(mpole_expn_class, knl),
            partial(local_expn_class, knl),
            target_kernels=tgt_knls,
            source_kernels=[knl],
            exclude_self=True)

    wrangler = SumpyExpansionWrangler(tree_indep, trav, dtype,
            fmm_level_to_order=lambda kernel, kernel_args, tree, lev: order,
            self_extra_kwargs=self_extra_kwargs)

    from boxtree.fmm import drive_fmm

    pot0, pot1, pot2 = drive_fmm(actx, wrangler, (weights,))
    pot0, pot1, pot2 = actx.to_numpy(pot0), actx.to_numpy(pot1), actx.to_numpy(pot2)
    if deriv_axes == (0,):
        ref_pot = pot1 * actx.to_numpy(sources[0]) + pot2
    else:
        ref_pot = pot1 * actx.to_numpy(sources[0])

    rel_err = la.norm(pot0 - ref_pot) / la.norm(ref_pot)
    logger.info("order %d -> relative l2 error: %g", order, rel_err)

    assert np.isclose(rel_err, 0, atol=1e-5)

# }}}


"""
You can test individual routines by typing
$ python test/test_fmm.py 'test_sumpy_fmm(_acf, LaplaceKernel(2),
      VolumeTaylorLocalExpansion, VolumeTaylorMultipoleExpansion,
      order_varies_with_level=False, use_translation_classes=True, use_fft=True,
      fft_backend="pyvkfft", visualize=True)'
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
