__copyright__ = "Copyright (C) 2018 Alexandru Fikl"

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

import pytest
import sys

import numpy as np
import numpy.linalg as la

from arraycontext import pytest_generate_tests_for_array_contexts
from sumpy.array_context import (                                 # noqa: F401
        PytestPyOpenCLArrayContextFactory, _acf)

import logging
logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


def _build_geometry(actx, ntargets, nsources, mode, target_radius=1.0):
    # source points
    t = np.linspace(0.0, 2.0 * np.pi, nsources, endpoint=False)
    sources = np.array([np.cos(t), np.sin(t)])

    # density
    sigma = np.cos(mode * t)

    # target points
    t = np.linspace(0.0, 2.0 * np.pi, ntargets, endpoint=False)
    targets = target_radius * np.array([np.cos(t), np.sin(t)])

    # target centers and expansion radii
    h = 2.0 * np.pi * target_radius / ntargets
    radius = 7.0 * h
    centers = (1.0 - radius) * targets
    expansion_radii = np.full(ntargets, radius)

    return (actx.from_numpy(targets),
            actx.from_numpy(sources),
            actx.from_numpy(centers),
            actx.from_numpy(expansion_radii),
            actx.from_numpy(sigma))


def _build_subset_indices(actx, ntargets, nsources, factor):
    tgtindices = np.arange(0, ntargets)
    srcindices = np.arange(0, nsources)

    rng = np.random.default_rng()
    if abs(factor - 1.0) > 1.0e-14:
        tgtindices = rng.choice(tgtindices,
                size=int(factor * ntargets), replace=False)
        srcindices = rng.choice(srcindices,
                size=int(factor * nsources), replace=False)
    else:
        rng.shuffle(tgtindices)
        rng.shuffle(srcindices)

    tgtindices, srcindices = np.meshgrid(tgtindices, srcindices)
    return (
            actx.freeze(actx.from_numpy(tgtindices.ravel())),
            actx.freeze(actx.from_numpy(srcindices.ravel())))


# {{{ test_qbx_direct

@pytest.mark.parametrize("factor", [1.0, 0.6])
@pytest.mark.parametrize("lpot_id", [1, 2])
def test_qbx_direct(actx_factory, factor, lpot_id, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    ndim = 2
    order = 12
    mode_nr = 25

    from sumpy.kernel import LaplaceKernel, DirectionalSourceDerivative
    if lpot_id == 1:
        base_knl = LaplaceKernel(ndim)
        knl = base_knl
    elif lpot_id == 2:
        base_knl = LaplaceKernel(ndim)
        knl = DirectionalSourceDerivative(base_knl, dir_vec_name="dsource_vec")
    else:
        raise ValueError(f"unknown lpot_id: {lpot_id}")

    from sumpy.expansion.local import LineTaylorLocalExpansion
    expn = LineTaylorLocalExpansion(knl, order)

    from sumpy.qbx import LayerPotential
    lpot = LayerPotential(actx.context, expansion=expn, source_kernels=(knl,),
            target_kernels=(base_knl,))

    from sumpy.qbx import LayerPotentialMatrixGenerator
    mat_gen = LayerPotentialMatrixGenerator(actx.context,
            expansion=expn,
            source_kernels=(knl,),
            target_kernels=(base_knl,))

    from sumpy.qbx import LayerPotentialMatrixSubsetGenerator
    blk_gen = LayerPotentialMatrixSubsetGenerator(actx.context,
            expansion=expn,
            source_kernels=(knl,),
            target_kernels=(base_knl,))

    for n in [200, 300, 400]:
        targets, sources, centers, expansion_radii, sigma = \
                _build_geometry(actx, n, n, mode_nr, target_radius=1.2)

        h = 2 * np.pi / n
        strengths = (sigma * h,)
        tgtindices, srcindices = _build_subset_indices(actx,
                ntargets=n, nsources=n, factor=factor)

        extra_kwargs = {}
        if lpot_id == 2:
            from pytools.obj_array import make_obj_array
            extra_kwargs["dsource_vec"] = (
                    actx.from_numpy(make_obj_array(np.ones((ndim, n))))
                    )

        _, (result_lpot,) = lpot(actx.queue,
                targets=targets,
                sources=sources,
                centers=centers,
                expansion_radii=expansion_radii,
                strengths=strengths, **extra_kwargs)
        result_lpot = actx.to_numpy(result_lpot)

        _, (mat,) = mat_gen(actx.queue,
                targets=targets,
                sources=sources,
                centers=centers,
                expansion_radii=expansion_radii, **extra_kwargs)
        mat = actx.to_numpy(mat)
        result_mat = mat @ actx.to_numpy(strengths[0])

        _, (blk,) = blk_gen(actx.queue,
                targets=targets,
                sources=sources,
                centers=centers,
                expansion_radii=expansion_radii,
                tgtindices=tgtindices,
                srcindices=srcindices, **extra_kwargs)
        blk = actx.to_numpy(blk)

        tgtindices = actx.to_numpy(tgtindices)
        srcindices = actx.to_numpy(srcindices)

        eps = 1.0e-10 * la.norm(result_lpot)
        assert la.norm(result_mat - result_lpot) < eps
        assert la.norm(blk - mat[tgtindices, srcindices]) < eps

# }}}


# {{{ test_p2p_direct

@pytest.mark.parametrize("exclude_self", [True, False])
@pytest.mark.parametrize("factor", [1.0, 0.6])
@pytest.mark.parametrize("lpot_id", [1, 2])
def test_p2p_direct(actx_factory, exclude_self, factor, lpot_id, visualize=False):
    if visualize:
        logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    ndim = 2
    mode_nr = 25

    from sumpy.kernel import LaplaceKernel, DirectionalSourceDerivative
    if lpot_id == 1:
        lknl = LaplaceKernel(ndim)
    elif lpot_id == 2:
        lknl = LaplaceKernel(ndim)
        lknl = DirectionalSourceDerivative(lknl, dir_vec_name="dsource_vec")
    else:
        raise ValueError(f"unknown lpot_id: '{lpot_id}'")

    from sumpy.p2p import P2P
    lpot = P2P(actx.context, [lknl], exclude_self=exclude_self)

    from sumpy.p2p import P2PMatrixGenerator
    mat_gen = P2PMatrixGenerator(actx.context, [lknl], exclude_self=exclude_self)

    from sumpy.p2p import P2PMatrixSubsetGenerator
    blk_gen = P2PMatrixSubsetGenerator(
        actx.context, [lknl], exclude_self=exclude_self)

    for n in [200, 300, 400]:
        targets, sources, _, _, sigma = (
            _build_geometry(actx, n, n, mode_nr, target_radius=1.2))

        h = 2 * np.pi / n
        strengths = (sigma * h,)
        tgtindices, srcindices = _build_subset_indices(actx,
                ntargets=n, nsources=n, factor=factor)

        extra_kwargs = {}
        if exclude_self:
            extra_kwargs["target_to_source"] = (
                actx.from_numpy(np.arange(n, dtype=np.int32))
                )
        if lpot_id == 2:
            from pytools.obj_array import make_obj_array
            extra_kwargs["dsource_vec"] = (
                    actx.from_numpy(make_obj_array(np.ones((ndim, n)))))

        _, (result_lpot,) = lpot(actx.queue,
                targets=targets,
                sources=sources,
                strength=strengths, **extra_kwargs)
        result_lpot = actx.to_numpy(result_lpot)

        _, (mat,) = mat_gen(actx.queue,
                targets=targets,
                sources=sources, **extra_kwargs)
        mat = actx.to_numpy(mat)
        result_mat = mat @ actx.to_numpy(strengths[0])

        _, (blk,) = blk_gen(actx.queue,
                targets=targets,
                sources=sources,
                tgtindices=tgtindices,
                srcindices=srcindices, **extra_kwargs)
        blk = actx.to_numpy(blk)

        tgtindices = actx.to_numpy(tgtindices)
        srcindices = actx.to_numpy(srcindices)

        eps = 1.0e-10 * la.norm(result_lpot)
        assert la.norm(result_mat - result_lpot) < eps
        assert la.norm(blk - mat[tgtindices, srcindices]) < eps

# }}}


# You can test individual routines by typing
# $ python test_matrixgen.py 'test_p2p_direct(_acf, True, 1.0, 1, visualize=True)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
