from __future__ import division, absolute_import, print_function

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

import sys
import numpy as np
import numpy.linalg as la

import pyopencl as cl
import pyopencl.array  # noqa

from sumpy.tools import vector_to_device
from sumpy.tools import MatrixBlockIndexRanges

import pytest
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


def _build_geometry(queue, n, mode, target_radius=1.0):
    # parametrize circle
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    unit_circle = np.exp(1j * t)
    unit_circle = np.array([unit_circle.real, unit_circle.imag])

    # create density
    sigma = np.cos(mode * t)

    # create sources and targets
    h = 2.0 * np.pi / n
    targets = target_radius * unit_circle
    sources = unit_circle

    radius = 7.0 * h
    centers = unit_circle * (1.0 - radius)
    expansion_radii = radius * np.ones(n)

    return (cl.array.to_device(queue, targets),
            cl.array.to_device(queue, sources),
            vector_to_device(queue, centers),
            cl.array.to_device(queue, expansion_radii),
            cl.array.to_device(queue, sigma))


def _build_block_index(queue, nnodes, nblks, factor):
    indices = np.arange(0, nnodes)
    ranges = np.arange(0, nnodes + 1, nnodes // nblks)

    if abs(factor - 1.0) < 1.0e-14:
        ranges_ = ranges
        indices_ = indices
    else:
        indices_ = np.empty(ranges.shape[0] - 1, dtype=np.object)
        for i in range(ranges.shape[0] - 1):
            iidx = indices[np.s_[ranges[i]:ranges[i + 1]]]
            indices_[i] = np.sort(np.random.choice(iidx,
                size=int(factor * len(iidx)), replace=False))

        ranges_ = np.cumsum([0] + [r.shape[0] for r in indices_])
        indices_ = np.hstack(indices_)

    from sumpy.tools import BlockIndexRanges
    return BlockIndexRanges(queue.context,
                            cl.array.to_device(queue, indices_).with_queue(None),
                            cl.array.to_device(queue, ranges_).with_queue(None))


@pytest.mark.parametrize('factor', [1.0, 0.6])
@pytest.mark.parametrize('lpot_id', [1, 2])
def test_qbx_direct(ctx_factory, factor, lpot_id):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    ndim = 2
    nblks = 10
    order = 12
    mode_nr = 25

    from sumpy.kernel import LaplaceKernel, DirectionalSourceDerivative
    if lpot_id == 1:
        knl = LaplaceKernel(ndim)
    elif lpot_id == 2:
        knl = LaplaceKernel(ndim)
        knl = DirectionalSourceDerivative(knl, dir_vec_name="dsource_vec")
    else:
        raise ValueError("unknow lpot_id")

    from sumpy.expansion.local import LineTaylorLocalExpansion
    lknl = LineTaylorLocalExpansion(knl, order)

    from sumpy.qbx import LayerPotential
    lpot = LayerPotential(ctx, [lknl])

    from sumpy.qbx import LayerPotentialMatrixGenerator
    mat_gen = LayerPotentialMatrixGenerator(ctx, [lknl])

    from sumpy.qbx import LayerPotentialMatrixBlockGenerator
    blk_gen = LayerPotentialMatrixBlockGenerator(ctx, [lknl])

    for n in [200, 300, 400]:
        targets, sources, centers, expansion_radii, sigma = \
                _build_geometry(queue, n, mode_nr, target_radius=1.2)

        h = 2 * np.pi / n
        strengths = (sigma * h,)

        tgtindices = _build_block_index(queue, n, nblks, factor)
        srcindices = _build_block_index(queue, n, nblks, factor)
        index_set = MatrixBlockIndexRanges(ctx, tgtindices, srcindices)

        extra_kwargs = {}
        if lpot_id == 2:
            from pytools.obj_array import make_obj_array
            extra_kwargs["dsource_vec"] = \
                    vector_to_device(queue, make_obj_array(np.ones((ndim, n))))

        _, (result_lpot,) = lpot(queue,
                targets=targets,
                sources=sources,
                centers=centers,
                expansion_radii=expansion_radii,
                strengths=strengths, **extra_kwargs)
        result_lpot = result_lpot.get()

        _, (mat,) = mat_gen(queue,
                targets=targets,
                sources=sources,
                centers=centers,
                expansion_radii=expansion_radii, **extra_kwargs)
        mat = mat.get()
        result_mat = mat.dot(strengths[0].get())

        _, (blk,) = blk_gen(queue,
                targets=targets,
                sources=sources,
                centers=centers,
                expansion_radii=expansion_radii,
                index_set=index_set, **extra_kwargs)
        blk = blk.get()

        rowindices = index_set.linear_row_indices.get(queue)
        colindices = index_set.linear_col_indices.get(queue)

        eps = 1.0e-10 * la.norm(result_lpot)
        assert la.norm(result_mat - result_lpot) < eps
        assert la.norm(blk - mat[rowindices, colindices]) < eps


@pytest.mark.parametrize("exclude_self", [True, False])
@pytest.mark.parametrize("factor", [1.0, 0.6])
@pytest.mark.parametrize('lpot_id', [1, 2])
def test_p2p_direct(ctx_factory, exclude_self, factor, lpot_id):
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    ndim = 2
    nblks = 10
    mode_nr = 25

    from sumpy.kernel import LaplaceKernel, DirectionalSourceDerivative
    if lpot_id == 1:
        lknl = LaplaceKernel(ndim)
    elif lpot_id == 2:
        lknl = LaplaceKernel(ndim)
        lknl = DirectionalSourceDerivative(lknl, dir_vec_name="dsource_vec")
    else:
        raise ValueError("unknow lpot_id")

    from sumpy.p2p import P2P
    lpot = P2P(ctx, [lknl], exclude_self=exclude_self)

    from sumpy.p2p import P2PMatrixGenerator
    mat_gen = P2PMatrixGenerator(ctx, [lknl], exclude_self=exclude_self)

    from sumpy.p2p import P2PMatrixBlockGenerator
    blk_gen = P2PMatrixBlockGenerator(ctx, [lknl], exclude_self=exclude_self)

    for n in [200, 300, 400]:
        targets, sources, _, _, sigma = \
            _build_geometry(queue, n, mode_nr, target_radius=1.2)

        h = 2 * np.pi / n
        strengths = (sigma * h,)

        tgtindices = _build_block_index(queue, n, nblks, factor)
        srcindices = _build_block_index(queue, n, nblks, factor)
        index_set = MatrixBlockIndexRanges(ctx, tgtindices, srcindices)

        extra_kwargs = {}
        if exclude_self:
            extra_kwargs["target_to_source"] = \
                cl.array.arange(queue, 0, n, dtype=np.int)
        if lpot_id == 2:
            from pytools.obj_array import make_obj_array
            extra_kwargs["dsource_vec"] = \
                    vector_to_device(queue, make_obj_array(np.ones((ndim, n))))

        _, (result_lpot,) = lpot(queue,
                targets=targets,
                sources=sources,
                strength=strengths, **extra_kwargs)
        result_lpot = result_lpot.get()

        _, (mat,) = mat_gen(queue,
                targets=targets,
                sources=sources, **extra_kwargs)
        mat = mat.get()
        result_mat = mat.dot(strengths[0].get())

        _, (blk,) = blk_gen(queue,
                targets=targets,
                sources=sources,
                index_set=index_set, **extra_kwargs)
        blk = blk.get()

        eps = 1.0e-10 * la.norm(result_lpot)
        assert la.norm(result_mat - result_lpot) < eps

        index_set = index_set.get(queue)
        for i in range(index_set.nblocks):
            assert la.norm(index_set.block_take(blk, i)
                           - index_set.take(mat, i)) < eps


# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
