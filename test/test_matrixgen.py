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

import pytest
import pyopencl as cl
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from sumpy.tools import MatrixBlockIndex

import logging
logger = logging.getLogger(__name__)

try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


def create_arguments(n, mode, target_radius=1.0):
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

    return targets, sources, centers, sigma, expansion_radii


def create_index_subset(nnodes, nblks, factor):
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

    return indices_, ranges_


@pytest.mark.parametrize('factor', [1.0, 0.6])
def test_qbx_direct(ctx_getter, factor):
    # This evaluates a single layer potential on a circle.
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    from sumpy.kernel import LaplaceKernel
    lknl = LaplaceKernel(2)

    nblks = 10
    order = 12
    mode_nr = 25

    from sumpy.qbx import LayerPotential
    from sumpy.qbx import LayerPotentialMatrixGenerator
    from sumpy.qbx import LayerPotentialMatrixBlockGenerator
    from sumpy.expansion.local import LineTaylorLocalExpansion
    lpot = LayerPotential(ctx, [LineTaylorLocalExpansion(lknl, order)])
    mat_gen = LayerPotentialMatrixGenerator(ctx,
            [LineTaylorLocalExpansion(lknl, order)])
    blk_gen = LayerPotentialMatrixBlockGenerator(ctx,
            [LineTaylorLocalExpansion(lknl, order)])

    for n in [200, 300, 400]:
        targets, sources, centers, sigma, expansion_radii = \
                create_arguments(n, mode_nr)

        h = 2 * np.pi / n
        strengths = (sigma * h,)

        tgtindices, tgtranges = create_index_subset(n, nblks, factor)
        srcindices, srcranges = create_index_subset(n, nblks, factor)
        assert tgtranges.shape == srcranges.shape

        _, (mat,) = mat_gen(queue, targets, sources, centers,
                expansion_radii)
        result_mat = mat.dot(strengths[0])

        _, (result_lpot,) = lpot(queue, targets, sources, centers, strengths,
                expansion_radii)

        eps = 1.0e-10 * la.norm(result_lpot)
        assert la.norm(result_mat - result_lpot) < eps

        index_set = MatrixBlockIndex(queue,
            tgtindices, srcindices, tgtranges, srcranges)
        _, (blk,) = blk_gen(queue, targets, sources, centers, expansion_radii,
                            index_set)

        rowindices, colindices = index_set.linear_indices()
        eps = 1.0e-10 * la.norm(mat)
        assert la.norm(blk - mat[rowindices, colindices].reshape(-1)) < eps


@pytest.mark.parametrize(("exclude_self", "factor"),
    [(True, 1.0), (True, 0.6), (False, 1.0), (False, 0.6)])
def test_p2p_direct(ctx_getter, exclude_self, factor):
    # This does a point-to-point kernel evaluation on a circle.
    logging.basicConfig(level=logging.INFO)
    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    from sumpy.kernel import LaplaceKernel
    lknl = LaplaceKernel(2)

    nblks = 10
    mode_nr = 25

    from sumpy.p2p import P2P
    from sumpy.p2p import P2PMatrixGenerator, P2PMatrixBlockGenerator
    lpot = P2P(ctx, [lknl], exclude_self=exclude_self)
    mat_gen = P2PMatrixGenerator(ctx, [lknl], exclude_self=exclude_self)
    blk_gen = P2PMatrixBlockGenerator(ctx, [lknl], exclude_self=exclude_self)

    for n in [200, 300, 400]:
        targets, sources, _, sigma, _ = \
            create_arguments(n, mode_nr, target_radius=1.2)

        h = 2 * np.pi / n
        strengths = (sigma * h,)

        tgtindices, tgtranges = create_index_subset(n, nblks, factor)
        srcindices, srcranges = create_index_subset(n, nblks, factor)
        assert tgtranges.shape == srcranges.shape

        extra_kwargs = {}
        if exclude_self:
            extra_kwargs["target_to_source"] = np.arange(n, dtype=np.int32)

        _, (mat,) = mat_gen(queue, targets, sources, **extra_kwargs)
        result_mat = mat.dot(strengths[0])

        _, (result_lpot,) = lpot(queue, targets, sources, strengths,
                                 **extra_kwargs)

        eps = 1.0e-10 * la.norm(result_lpot)
        assert la.norm(result_mat - result_lpot) < eps

        index_set = MatrixBlockIndex(queue,
                tgtindices, srcindices, tgtranges, srcranges)
        _, (blk,) = blk_gen(queue, targets, sources, index_set, **extra_kwargs)

        rowindices, colindices = index_set.linear_indices()
        eps = 1.0e-10 * la.norm(mat)
        assert la.norm(blk - mat[rowindices, colindices].reshape(-1)) < eps


# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])

# vim: fdm=marker
