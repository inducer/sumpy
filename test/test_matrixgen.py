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


def test_qbx_direct(ctx_getter):
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

        tgtindices = np.arange(0, n)
        tgtindices = np.random.choice(tgtindices, size=int(0.8 * n))
        tgtranges = np.arange(0, tgtindices.shape[0] + 1,
                              tgtindices.shape[0] // nblks)
        srcindices = np.arange(0, n)
        srcindices = np.random.choice(srcindices, size=int(0.8 * n))
        srcranges = np.arange(0, srcindices.shape[0] + 1,
                              srcindices.shape[0] // nblks)
        assert tgtranges.shape == srcranges.shape

        _, (mat,) = mat_gen(queue, targets, sources, centers,
                expansion_radii)
        result_mat = mat.dot(strengths[0])

        _, (result_lpot,) = lpot(queue, targets, sources, centers, strengths,
                expansion_radii)

        eps = 1.0e-10 * la.norm(result_lpot)
        assert la.norm(result_mat - result_lpot) < eps

        _, (blk,) = blk_gen(queue, targets, sources, centers, expansion_radii,
                            tgtindices, srcindices, tgtranges, srcranges)

        for i in range(srcranges.shape[0] - 1):
            itgt = np.s_[tgtranges[i]:tgtranges[i + 1]]
            isrc = np.s_[srcranges[i]:srcranges[i + 1]]
            block = np.ix_(tgtindices[itgt], srcindices[isrc])

            eps = 1.0e-10 * la.norm(mat[block])
            assert la.norm(blk[itgt, isrc] - mat[block]) < eps


@pytest.mark.parametrize("exclude_self", [True, False])
def test_p2p_direct(ctx_getter, exclude_self):
    # This evaluates a single layer potential on a circle.
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

        tgtindices = np.arange(0, n)
        tgtindices = np.random.choice(tgtindices, size=int(0.8 * n))
        tgtranges = np.arange(0, tgtindices.shape[0] + 1,
                              tgtindices.shape[0] // nblks)
        srcindices = np.arange(0, n)
        srcindices = np.random.choice(srcindices, size=int(0.8 * n))
        srcranges = np.arange(0, srcindices.shape[0] + 1,
                              srcindices.shape[0] // nblks)
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

        _, (blk,) = blk_gen(queue, targets, sources,
                            tgtindices, srcindices, tgtranges, srcranges,
                            **extra_kwargs)

        for i in range(srcranges.shape[0] - 1):
            itgt = np.s_[tgtranges[i]:tgtranges[i + 1]]
            isrc = np.s_[srcranges[i]:srcranges[i + 1]]
            block = np.ix_(tgtindices[itgt], srcindices[isrc])

            eps = 1.0e-10 * la.norm(mat[block])
            assert la.norm(blk[itgt, isrc] - mat[block]) < eps


# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
