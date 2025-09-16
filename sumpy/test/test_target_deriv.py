from __future__ import annotations


__copyright__ = """
Copyright (C) 2025 Shawn Lin
Copyright (C) 2025 University of Illinois Board of Trustees
"""

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
from typing import TYPE_CHECKING

import numpy as np
import pytest

from arraycontext import (
    PyOpenCLArrayContext,
    pytest_generate_tests_for_array_contexts,
)
from pytools.convergence import EOCRecorder

from sumpy.array_context import (  # noqa: F401
    PytestPyOpenCLArrayContextFactory,
    _acf,  # pyright: ignore[reportUnusedImport]
)
from sumpy.expansion.local import LineTaylorLocalExpansion
from sumpy.kernel import AxisTargetDerivative, Kernel, LaplaceKernel
from sumpy.test.geometries import make_starfish


if TYPE_CHECKING:
    from arraycontext import ArrayContextFactory

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


@pytest.mark.parametrize("knl", [LaplaceKernel(2)])
def test_lpot_dx_jump_relation_convergence(
            actx_factory: ArrayContextFactory,
            knl: Kernel):
    """Test convergence of jump relations for single layer potential derivatives."""

    actx = actx_factory()
    if not isinstance(actx, PyOpenCLArrayContext):
        pytest.skip()

    qbx_order = 5

    ntargets = 20
    target_geo = make_starfish(npoints=ntargets)
    targets_h = target_geo.nodes
    targets = actx.from_numpy(targets_h)

    from sumpy.qbx import LayerPotential
    expansion = LineTaylorLocalExpansion(knl, qbx_order)
    lplot_dx = LayerPotential(
        actx.context,
        expansion=expansion,
        target_kernels=(AxisTargetDerivative(0, knl),),
        source_kernels=(knl,)
    )
    lplot_dy = LayerPotential(
        actx.context,
        expansion=expansion,
        target_kernels=(AxisTargetDerivative(1, knl),),
        source_kernels=(knl,)
    )
    eocrec = EOCRecorder()

    for nsources in [320, 640, 1280, 2560]:
        source_geo = make_starfish(npoints=nsources)
        sources = actx.from_numpy(source_geo.nodes)

        weights_nodes_h = source_geo.area_elements * source_geo.weights
        weights_nodes = actx.from_numpy(weights_nodes_h)

        expansion_radii_h = 4 * target_geo.area_elements / nsources
        centers_in = actx.from_numpy(
                            targets_h - target_geo.normals * expansion_radii_h)
        centers_out = actx.from_numpy(
                            targets_h + target_geo.normals * expansion_radii_h)

        strengths = (weights_nodes,)
        _, (eval_in_dx,) = lplot_dx(
            actx.queue,
            targets, sources, centers_in, strengths,
            expansion_radii=expansion_radii_h
        )

        _, (eval_in_dy,) = lplot_dy(
            actx.queue,
            targets, sources, centers_in, strengths,
            expansion_radii=expansion_radii_h
        )

        _, (eval_out_dx,) = lplot_dx(
            actx.queue,
            targets, sources, centers_out, strengths,
            expansion_radii=expansion_radii_h
        )

        _, (eval_out_dy,) = lplot_dy(
            actx.queue,
            targets, sources, centers_out, strengths,
            expansion_radii=expansion_radii_h
        )

        eval_in_dx = actx.to_numpy(eval_in_dx)
        eval_in_dy = actx.to_numpy(eval_in_dy)
        eval_out_dx = actx.to_numpy(eval_out_dx)
        eval_out_dy = actx.to_numpy(eval_out_dy)

        eval_in = eval_in_dx * target_geo.normals[0] + \
                   eval_in_dy * target_geo.normals[1]
        eval_out = eval_out_dx * target_geo.normals[0] + \
                   eval_out_dy * target_geo.normals[1]

        # check jump relation: S'_int - S'_ext = sigma (=1 for constant density)
        jump_error = np.abs(eval_in - eval_out - 1)

        h_max = 1/nsources
        eocrec.add_data_point(h_max, np.max(jump_error))

    print(eocrec)
    assert eocrec.order_estimate() > qbx_order - 1.5


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
