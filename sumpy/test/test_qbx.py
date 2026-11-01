from __future__ import annotations


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

import numpy as np
import pytest

from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts

from sumpy.array_context import (  # ruff:ignore[unused-import]
    PytestPyOpenCLArrayContextFactory,
    _acf,
)
from sumpy.expansion.local import (
    LineTaylorLocalExpansion,
    ODELineTaylorLocalExpansion,
    VolumeTaylorLocalExpansion,
)
from sumpy.kernel import BiharmonicKernel, LaplaceKernel


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_direct_qbx_vs_eigval

@pytest.mark.parametrize(("expn_class", "kernel_cls"), [
    (LineTaylorLocalExpansion, LaplaceKernel),
    (ODELineTaylorLocalExpansion, LaplaceKernel),
    (VolumeTaylorLocalExpansion, LaplaceKernel),
    (ODELineTaylorLocalExpansion, BiharmonicKernel),
])
def test_direct_qbx_vs_eigval(
            actx_factory: ArrayContextFactory,
            expn_class,
            kernel_cls,
            visualize=False):
    """This evaluates a single layer potential on a circle using a known
    eigenvalue/eigenvector combination.
    """
    if visualize:
        logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    kernel = kernel_cls(2)

    order = 12

    from sumpy.qbx import LayerPotential

    lpot = LayerPotential(
            expansion=expn_class(kernel, order),
            target_kernels=(kernel,),
            source_kernels=(kernel,))

    mode_nr = 25

    from pytools.convergence import EOCRecorder

    eocrec = EOCRecorder()

    for n in [400, 500, 600]:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        unit_circle = np.exp(1j * t)
        unit_circle = np.array([unit_circle.real, unit_circle.imag])

        sigma = np.cos(mode_nr * t)
        if kernel_cls is LaplaceKernel:
            eigval = 1/(2*mode_nr)
        else:
            eigval = 1/(4*mode_nr*(mode_nr**2-1))

        result_ref = eigval * sigma

        h = 2 * np.pi / n

        targets = actx.from_numpy(unit_circle)
        sources = actx.from_numpy(unit_circle)

        radius = 7 * h
        centers = actx.from_numpy((1 - radius) * unit_circle)
        expansion_radii = actx.from_numpy(radius * np.ones(n))
        strengths = (actx.from_numpy(sigma * h),)

        result_qbx, = lpot(
                actx,
                targets, sources, centers, strengths,
                expansion_radii=expansion_radii)
        result_qbx = actx.to_numpy(result_qbx)

        error = np.linalg.norm(result_ref - result_qbx, np.inf)
        eocrec.add_data_point(h, error)

    logger.info("eoc:\n%s", eocrec)

    slack = 1.5
    assert eocrec.order_estimate() > order - slack

# }}}


# {{{ test_direct_qbx_vs_eigval_with_tgt_deriv

@pytest.mark.parametrize("expn_class", [
    LineTaylorLocalExpansion,
    ODELineTaylorLocalExpansion,
    VolumeTaylorLocalExpansion,
])
def test_direct_qbx_vs_eigval_with_tgt_deriv(
        actx_factory, expn_class, visualize=False):
    """Check the interior normal derivative of a single layer on the unit circle.
    """
    if visualize:
        logging.basicConfig(level=logging.INFO)

    actx = actx_factory()

    from sumpy.kernel import AxisTargetDerivative
    kernel = LaplaceKernel(2)

    order = 8
    mode_nr = 15
    eigval = 0.5

    from sumpy.qbx import LayerPotential

    target_kernels = (
        AxisTargetDerivative(0, kernel),
        AxisTargetDerivative(1, kernel),
    )
    if expn_class is ODELineTaylorLocalExpansion:
        expansion = expn_class(kernel, order, target_kernels=target_kernels)
    else:
        expansion = expn_class(kernel, order)
    lpot = LayerPotential(
        expansion=expansion,
        target_kernels=target_kernels,
        source_kernels=(kernel,))

    from pytools.convergence import EOCRecorder

    eocrec = EOCRecorder()

    for n in [200, 300, 400]:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        unit_circle = np.exp(1j * t)
        unit_circle = np.array([unit_circle.real, unit_circle.imag])

        sigma = np.cos(mode_nr * t)
        result_ref = eigval * sigma

        h = 2 * np.pi / n

        targets = actx.from_numpy(unit_circle)
        sources = actx.from_numpy(unit_circle)

        radius = 7 * h
        centers = actx.from_numpy((1 - radius) * unit_circle)
        expansion_radii = actx.from_numpy(radius * np.ones(n))
        strengths = (actx.from_numpy(sigma * h),)

        result_qbx_dx, result_qbx_dy = lpot(
                actx,
                targets, sources, centers, strengths,
                expansion_radii=expansion_radii)

        result_qbx_dx = actx.to_numpy(result_qbx_dx)
        result_qbx_dy = actx.to_numpy(result_qbx_dy)

        normals = unit_circle
        result_qbx = normals[0] * result_qbx_dx + normals[1] * result_qbx_dy

        error = np.linalg.norm(result_ref - result_qbx, np.inf)
        eocrec.add_data_point(h, error)

    if expn_class is not LineTaylorLocalExpansion:
        logger.info("eoc:\n%s", eocrec)

        slack = 1.5
        assert eocrec.order_estimate() > order - slack

# }}}


# {{{ test_direct_qbx_vs_eigval_hypersingular

def test_direct_qbx_vs_eigval_hypersingular(actx_factory: ArrayContextFactory):
    """Check the normal derivative of a double layer on the unit circle.
    """
    actx = actx_factory()

    from sumpy.kernel import AxisSourceDerivative, AxisTargetDerivative
    kernel = LaplaceKernel(2)

    order = 8
    mode_nr = 15
    eigval = -mode_nr/2

    source_kernels = (
        AxisSourceDerivative(0, kernel),
        AxisSourceDerivative(1, kernel),
    )
    target_kernels = (
        AxisTargetDerivative(0, kernel),
        AxisTargetDerivative(1, kernel),
    )
    expansion = ODELineTaylorLocalExpansion(
        kernel, order, target_kernels=target_kernels)

    from sumpy.qbx import LayerPotential
    lpot = LayerPotential(
        expansion=expansion,
        target_kernels=target_kernels,
        source_kernels=source_kernels)

    from pytools.convergence import EOCRecorder
    eocrec = EOCRecorder()

    for n in [200, 300, 400]:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        unit_circle = np.exp(1j * t)
        unit_circle = np.array([unit_circle.real, unit_circle.imag])
        normals = unit_circle

        sigma = np.cos(mode_nr * t)
        result_ref = eigval * sigma

        h = 2 * np.pi / n
        targets = actx.from_numpy(unit_circle)
        sources = actx.from_numpy(unit_circle)

        radius = 7 * h
        centers = actx.from_numpy((1 - radius) * unit_circle)
        expansion_radii = actx.from_numpy(radius * np.ones(n))
        strengths = tuple(actx.from_numpy(sigma * h * normal)
                          for normal in normals)

        result_qbx_dx, result_qbx_dy = lpot(
            actx, targets, sources, centers, strengths,
            expansion_radii=expansion_radii)
        result_qbx_dx = actx.to_numpy(result_qbx_dx)
        result_qbx_dy = actx.to_numpy(result_qbx_dy)
        result_qbx = normals[0] * result_qbx_dx + normals[1] * result_qbx_dy

        error = np.linalg.norm(result_ref - result_qbx, np.inf)
        eocrec.add_data_point(h, error)

    logger.info("eoc:\n%s", eocrec)

    slack = 1.5
    assert eocrec.order_estimate() > order - slack

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
