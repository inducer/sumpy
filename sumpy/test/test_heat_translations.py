from __future__ import annotations


__copyright__ = "Copyright (C) 2012 Chaoqi Lin"

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
from functools import partial

import numpy as np
import numpy.linalg as la
import pytest

from arraycontext import ArrayContextFactory, pytest_generate_tests_for_array_contexts
from pytools.convergence import EOCRecorder

import sumpy.toys as t
from sumpy.array_context import PytestPyOpenCLArrayContextFactory, _acf  # noqa: F401
from sumpy.expansion.local import (
    LinearPDEConformingVolumeTaylorLocalExpansion,
    VolumeTaylorLocalExpansion,
)
from sumpy.expansion.m2l import NonFFTM2LTranslationClassFactory
from sumpy.expansion.multipole import (
    LinearPDEConformingVolumeTaylorMultipoleExpansion,
    VolumeTaylorMultipoleExpansion,
)
from sumpy.kernel import HeatKernel


logger = logging.getLogger(__name__)

pytest_generate_tests = pytest_generate_tests_for_array_contexts([
    PytestPyOpenCLArrayContextFactory,
    ])


# {{{ test_heat_m2m

@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize(("knl", "mpole_expn_class"), [
    (HeatKernel(1), LinearPDEConformingVolumeTaylorMultipoleExpansion),
    (HeatKernel(1), VolumeTaylorMultipoleExpansion),
    ])
def test_heat_m2m(
            actx_factory: ArrayContextFactory,
            knl,
            mpole_expn_class,
            alpha,
            order):
    # h-convergence of the heat-kernel M2M translation.
    # src_center = (1, 0.5), shrinks with h (half-sizes h_x, h_t).
    # tgt_center = (0, 2.5), fixed (half-sizes 1, 0.5).
    # M2M from src_center to a shifted center c5 = src_center + (-h_x, h_t).
    actx = actx_factory()

    extra_kwargs = {"alpha": alpha}

    t_sep = 1.0
    L = np.sqrt(4 * alpha * t_sep)  # noqa: N806

    src_center = np.array([1.0, 0.5])
    tgt_center = np.array([0.0, 2.5])

    grid = np.linspace(-1.0, 1.0, 5)
    xs, ts = np.meshgrid(grid, grid, indexing="xy")
    targets = np.vstack([
        tgt_center[0] + 1.0 * xs.ravel(),
        tgt_center[1] + 0.5 * ts.ravel(),
    ])

    toy_ctx = t.ToyContext(
            kernel=knl,
            mpole_expn_class=mpole_expn_class,
            extra_kernel_kwargs=extra_kwargs,
    )

    h_values = [1/4, 1/8, 1/16, 1/32]
    rscale = 1 / order

    eoc_rec = EOCRecorder()
    for h in h_values:
        h_x = h * L
        h_t = h * t_sep

        src_grid = np.linspace(-1.0, 1.0, 11)
        sxs, sts = np.meshgrid(src_grid, src_grid, indexing="xy")
        sources = np.vstack([
            src_center[0] + h_x * sxs.ravel(),
            src_center[1] + h_t * sts.ravel(),
        ])
        strengths = np.ones(sources.shape[1])

        pt_src = t.PointSources(toy_ctx, sources, weights=strengths)

        c5 = src_center + np.array([-h_x, h_t])

        p2m = t.multipole_expand(actx, pt_src, src_center,
                                 order=order, rscale=rscale)
        m2m = t.multipole_expand(actx, p2m, c5, order=order, rscale=rscale)
        p2m_direct = t.multipole_expand(actx, pt_src, c5,
                                        order=order, rscale=rscale)

        m2m_vals = np.asarray(m2m.eval(actx, targets)).ravel()
        p2m_direct_vals = np.asarray(p2m_direct.eval(actx, targets)).ravel()
        err = la.norm(m2m_vals - p2m_direct_vals) / la.norm(p2m_direct_vals)
        eoc_rec.add_data_point(h, err)

    logger.info("knl %s order %d", knl, order)
    logger.info("M2M:\n%s", eoc_rec)

    tgt_order = order + 1
    slack = 0.5
    assert eoc_rec.order_estimate() > tgt_order - slack

# }}}


# {{{ test_heat_l2l

@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize(("knl", "local_expn_class"), [
    (HeatKernel(1), LinearPDEConformingVolumeTaylorLocalExpansion),
    (HeatKernel(1), VolumeTaylorLocalExpansion),
    ])
def test_heat_l2l(
            actx_factory: ArrayContextFactory,
            knl,
            local_expn_class,
            alpha,
            order):
    # h-convergence of the heat-kernel L2L translation.
    # src_center = (0, 0.5), fixed (half-widths 1, 0.5).
    # tgt_center = (0, 2.5), shrinks with h (half-sizes h_x, h_t).
    # L2L from tgt_center to a shifted center c2 = tgt_center - (h_x, h_t).
    actx = actx_factory()

    extra_kwargs = {"alpha": alpha}

    src_center = np.array([0.0, 0.5])
    src_hx = 1.0
    src_ht = 0.5
    tgt_center = np.array([0.0, 2.5])

    t_sep = 2 * src_ht
    L = np.sqrt(4 * alpha * t_sep)  # noqa: N806

    rng = np.random.default_rng(0)
    src_grid = np.linspace(-1.0, 1.0, 11)
    sxs, sts = np.meshgrid(src_grid, src_grid, indexing="xy")
    sources = np.vstack([
        src_center[0] + src_hx * sxs.ravel(),
        src_center[1] + src_ht * sts.ravel(),
    ])
    strengths = rng.random(sources.shape[1])

    toy_ctx = t.ToyContext(
            kernel=knl,
            local_expn_class=local_expn_class,
            extra_kernel_kwargs=extra_kwargs,
    )
    pt_src = t.PointSources(toy_ctx, sources, weights=strengths)

    rscale = 1 / order
    p2l = t.local_expand(actx, pt_src, tgt_center,
                         order=order, rscale=rscale)

    h_values = [1/4, 1/8, 1/16, 1/32]

    eoc_rec = EOCRecorder()
    for h in h_values:
        h_x = h * L
        h_t = h * t_sep
        c2 = tgt_center - np.array([h_x, h_t])

        l2l = t.local_expand(actx, p2l, c2, order=order, rscale=rscale)
        p2l_direct = t.local_expand(actx, pt_src, c2,
                                    order=order, rscale=rscale)

        tgt_grid = np.linspace(-1.0, 1.0, 5)
        txs, tts = np.meshgrid(tgt_grid, tgt_grid, indexing="xy")
        targets = np.vstack([
            tgt_center[0] + h_x * txs.ravel(),
            tgt_center[1] + h_t * tts.ravel(),
        ])
        l2l_vals = np.asarray(l2l.eval(actx, targets)).ravel()
        p2l_direct_vals = np.asarray(p2l_direct.eval(actx, targets)).ravel()
        err = la.norm(l2l_vals - p2l_direct_vals) / la.norm(p2l_direct_vals)
        eoc_rec.add_data_point(h, err)

    logger.info("knl %s order %d", knl, order)
    logger.info("L2L:\n%s", eoc_rec)

    tgt_order = order + 1
    slack = 0.5
    assert eoc_rec.order_estimate() > tgt_order - slack

# }}}


# {{{ test_heat_m2l

@pytest.mark.parametrize("order", [4])
@pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize(("knl", "local_expn_class", "mpole_expn_class"), [
    (HeatKernel(1),
     LinearPDEConformingVolumeTaylorLocalExpansion,
     LinearPDEConformingVolumeTaylorMultipoleExpansion),
    ])
def test_heat_m2l(
            actx_factory: ArrayContextFactory,
            knl,
            local_expn_class,
            mpole_expn_class,
            alpha,
            order):
    # h-convergence of the heat-kernel M2L translation.
    # src_center = (0, 0.5), fixed (half-widths 1, 0.5).
    # tgt_center = (0, 2.5), shrinks with h (half-sizes h_x, h_t).
    # Local expansion formed at c2 = tgt_center - (h_x, h_t).
    actx = actx_factory()

    extra_kwargs = {"alpha": alpha}

    src_center = np.array([0.0, 0.5])
    src_hx = 1.0
    src_ht = 0.5
    tgt_center = np.array([0.0, 2.5])

    t_sep = 2 * src_ht
    L = np.sqrt(4 * alpha * t_sep)  # noqa: N806

    rng = np.random.default_rng(0)
    src_grid = np.linspace(-1.0, 1.0, 11)
    sxs, sts = np.meshgrid(src_grid, src_grid, indexing="xy")
    sources = np.vstack([
        src_center[0] + src_hx * sxs.ravel(),
        src_center[1] + src_ht * sts.ravel(),
    ])
    strengths = rng.random(sources.shape[1])

    m2l_factory = NonFFTM2LTranslationClassFactory()
    m2l_translation = m2l_factory.get_m2l_translation_class(knl, local_expn_class)()
    toy_ctx = t.ToyContext(
            kernel=knl,
            local_expn_class=partial(local_expn_class,
                m2l_translation_override=m2l_translation),
            mpole_expn_class=mpole_expn_class,
            extra_kernel_kwargs=extra_kwargs,
    )
    pt_src = t.PointSources(toy_ctx, sources, weights=strengths)

    rscale = 1 / order
    p2m = t.multipole_expand(actx, pt_src, src_center,
                             order=order, rscale=rscale)

    h_values = [1/4, 1/8, 1/16, 1/32]

    eoc_rec = EOCRecorder()
    for h in h_values:
        h_x = h * L
        h_t = h * t_sep
        c2 = tgt_center - np.array([h_x, h_t])

        m2l = t.local_expand(actx, p2m, c2, order=order, rscale=rscale)

        tgt_grid = np.linspace(-1.0, 1.0, 5)
        txs, tts = np.meshgrid(tgt_grid, tgt_grid, indexing="xy")
        targets = np.vstack([
            tgt_center[0] + h_x * txs.ravel(),
            tgt_center[1] + h_t * tts.ravel(),
        ])
        m2l_vals = np.asarray(m2l.eval(actx, targets)).ravel()
        p2m_vals = np.asarray(p2m.eval(actx, targets)).ravel()
        err = la.norm(m2l_vals - p2m_vals) / la.norm(p2m_vals)
        eoc_rec.add_data_point(h, err)

    logger.info("knl %s order %d", knl, order)
    logger.info("M2L:\n%s", eoc_rec)

    tgt_order = order + 1
    slack = 0.5
    assert eoc_rec.order_estimate() > tgt_order - slack

# }}}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])
