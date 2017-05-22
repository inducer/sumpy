from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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

import six  # noqa: F401
from six.moves import range  # noqa: F401

from pytools import memoize_method

import numpy as np  # noqa: F401
import loopy as lp  # noqa: F401
import pyopencl as cl

import logging
logger = logging.getLogger(__name__)


# {{{ context

class ToyContext(object):
    def __init__(self, cl_context, kernel,
            mpole_expn_class=None,
            local_expn_class=None,
            extra_source_kwargs=None):
        self.cl_context = cl_context
        self.queue = cl.CommandQueue(self.cl_context)
        self.kernel = kernel

        if mpole_expn_class is None:
            from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
            mpole_expn_class = VolumeTaylorMultipoleExpansion
        if local_expn_class is None:
            from sumpy.expansion.local import VolumeTaylorLocalExpansion
            local_expn_class = VolumeTaylorLocalExpansion

        if extra_source_kwargs is None:
            extra_source_kwargs = {}

        self.mpole_expn_class = mpole_expn_class
        self.local_expn_class = local_expn_class
        self.extra_source_kwargs = extra_source_kwargs

    @memoize_method
    def get_p2p(self):
        from sumpy.p2p import P2P
        return P2P(self.cl_context, [self.kernel], exclude_self=False)

    @memoize_method
    def get_p2m(self, order):
        from sumpy import P2EFromSingleBox
        return P2EFromSingleBox(self.cl_context,
                self.mpole_expn_class(self.kernel, order),
                [self.kernel])

    @memoize_method
    def get_p2l(self, order):
        from sumpy import P2EFromSingleBox
        return P2EFromSingleBox(self.cl_context,
                self.local_expn_class(self.kernel, order),
                [self.kernel])

# }}}


# {{{ helpers

def _p2e(psource, center, order, p2e, expn_class):
    source_boxes = np.array([0], dtype=np.int32)
    box_source_starts = np.array([0], dtype=np.int32)
    box_source_counts_nonchild = np.array(
            [psource.points.shape[-1]], dtype=np.int32)

    toy_ctx = psource.toy_ctx
    center = np.asarray(center)
    centers = np.array(center, dtype=np.float64).reshape(
            toy_ctx.kernel.dim, 1)

    evt, (coeffs,) = p2e(toy_ctx.queue,
            source_boxes=source_boxes,
            box_source_starts=box_source_starts,
            box_source_counts_nonchild=box_source_counts_nonchild,
            centers=centers,
            sources=psource.points,
            strengths=psource.weights,
            nboxes=1,
            tgt_base_ibox=0,

            #flags="print_hl_cl",
            out_host=True, **toy_ctx.extra_source_kwargs)

    return expn_class(toy_ctx, center, order, coeffs[0])


def _e2p(psource, targets, e2p):
    ntargets = targets.shape[-1]

    _boxes = np.array([0], dtype=np.int32)

    box_target_starts = np.array([0], dtype=np.int32)
    box_target_counts_nonchild = np.array([ntargets], dtype=np.int32)

    coeffs = np.array([psource.coeffs])
    toy_ctx = psource.toy_ctx
    evt, (pot,) = e2p(
            toy_ctx.queue,
            src_expansions=coeffs,
            src_base_ibox=0,
            target_boxes=source_boxes,
            box_target_starts=box_target_starts,
            box_target_counts_nonchild=box_target_counts_nonchild,
            centers=centers,
            targets=targets,
            #flags="print_hl_cl",
            out_host=True, **extra_kwargs)

# }}}


class PotentialSource(object):
    def __init__(self, toy_ctx):
        self.toy_ctx = toy_ctx

    def eval(self, targets):
        raise NotImplementedError()

    def __add__(self, other):
        if not isinstance(other, PointSources):
            raise TypeError()

        return LinearCombination((1, 1), (self, other))

    def __sub__(self, other):
        if not isinstance(other, PointSources):
            raise TypeError()

        return LinearCombination((1, -1), (self, other))


class PointSources(PotentialSource):
    """
    .. attribute:: points

        ``[ndim, npoints]``
    """

    def __init__(self, toy_ctx, points, weights):
        super(PointSources, self).__init__(toy_ctx)

        self.points = points
        self.weights = weights

    def eval(self, targets):
        evt, (potential,) = self.toy_ctx.get_p2p()(
                self.toy_ctx.queue, targets, self.points, [self.weights],
                out_host=True)

        return potential


class ExpansionPotentialSource(PotentialSource):
    def __init__(self, toy_ctx, center, order, coeffs):
        super(ExpansionPotentialSource, self).__init__(toy_ctx)


class MultipoleExpansion(ExpansionPotentialSource):
    pass


class LocalExpansion(ExpansionPotentialSource):
    def eval(self, targets):
        raise NotImplementedError()


def multipole_expand(psource, center, order):
    if isinstance(psource, PointSources):
        return _p2e(psource, center, order, psource.toy_ctx.get_p2m(order),
                MultipoleExpansion)

    elif isinstance(psource, MultipoleExpansion):
        raise NotImplementedError()
    else:
        raise TypeError("do not know how to expand '%s'"
                % type(psource).__name__)


def local_expand(psource, center, order):
    if isinstance(psource, PointSources):
        return _p2e(psource, center, order, psource.toy_ctx.get_p2l(order),
                LocalExpansion)

    elif isinstance(psource, MultipoleExpansion):
        raise NotImplementedError()
    elif isinstance(psource, LocalExpansion):
        raise NotImplementedError()
    else:
        raise TypeError("do not know how to expand '%s'"
                % type(psource).__name__)


class LinearCombination(PotentialSource):
    def __init__(self, coeffs, psources):
        from pytools import single_valued
        super(LinearCombination, self).__init__(
                single_valued(psource.toy_ctx for psource in psources))

        self.coeffs = coeffs
        self.psources = psources

    def eval(self, targets):
        result = 0
        for coeff, psource in zip(self.coeffs, self.psources):
            result += coeff * psource.eval(targets)

        return result


def logplot(fp, psource, **kwargs):
    fp.show_scalar_in_matplotlib(
            np.log10(np.abs(psource.eval(fp.points) + 1e-15)), **kwargs)

# vim: foldmethod=marker
