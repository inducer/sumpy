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
from numbers import Number

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

    @memoize_method
    def get_m2p(self, order):
        from sumpy import E2PFromSingleBox
        return E2PFromSingleBox(self.cl_context,
                self.mpole_expn_class(self.kernel, order),
                [self.kernel])

    @memoize_method
    def get_l2p(self, order):
        from sumpy import E2PFromSingleBox
        return E2PFromSingleBox(self.cl_context,
                self.local_expn_class(self.kernel, order),
                [self.kernel])

    @memoize_method
    def get_m2m(self, from_order, to_order):
        from sumpy import E2EFromCSR
        return E2EFromCSR(self.cl_context,
                self.mpole_expn_class(self.kernel, from_order),
                self.mpole_expn_class(self.kernel, to_order))

    @memoize_method
    def get_m2l(self, from_order, to_order):
        from sumpy import E2EFromCSR
        return E2EFromCSR(self.cl_context,
                self.mpole_expn_class(self.kernel, from_order),
                self.local_expn_class(self.kernel, to_order))

    @memoize_method
    def get_l2l(self, from_order, to_order):
        from sumpy import E2EFromCSR
        return E2EFromCSR(self.cl_context,
                self.local_expn_class(self.kernel, from_order),
                self.local_expn_class(self.kernel, to_order))

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

    boxes = np.array([0], dtype=np.int32)

    box_target_starts = np.array([0], dtype=np.int32)
    box_target_counts_nonchild = np.array([ntargets], dtype=np.int32)

    toy_ctx = psource.toy_ctx
    centers = np.array(psource.center, dtype=np.float64).reshape(
            toy_ctx.kernel.dim, 1)

    coeffs = np.array([psource.coeffs])
    evt, (pot,) = e2p(
            toy_ctx.queue,
            src_expansions=coeffs,
            src_base_ibox=0,
            target_boxes=boxes,
            box_target_starts=box_target_starts,
            box_target_counts_nonchild=box_target_counts_nonchild,
            centers=centers,
            targets=targets,
            #flags="print_hl_cl",
            out_host=True, **toy_ctx.extra_source_kwargs)

    return pot


def _e2e(psource, to_center, to_order, e2e, expn_class):
    toy_ctx = psource.toy_ctx

    target_boxes = np.array([1], dtype=np.int32)
    src_box_starts = np.array([0, 1], dtype=np.int32)
    src_box_lists = np.array([0], dtype=np.int32)

    centers = (np.array(
            [
                # box 0: source
                psource.center,

                # box 1: target
                to_center,
                ],
            dtype=np.float64)).T.copy()

    coeffs = np.array([psource.coeffs])

    evt, (to_coeffs,) = e2e(
            toy_ctx.queue,
            src_expansions=coeffs,
            src_base_ibox=0,
            tgt_base_ibox=0,
            ntgt_level_boxes=2,

            target_boxes=target_boxes,

            src_box_starts=src_box_starts,
            src_box_lists=src_box_lists,
            centers=centers,
            #flags="print_hl_cl",
            out_host=True, **toy_ctx.extra_source_kwargs)

    return expn_class(toy_ctx, to_center, to_order, to_coeffs[1])

# }}}


# {{{ potential source classes

class PotentialSource(object):
    def __init__(self, toy_ctx):
        self.toy_ctx = toy_ctx

    def eval(self, targets):
        raise NotImplementedError()

    def __neg__(self):
        return -1*self

    def __add__(self, other):
        if isinstance(other, (Number, np.number)):
            other = ConstantPotential(self.toy_ctx, other)
        elif not isinstance(other, PotentialSource):
            return NotImplemented

        return Sum((self, other))

    __radd__ = __add__

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        if isinstance(other, (Number, np.number)):
            other = ConstantPotential(self.toy_ctx, other)
        elif not isinstance(other, PotentialSource):
            return NotImplemented

        return Product((self, other))

    __rmul__ = __mul__


class ConstantPotential(PotentialSource):
    def __init__(self, toy_ctx, value):
        super(ConstantPotential, self).__init__(toy_ctx)
        self.value = np.array(value)

    def eval(self, targets):
        pot = np.empty(targets.shape[-1], dtype=self.value.dtype)
        pot.fill(self.value)
        return pot


class OneOnBallPotential(PotentialSource):
    def __init__(self, toy_ctx, center, radius):
        super(OneOnBallPotential, self).__init__(toy_ctx)
        self.center = np.asarray(center)
        self.radius = radius

    def eval(self, targets):
        dist_vec = targets - self.center[:, np.newaxis]
        return (np.sum(dist_vec**2, axis=0) < self.radius).astype(np.float64)


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
        self.center = np.asarray(center)
        self.order = order
        self.coeffs = coeffs


class MultipoleExpansion(ExpansionPotentialSource):
    def eval(self, targets):
        return _e2p(self, targets, self.toy_ctx.get_m2p(self.order))


class LocalExpansion(ExpansionPotentialSource):
    def eval(self, targets):
        return _e2p(self, targets, self.toy_ctx.get_l2p(self.order))


class PotentialExpressionNode(PotentialSource):
    def __init__(self, psources):
        from pytools import single_valued
        super(PotentialExpressionNode, self).__init__(
                single_valued(psource.toy_ctx for psource in psources))

        self.psources = psources


class Sum(PotentialExpressionNode):
    def eval(self, targets):
        result = 0
        for psource in self.psources:
            result = result + psource.eval(targets)

        return result


class Product(PotentialExpressionNode):
    def eval(self, targets):
        result = 1
        for psource in self.psources:
            result = result * psource.eval(targets)

        return result

# }}}


def multipole_expand(psource, center, order=None):
    if isinstance(psource, PointSources):
        if order is None:
            raise ValueError("order may not be None")

        return _p2e(psource, center, order, psource.toy_ctx.get_p2m(order),
                MultipoleExpansion)

    elif isinstance(psource, MultipoleExpansion):
        if order is None:
            order = psource.order

        return _e2e(psource, center, order,
                psource.toy_ctx.get_m2m(psource.order, order),
                MultipoleExpansion)

    else:
        raise TypeError("do not know how to expand '%s'"
                % type(psource).__name__)


def local_expand(psource, center, order=None):
    if isinstance(psource, PointSources):
        if order is None:
            raise ValueError("order may not be None")

        return _p2e(psource, center, order, psource.toy_ctx.get_p2l(order),
                LocalExpansion)

    elif isinstance(psource, MultipoleExpansion):
        if order is None:
            order = psource.order

        return _e2e(psource, center, order,
                psource.toy_ctx.get_m2l(psource.order, order),
                LocalExpansion)

    elif isinstance(psource, LocalExpansion):
        if order is None:
            order = psource.order

        return _e2e(psource, center, order,
                psource.toy_ctx.get_l2l(psource.order, order),
                LocalExpansion)

    else:
        raise TypeError("do not know how to expand '%s'"
                % type(psource).__name__)


def logplot(fp, psource, **kwargs):
    fp.show_scalar_in_matplotlib(
            np.log10(np.abs(psource.eval(fp.points) + 1e-15)), **kwargs)


def restrict_inner(psource, radius, center=None):
    if center is None:
        center = psource.center

    return psource * OneOnBallPotential(psource.toy_ctx, center, radius)


def restrict_outer(psource, radius, center=None):
    if center is None:
        center = psource.center

    return psource * (1-OneOnBallPotential(psource.toy_ctx, center, radius))


def l_inf(psource, radius, center=None, npoints=100):
    if center is None:
        center = psource.center

    restr = psource * OneOnBallPotential(psource.toy_ctx, center, radius)

    from sumpy.visualization import FieldPlotter
    fp = FieldPlotter(center, extent=2*radius, npoints=npoints)
    return np.max(np.abs(restr.eval(fp.points)))


# vim: foldmethod=marker
