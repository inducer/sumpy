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
            expansion_factory=None,
            extra_source_kwargs=None):
        self.cl_context = cl_context
        self.queue = cl.CommandQueue(self.cl_context)
        self.kernel = kernel

        if expansion_factory is None:
            from sumpy.expansion import DefaultExpansionFactory
            expansion_factory = DefaultExpansionFactory()
        if mpole_expn_class is None:
            mpole_expn_class = \
                    expansion_factory.get_multipole_expansion_class(kernel)
        if local_expn_class is None:
            local_expn_class = \
                    expansion_factory.get_local_expansion_class(kernel)

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

def _p2e(psource, center, order, p2e, expn_class, expn_kwargs):
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

    return expn_class(toy_ctx, center, order, coeffs[0], derived_from=psource,
            **expn_kwargs)


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


def _e2e(psource, to_center, to_order, e2e, expn_class, expn_kwargs):
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

    return expn_class(toy_ctx, to_center, to_order, to_coeffs[1],
            derived_from=psource, **expn_kwargs)

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
        return (np.sum(dist_vec**2, axis=0) < self.radius**2).astype(np.float64)


class PointSources(PotentialSource):
    """
    .. attribute:: points

        ``[ndim, npoints]``
    """

    def __init__(self, toy_ctx, points, weights, center=None):
        super(PointSources, self).__init__(toy_ctx)

        self.points = points
        self.weights = weights
        self._center = center

    def eval(self, targets):
        evt, (potential,) = self.toy_ctx.get_p2p()(
                self.toy_ctx.queue, targets, self.points, [self.weights],
                out_host=True, **self.toy_ctx.extra_source_kwargs)

        return potential

    @property
    def center(self):
        if self._center is not None:
            return self._center

        return np.average(self.points, axis=1)


class ExpansionPotentialSource(PotentialSource):
    """
    .. attribute:: radius

        Not used mathematically. Just for visualization, purely advisory.
    """
    def __init__(self, toy_ctx, center, order, coeffs, derived_from,
            radius=None, expn_style=None):
        super(ExpansionPotentialSource, self).__init__(toy_ctx)
        self.center = np.asarray(center)
        self.order = order
        self.coeffs = coeffs

        self.derived_from = derived_from
        self.radius = radius
        self.expn_style = expn_style


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

    @property
    def center(self):
        for psource in self.psources:
            try:
                return psource.center
            except AttributeError:
                pass

        raise ValueError("no psource with a center found")


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


def multipole_expand(psource, center, order=None, **expn_kwargs):
    if isinstance(psource, PointSources):
        if order is None:
            raise ValueError("order may not be None")

        return _p2e(psource, center, order, psource.toy_ctx.get_p2m(order),
                MultipoleExpansion, expn_kwargs)

    elif isinstance(psource, MultipoleExpansion):
        if order is None:
            order = psource.order

        return _e2e(psource, center, order,
                psource.toy_ctx.get_m2m(psource.order, order),
                MultipoleExpansion, expn_kwargs)

    else:
        raise TypeError("do not know how to expand '%s'"
                % type(psource).__name__)


def local_expand(psource, center, order=None, **expn_kwargs):
    if isinstance(psource, PointSources):
        if order is None:
            raise ValueError("order may not be None")

        return _p2e(psource, center, order, psource.toy_ctx.get_p2l(order),
                LocalExpansion, expn_kwargs)

    elif isinstance(psource, MultipoleExpansion):
        if order is None:
            order = psource.order

        return _e2e(psource, center, order,
                psource.toy_ctx.get_m2l(psource.order, order),
                LocalExpansion, expn_kwargs)

    elif isinstance(psource, LocalExpansion):
        if order is None:
            order = psource.order

        return _e2e(psource, center, order,
                psource.toy_ctx.get_l2l(psource.order, order),
                LocalExpansion, expn_kwargs)

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


def l_inf(psource, radius, center=None, npoints=100, debug=False):
    if center is None:
        center = psource.center

    restr = psource * OneOnBallPotential(psource.toy_ctx, center, radius)

    from sumpy.visualization import FieldPlotter
    fp = FieldPlotter(center, extent=2*radius, npoints=npoints)
    z = restr.eval(fp.points)

    if debug:
        fp.show_scalar_in_matplotlib(
                np.log10(np.abs(z + 1e-15)))
        import matplotlib.pyplot as pt
        pt.colorbar()
        pt.show()

    return np.max(np.abs(z))


# {{{ schematic visualization

def draw_box(el, eh, **kwargs):
    import matplotlib.pyplot as pt
    import matplotlib.patches as mpatches
    from matplotlib.path import Path

    pathdata = [
        (Path.MOVETO, (el[0], el[1])),
        (Path.LINETO, (eh[0], el[1])),
        (Path.LINETO, (eh[0], eh[1])),
        (Path.LINETO, (el[0], eh[1])),
        (Path.CLOSEPOLY, (el[0], el[1])),
        ]

    codes, verts = zip(*pathdata)
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, **kwargs)
    pt.gca().add_patch(patch)


def draw_circle(center, radius, **kwargs):
    center = np.asarray(center)

    import matplotlib.pyplot as plt
    plt.gca().add_patch(plt.Circle((center[0], center[1]), radius, **kwargs))


def draw_point(loc, **kwargs):
    import matplotlib.pyplot as plt
    plt.plot(*loc, marker="o", **kwargs)


def draw_arrow(from_pt, to_pt, shorten=0, **kwargs):
    import matplotlib.pyplot as plt
    dist = to_pt - from_pt

    from_pt = from_pt + shorten*dist
    dist = dist - 2*shorten*dist
    plt.arrow(from_pt[0], from_pt[1], dist[0], dist[1], **kwargs)


def draw_annotation(from_pt, to_pt, label, **kwargs):
    import matplotlib.pyplot as plt
    color = kwargs.setdefault("color", "white")

    arrowprops = dict(
            facecolor=color,
            edgecolor=color,
            shrink=0.05,
            width=1,
            headwidth=5)

    plt.gca().annotate(label, xy=from_pt, xytext=to_pt,
            arrowprops=arrowprops, **kwargs)


class SchematicVisitor(object):
    def __init__(self, default_expn_style="circle"):
        self.default_expn_style = default_expn_style

    def rec(self, psource):
        getattr(self, "visit_"+type(psource).__name__.lower())(psource)

    def visit_pointsources(self, psource):
        import matplotlib.pyplot as plt
        plt.plot(psource.points[0], psource.points[1], "o", label="source")

    def visit_sum(self, psource):
        for ps in psource.psources:
            self.rec(ps)

    visit_product = visit_sum

    def visit_multipoleexpansion(self, psource):
        expn_style = self.default_expn_style
        if psource.expn_style is not None:
            expn_style = psource.expn_style

        if psource.radius is not None:
            if expn_style == "box":
                r2 = psource.radius / np.sqrt(2)
                draw_box(psource.center - r2, psource.center + r2, fill=None)
            elif expn_style == "circle":
                draw_circle(psource.center, psource.radius, fill=None)
            else:
                raise ValueError("unknown expn_style: %s" % self.expn_style)

        import matplotlib.pyplot as plt
        plt.gca().text(psource.center[0], psource.center[1],
                type(psource).__name__[0],
                verticalalignment='center', horizontalalignment='center')

        if psource.derived_from is not None:
            xmin, xmax = plt.xlim()
            plt_width = xmax - xmin
            draw_arrow(psource.derived_from.center, psource.center, shorten=0.1,
                       facecolor="black", length_includes_head=True,
                       width=0.0005 * plt_width)
            self.rec(psource.derived_from)

    visit_localexpansion = visit_multipoleexpansion


def draw_schematic(psource, **kwargs):
    SchematicVisitor(**kwargs).rec(psource)
    import matplotlib.pyplot as plt
    plt.gca().set_aspect("equal")
    plt.tight_layout()

# }}}

# vim: foldmethod=marker
