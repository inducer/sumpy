from __future__ import annotations


__copyright__ = """
Copyright (C) 2017 Andreas Kloeckner
Copyright (C) 2017 Matt Wala
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

from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Optional, Sequence, Union

from pytools import memoize_method

from sumpy.kernel import TargetTransformationRemover


if TYPE_CHECKING:
    import pyopencl

    from sumpy.kernel import Kernel
    from sumpy.visualization import FieldPlotter

import logging

import numpy as np

import loopy as lp  # noqa: F401
import pyopencl as cl
import pyopencl.array


logger = logging.getLogger(__name__)

__doc__ = """

This module provides a convenient interface for numerical experiments with
local and multipole expansions.

.. autoclass:: ToyContext
.. autoclass:: PotentialSource
.. autoclass:: ConstantPotential
.. autoclass:: PointSources

These functions manipulate these potentials:

.. autofunction:: multipole_expand
.. autofunction:: local_expand
.. autofunction:: logplot
.. autofunction:: combine_inner_outer
.. autofunction:: combine_halfspace
.. autofunction:: combine_halfspace_and_outer
.. autofunction:: l_inf

These functions help with plotting:

.. autofunction:: draw_box
.. autofunction:: draw_circle
.. autofunction:: draw_annotation
.. autofunction:: draw_schematic

These are created behind the scenes and are not typically directly instantiated
by users:

.. autoclass:: OneOnBallPotential
.. autoclass:: HalfspaceOnePotential
.. autoclass:: ExpansionPotentialSource
.. autoclass:: MultipoleExpansion
.. autoclass:: LocalExpansion
.. autoclass:: PotentialExpressionNode
.. autoclass:: Sum
.. autoclass:: Product
.. autoclass:: SchematicVisitor

"""


# {{{ context

class ToyContext:
    """This class functions as a container for generated code and 'behind-the-scenes'
    information.

    .. automethod:: __init__
    """

    def __init__(self, cl_context: pyopencl.Context, kernel: Kernel,
            mpole_expn_class=None,
            local_expn_class=None,
            expansion_factory=None,
            extra_source_kwargs=None,
            extra_kernel_kwargs=None, m2l_use_fft=None):
        self.cl_context = cl_context
        self.queue = cl.CommandQueue(self.cl_context)
        self.kernel = kernel

        self.no_target_deriv_kernel = TargetTransformationRemover()(kernel)

        if expansion_factory is None:
            from sumpy.expansion import DefaultExpansionFactory
            expansion_factory = DefaultExpansionFactory()
        if mpole_expn_class is None:
            mpole_expn_class = \
                    expansion_factory.get_multipole_expansion_class(kernel)
        if local_expn_class is None:
            from sumpy.expansion.m2l import (
                FFTM2LTranslationClassFactory,
                NonFFTM2LTranslationClassFactory,
            )
            if m2l_use_fft:
                m2l_translation_class_factory = FFTM2LTranslationClassFactory()
            else:
                m2l_translation_class_factory = NonFFTM2LTranslationClassFactory()
            local_expn_class = \
                    expansion_factory.get_local_expansion_class(kernel)
            m2l_translation_class = \
                    m2l_translation_class_factory.get_m2l_translation_class(
                        kernel, local_expn_class)
            local_expn_class = partial(local_expn_class,
                    m2l_translation=m2l_translation_class())
        elif m2l_use_fft is not None:
            raise ValueError("local_expn_class and m2l_use_fft are both supplied. "
                             "Use only one of these arguments")

        self.mpole_expn_class = mpole_expn_class
        self.local_expn_class = local_expn_class

        if extra_source_kwargs is None:
            extra_source_kwargs = {}
        if extra_kernel_kwargs is None:
            extra_kernel_kwargs = {}

        self.extra_source_kwargs = extra_source_kwargs
        self.extra_kernel_kwargs = extra_kernel_kwargs

        extra_source_and_kernel_kwargs = extra_source_kwargs.copy()
        extra_source_and_kernel_kwargs.update(extra_kernel_kwargs)
        self.extra_source_and_kernel_kwargs = extra_source_and_kernel_kwargs

    @memoize_method
    def get_p2p(self):
        from sumpy.p2p import P2P
        return P2P(self.cl_context, (self.kernel,), exclude_self=False)

    @memoize_method
    def get_p2m(self, order):
        from sumpy import P2EFromSingleBox
        return P2EFromSingleBox(self.cl_context,
                self.mpole_expn_class(self.no_target_deriv_kernel, order),
                kernels=(self.kernel,))

    @memoize_method
    def get_p2l(self, order):
        from sumpy import P2EFromSingleBox
        return P2EFromSingleBox(self.cl_context,
                self.local_expn_class(self.no_target_deriv_kernel, order),
                kernels=(self.kernel,))

    @memoize_method
    def get_m2p(self, order):
        from sumpy import E2PFromSingleBox
        return E2PFromSingleBox(self.cl_context,
                self.mpole_expn_class(self.no_target_deriv_kernel, order),
                (self.kernel,))

    @memoize_method
    def get_l2p(self, order):
        from sumpy import E2PFromSingleBox
        return E2PFromSingleBox(self.cl_context,
                self.local_expn_class(self.no_target_deriv_kernel, order),
                (self.kernel,))

    @memoize_method
    def get_m2m(self, from_order, to_order):
        from sumpy import E2EFromCSR
        return E2EFromCSR(self.cl_context,
                self.mpole_expn_class(self.no_target_deriv_kernel, from_order),
                self.mpole_expn_class(self.no_target_deriv_kernel, to_order))

    @memoize_method
    def use_translation_classes_dependent_data(self):
        l_expn = self.local_expn_class(self.no_target_deriv_kernel, 2)
        return l_expn.m2l_translation.use_preprocessing

    @memoize_method
    def use_fft(self):
        l_expn = self.local_expn_class(self.no_target_deriv_kernel, 2)
        return l_expn.m2l_translation.use_fft

    @memoize_method
    def get_m2l(self, from_order, to_order):
        from sumpy import E2EFromCSR, M2LUsingTranslationClassesDependentData
        if self.use_translation_classes_dependent_data():
            m2l_class = M2LUsingTranslationClassesDependentData
        else:
            m2l_class = E2EFromCSR
        return m2l_class(self.cl_context,
                self.mpole_expn_class(self.no_target_deriv_kernel, from_order),
                self.local_expn_class(self.no_target_deriv_kernel, to_order))

    @memoize_method
    def get_m2l_translation_class_dependent_data_kernel(self, from_order, to_order):
        from sumpy import M2LGenerateTranslationClassesDependentData
        return M2LGenerateTranslationClassesDependentData(self.cl_context,
                self.mpole_expn_class(self.no_target_deriv_kernel, from_order),
                self.local_expn_class(self.no_target_deriv_kernel, to_order))

    @memoize_method
    def get_m2l_expansion_size(self, from_order, to_order):
        m_expn = self.mpole_expn_class(self.no_target_deriv_kernel, from_order)
        l_expn = self.local_expn_class(self.no_target_deriv_kernel, to_order)
        return l_expn.m2l_translation.preprocess_multipole_nexprs(l_expn, m_expn)

    @memoize_method
    def get_m2l_preprocess_mpole_kernel(self, from_order, to_order):
        from sumpy import M2LPreprocessMultipole
        return M2LPreprocessMultipole(self.cl_context,
                self.mpole_expn_class(self.no_target_deriv_kernel, from_order),
                self.local_expn_class(self.no_target_deriv_kernel, to_order))

    @memoize_method
    def get_m2l_postprocess_local_kernel(self, from_order, to_order):
        from sumpy import M2LPostprocessLocal
        return M2LPostprocessLocal(self.cl_context,
                self.mpole_expn_class(self.no_target_deriv_kernel, from_order),
                self.local_expn_class(self.no_target_deriv_kernel, to_order))

    @memoize_method
    def get_l2l(self, from_order, to_order):
        from sumpy import E2EFromCSR
        return E2EFromCSR(self.cl_context,
                self.local_expn_class(self.no_target_deriv_kernel, from_order),
                self.local_expn_class(self.no_target_deriv_kernel, to_order))

# }}}


# {{{ helpers

def _p2e(psource, center, rscale, order, p2e, expn_class, expn_kwargs):
    toy_ctx = psource.toy_ctx
    queue = toy_ctx.queue

    source_boxes = cl.array.to_device(
        queue, np.array([0], dtype=np.int32))
    box_source_starts = cl.array.to_device(
        queue, np.array([0], dtype=np.int32))
    box_source_counts_nonchild = cl.array.to_device(
        queue, np.array([psource.points.shape[-1]], dtype=np.int32))

    center = np.asarray(center)
    centers = cl.array.to_device(
        queue,
        np.array(center, dtype=np.float64).reshape(toy_ctx.kernel.dim, 1))

    evt, (coeffs,) = p2e(toy_ctx.queue,
            source_boxes=source_boxes,
            box_source_starts=box_source_starts,
            box_source_counts_nonchild=box_source_counts_nonchild,
            centers=centers,
            sources=cl.array.to_device(queue, psource.points),
            strengths=(cl.array.to_device(queue, psource.weights),),
            rscale=rscale,
            nboxes=1,
            tgt_base_ibox=0,
            **toy_ctx.extra_source_and_kernel_kwargs)

    return expn_class(toy_ctx, center, rscale, order, coeffs[0].get(queue),
            derived_from=psource, **expn_kwargs)


def _e2p(psource, targets, e2p):
    toy_ctx = psource.toy_ctx
    queue = toy_ctx.queue

    ntargets = targets.shape[-1]
    boxes = cl.array.to_device(
        queue, np.array([0], dtype=np.int32))
    box_target_starts = cl.array.to_device(
        queue, np.array([0], dtype=np.int32))
    box_target_counts_nonchild = cl.array.to_device(
        queue, np.array([ntargets], dtype=np.int32))

    centers = cl.array.to_device(
        queue,
        np.array(psource.center, dtype=np.float64).reshape(toy_ctx.kernel.dim, 1))

    from pytools.obj_array import make_obj_array

    from sumpy.tools import vector_to_device

    coeffs = cl.array.to_device(queue, np.array([psource.coeffs]))
    evt, (pot,) = e2p(
            toy_ctx.queue,
            src_expansions=coeffs,
            src_base_ibox=0,
            target_boxes=boxes,
            box_target_starts=box_target_starts,
            box_target_counts_nonchild=box_target_counts_nonchild,
            centers=centers,
            rscale=psource.rscale,
            targets=vector_to_device(queue, make_obj_array(targets)),
            **toy_ctx.extra_kernel_kwargs)

    return pot.get(queue)


def _e2e(psource, to_center, to_rscale, to_order, e2e, expn_class, expn_kwargs,
         extra_kernel_kwargs):
    toy_ctx = psource.toy_ctx
    queue = toy_ctx.queue

    target_boxes = cl.array.to_device(
        queue, np.array([1], dtype=np.int32))
    src_box_starts = cl.array.to_device(
        queue, np.array([0, 1], dtype=np.int32))
    src_box_lists = cl.array.to_device(
        queue, np.array([0], dtype=np.int32))

    centers = cl.array.to_device(
        queue,
        np.array(
            [
                # box 0: source
                psource.center,

                # box 1: target
                to_center,
            ],
            dtype=np.float64).T.copy()
        )

    coeffs = cl.array.to_device(queue, np.array([psource.coeffs]))
    args = {
        "queue": toy_ctx.queue,
        "src_expansions": coeffs,
        "src_base_ibox": 0,
        "tgt_base_ibox": 0,
        "ntgt_level_boxes": 2,
        "target_boxes": target_boxes,
        "src_box_starts": src_box_starts,
        "src_box_lists": src_box_lists,
        "centers": centers,
        "src_rscale": psource.rscale,
        "tgt_rscale": to_rscale,
        **extra_kernel_kwargs,
        **toy_ctx.extra_kernel_kwargs,
    }

    evt, (to_coeffs,) = e2e(**args)

    return expn_class(
            toy_ctx, to_center, to_rscale, to_order, to_coeffs[1].get(queue),
            derived_from=psource, **expn_kwargs)


def _m2l(psource, to_center, to_rscale, to_order, e2e, expn_class, expn_kwargs,
         translation_classes_kwargs):
    toy_ctx = psource.toy_ctx
    queue = toy_ctx.queue

    coeffs = cl.array.to_device(queue, np.array([psource.coeffs]))

    m2l_use_translation_classes_dependent_data = \
            toy_ctx.use_translation_classes_dependent_data()

    if m2l_use_translation_classes_dependent_data:
        data_kernel = translation_classes_kwargs["data_kernel"]
        preprocess_kernel = translation_classes_kwargs["preprocess_kernel"]
        postprocess_kernel = translation_classes_kwargs["postprocess_kernel"]
        expn_size = translation_classes_kwargs["m2l_expn_size"]

        # Preprocess the mpole expansion
        preprocessed_src_expansions = cl.array.zeros(queue, (1, expn_size),
                dtype=np.complex128)
        evt, _ = preprocess_kernel(queue,
                src_expansions=coeffs,
                preprocessed_src_expansions=preprocessed_src_expansions,
                src_rscale=np.float64(psource.rscale),
                **toy_ctx.extra_kernel_kwargs)

        from sumpy.tools import get_native_event, get_opencl_fft_app, run_opencl_fft

        if toy_ctx.use_fft:
            fft_app = get_opencl_fft_app(queue, (expn_size,),
                dtype=preprocessed_src_expansions.dtype, inverse=False)
            ifft_app = get_opencl_fft_app(queue, (expn_size,),
                dtype=preprocessed_src_expansions.dtype, inverse=True)

            evt, preprocessed_src_expansions = run_opencl_fft(fft_app, queue,
                    preprocessed_src_expansions, inverse=False, wait_for=[evt])

        # Compute translation classes data
        m2l_translation_classes_lists = cl.array.to_device(queue,
                np.array([0], dtype=np.int32))
        dist = np.array(to_center - psource.center, dtype=np.float64)
        dim = toy_ctx.kernel.dim
        m2l_translation_vectors = cl.array.to_device(queue, dist.reshape(dim, 1))
        m2l_translation_classes_dependent_data = cl.array.zeros(queue,
                (1, expn_size), dtype=np.complex128)

        evt, _ = data_kernel(
                queue,
                src_rscale=np.float64(psource.rscale),
                ntranslation_classes=1,
                translation_classes_level_start=0,
                m2l_translation_vectors=m2l_translation_vectors,
                m2l_translation_classes_dependent_data=(
                    m2l_translation_classes_dependent_data),
                ntranslation_vectors=1,
                wait_for=[get_native_event(evt)],
                **toy_ctx.extra_kernel_kwargs)

        if toy_ctx.use_fft:
            evt, m2l_translation_classes_dependent_data = run_opencl_fft(fft_app,
                queue, m2l_translation_classes_dependent_data, inverse=False,
                wait_for=[evt])

        ret = _e2e(psource, to_center, to_rscale, to_order,
            e2e, expn_class, expn_kwargs,
            {
                "src_expansions": preprocessed_src_expansions,
                "m2l_translation_classes_lists": m2l_translation_classes_lists,
                "m2l_translation_classes_dependent_data": (
                    m2l_translation_classes_dependent_data),
                "translation_classes_level_start": 0,
            }
        )

        # Postprocess the local expansion
        local_before = cl.array.to_device(queue, np.array([ret.coeffs]))
        to_coeffs = cl.array.zeros(queue, (1, len(data_kernel.tgt_expansion)),
                                           dtype=coeffs.dtype)

        if toy_ctx.use_fft:
            evt, local_before = run_opencl_fft(ifft_app, queue,
                local_before, inverse=True, wait_for=[get_native_event(evt)])

        evt, _ = postprocess_kernel(queue=queue,
                tgt_expansions_before_postprocessing=local_before,
                tgt_expansions=to_coeffs,
                src_rscale=np.float64(psource.rscale),
                tgt_rscale=np.float64(to_rscale),
                wait_for=[get_native_event(evt)],
                **toy_ctx.extra_kernel_kwargs)

        return expn_class(
            toy_ctx, to_center, to_rscale, to_order, to_coeffs.get(queue)[0],
            derived_from=psource, **expn_kwargs)
    else:
        ret = _e2e(psource, to_center, to_rscale, to_order, e2e, expn_class,
                expn_kwargs, {})

    return ret


# }}}

# {{{ potential source classes

class PotentialSource:
    """A base class for all classes representing potentials that can be
    evaluated anywhere in space.

    .. automethod:: eval

    Supports (lazy) arithmetic:

    .. automethod:: __neg__
    .. automethod:: __add__
    .. automethod:: __radd__
    .. automethod:: __sub__
    .. automethod:: __rsub__
    .. automethod:: __mul__
    .. automethod:: __rmul__
    """

    def __init__(self, toy_ctx: ToyContext):
        self.toy_ctx = toy_ctx

    def eval(self, targets: np.ndarray) -> np.ndarray:
        """
        :param targets: An array of shape ``(dim, ntargets)``.
        :returns: an array of shape ``(ntargets,)``.
        """
        raise NotImplementedError()

    def __neg__(self) -> PotentialSource:
        return -1*self

    def __add__(self, other: Union[Number, np.number, PotentialSource]
                ) -> PotentialSource:
        if isinstance(other, (Number, np.number)):
            other = ConstantPotential(self.toy_ctx, other)
        elif not isinstance(other, PotentialSource):
            return NotImplemented

        return Sum((self, other))

    __radd__ = __add__

    def __sub__(self,
                other: Union[Number, np.number, PotentialSource]) -> PotentialSource:
        return self.__add__(-other)

    def __rsub__(self,
                 other: Union[Number, np.number, PotentialSource]
                 ) -> PotentialSource:
        return (-self).__add__(other)

    def __mul__(self,
                other: Union[Number, np.number, PotentialSource]) -> PotentialSource:
        if isinstance(other, (Number, np.number)):
            other = ConstantPotential(self.toy_ctx, other)
        elif not isinstance(other, PotentialSource):
            return NotImplemented

        return Product((self, other))

    __rmul__ = __mul__


class ConstantPotential(PotentialSource):
    """
    Inherits from :class:`PotentialSource`.

    .. automethod:: __init__
    """

    def __init__(self, toy_ctx: ToyContext, value):
        super().__init__(toy_ctx)
        self.value = np.array(value)

    def eval(self, targets: np.ndarray) -> np.ndarray:
        pot = np.empty(targets.shape[-1], dtype=self.value.dtype)
        pot.fill(self.value)
        return pot


class OneOnBallPotential(PotentialSource):
    """
    A potential that is the characteristic function on a ball.

    Inherits from :class:`PotentialSource`.

    .. automethod:: __init__
    """
    def __init__(self,
                 toy_ctx: ToyContext, center: np.ndarray, radius: float) -> None:
        super().__init__(toy_ctx)
        self.center = np.asarray(center)
        self.radius = radius

    def eval(self, targets: np.ndarray) -> np.ndarray:
        dist_vec = targets - self.center[:, np.newaxis]
        return (np.sum(dist_vec**2, axis=0) < self.radius**2).astype(np.float64)


class HalfspaceOnePotential(PotentialSource):
    """
    A potential that is the characteristic function of a halfspace.

    .. automethod:: __init__
    """
    def __init__(self, toy_ctx: ToyContext, center: np.ndarray,
                 axis: int, side: int = 1) -> None:
        super().__init__(toy_ctx)
        self.center = np.asarray(center)
        self.axis = axis
        self.side = side

    def eval(self, targets: np.ndarray) -> np.ndarray:
        return (
            (self.side*(targets[self.axis] - self.center[self.axis])) >= 0
            ).astype(np.float64)


class PointSources(PotentialSource):
    """
    Inherits from :class:`PotentialSource`.

    .. attribute:: points

        ``[ndim, npoints]``

    .. automethod:: __init__
    """

    def __init__(self,
                 toy_ctx: ToyContext, points: np.ndarray, weights: np.ndarray,
                 center: Optional[np.ndarray] = None):
        super().__init__(toy_ctx)

        self.points = points
        self.weights = weights
        self._center = center

    def eval(self, targets: np.ndarray) -> np.ndarray:
        queue = self.toy_ctx.queue
        evt, (potential,) = self.toy_ctx.get_p2p()(
                queue,
                cl.array.to_device(queue, targets),
                cl.array.to_device(queue, self.points),
                [cl.array.to_device(queue, self.weights)],
                **self.toy_ctx.extra_source_and_kernel_kwargs)

        return potential.get(queue)

    @property
    def center(self):
        if self._center is not None:
            return self._center

        return np.average(self.points, axis=1)


class ExpansionPotentialSource(PotentialSource):
    """
    Inherits from :class:`PotentialSource`.

    .. attribute:: radius

        Not used mathematically. Just for visualization, purely advisory.

    .. attribute:: text_kwargs

       Passed to :func:`matplotlib.pyplot.annotate`. Used for customizing the
       expansion label. Changing the label text is supported by passing the
       kwarg *s*.  Just for visualization, purely advisory.

    .. automethod:: __init__
    """
    def __init__(self, toy_ctx, center, rscale, order, coeffs, derived_from,
            radius=None, expn_style=None, text_kwargs=None):
        super().__init__(toy_ctx)
        self.center = np.asarray(center)
        self.rscale = rscale
        self.order = order
        self.coeffs = coeffs

        self.derived_from = derived_from
        self.radius = radius
        self.expn_style = expn_style
        self.text_kwargs = text_kwargs

    def with_coeffs(self, coeffs):
        return type(self)(self.toy_ctx, self.center, self.rscale, self.order,
                coeffs, self.derived_from, radius=self.radius,
                expn_style=self.expn_style, text_kwargs=self.text_kwargs)


class MultipoleExpansion(ExpansionPotentialSource):
    """
    Inherits from :class:`ExpansionPotentialSource`.
    """

    def eval(self, targets: np.ndarray) -> np.ndarray:
        return _e2p(self, targets, self.toy_ctx.get_m2p(self.order))


class LocalExpansion(ExpansionPotentialSource):
    """
    Inherits from :class:`ExpansionPotentialSource`.
    """

    def eval(self, targets: np.ndarray) -> np.ndarray:
        return _e2p(self, targets, self.toy_ctx.get_l2p(self.order))


class PotentialExpressionNode(PotentialSource):
    """
    Inherits from :class:`PotentialSource`.

    .. automethod:: __init__
    """

    def __init__(self, psources: Sequence[PotentialSource]) -> None:
        from pytools import single_valued
        super().__init__(
                single_valued(psource.toy_ctx for psource in psources))

        self.psources = psources

    @property
    def center(self) -> np.ndarray:
        for psource in self.psources:
            try:
                return psource.center
            except AttributeError:
                pass

        raise ValueError("no psource with a center found")


class Sum(PotentialExpressionNode):
    """
    Inherits from :class:`PotentialExpressionNode`.
    """

    def eval(self, targets: np.ndarray) -> np.ndarray:
        result = 0
        for psource in self.psources:
            result = result + psource.eval(targets)

        return result


class Product(PotentialExpressionNode):
    """
    Inherits from :class:`PotentialExpressionNode`.
    """

    def eval(self, targets: np.ndarray) -> np.ndarray:
        result = 1
        for psource in self.psources:
            result = result * psource.eval(targets)

        return result

# }}}


def multipole_expand(psource: PotentialSource,
                     center: np.ndarray, order: Optional[int] = None,
                     rscale: float = 1, **expn_kwargs) -> MultipoleExpansion:
    if isinstance(psource, PointSources):
        if order is None:
            raise ValueError("order may not be None")

        return _p2e(psource, center, rscale, order, psource.toy_ctx.get_p2m(order),
                MultipoleExpansion, expn_kwargs)

    elif isinstance(psource, MultipoleExpansion):
        if order is None:
            order = psource.order

        return _e2e(psource, center, rscale, order,
                psource.toy_ctx.get_m2m(psource.order, order),
                MultipoleExpansion, expn_kwargs, {})

    else:
        raise TypeError(f"do not know how to expand '{type(psource).__name__}'")


def local_expand(
        psource: PotentialSource,
        center: np.ndarray, order: Optional[int] = None, rscale: Number = 1,
        **expn_kwargs) -> LocalExpansion:
    if isinstance(psource, PointSources):
        if order is None:
            raise ValueError("order may not be None")

        return _p2e(psource, center, rscale, order, psource.toy_ctx.get_p2l(order),
                LocalExpansion, expn_kwargs)

    elif isinstance(psource, MultipoleExpansion):
        if order is None:
            order = psource.order

        toy_ctx = psource.toy_ctx
        translation_classes_kwargs = {}
        m2l_use_translation_classes_dependent_data = \
            toy_ctx.use_translation_classes_dependent_data()

        if m2l_use_translation_classes_dependent_data:
            data_kernel = toy_ctx.get_m2l_translation_class_dependent_data_kernel(
                    psource.order, order)
            preprocess_kernel = toy_ctx.get_m2l_preprocess_mpole_kernel(
                    psource.order, order)
            postprocess_kernel = toy_ctx.get_m2l_postprocess_local_kernel(
                    psource.order, order)
            translation_classes_kwargs["data_kernel"] = data_kernel
            translation_classes_kwargs["preprocess_kernel"] = preprocess_kernel
            translation_classes_kwargs["postprocess_kernel"] = postprocess_kernel
            translation_classes_kwargs["m2l_expn_size"] = \
                    toy_ctx.get_m2l_expansion_size(psource.order, order)

        return _m2l(psource, center, rscale, order,
                toy_ctx.get_m2l(psource.order, order),
                LocalExpansion, expn_kwargs,
                translation_classes_kwargs)

    elif isinstance(psource, LocalExpansion):
        if order is None:
            order = psource.order

        return _e2e(psource, center, rscale, order,
                psource.toy_ctx.get_l2l(psource.order, order),
                LocalExpansion, expn_kwargs, {})

    else:
        raise TypeError(f"do not know how to expand '{type(psource).__name__}'")


def logplot(fp: FieldPlotter, psource: PotentialSource, **kwargs) -> None:
    fp.show_scalar_in_matplotlib(
            np.log10(np.abs(psource.eval(fp.points) + 1e-15)), **kwargs)


def combine_inner_outer(
        psource_inner: PotentialSource,
        psource_outer: PotentialSource,
        radius: Optional[float],
        center: Optional[np.ndarray] = None) -> PotentialSource:
    if center is None:
        center = psource_inner.center
    if radius is None:
        radius = psource_inner.radius

    ball_one = OneOnBallPotential(psource_inner.toy_ctx, center, radius)
    return (
            psource_inner * ball_one
            + psource_outer * (1 - ball_one))


def combine_halfspace(psource_pos: PotentialSource,
                      psource_neg: PotentialSource, axis: int,
                      center: Optional[np.ndarray] = None) -> PotentialSource:
    if center is None:
        center = psource_pos.center

    halfspace_one = HalfspaceOnePotential(psource_pos.toy_ctx, center, axis)
    return (
        psource_pos * halfspace_one
        + psource_neg * (1-halfspace_one))


def combine_halfspace_and_outer(
        psource_pos: PotentialSource,
        psource_neg: PotentialSource,
        psource_outer: PotentialSource,
        axis: int, radius: Optional[Number] = None,
        center: Optional[np.ndarray] = None) -> PotentialSource:

    if center is None:
        center = psource_pos.center
    if radius is None:
        center = psource_pos.radius

    return combine_inner_outer(
            combine_halfspace(psource_pos, psource_neg, axis, center),
            psource_outer, radius, center)


def l_inf(psource: PotentialSource, radius: float,
          center: Optional[np.ndarray] = None, npoints: int = 100,
          debug: bool = False) -> np.number:
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
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as pt
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


def draw_annotation(to_pt, from_pt, label, arrowprops=None, **kwargs):
    """
    :arg to_pt: Head of arrow
    :arg from_pt: Tail of arrow
    :arg label: Annotation label
    :arg arrowprops: Passed to arrowprops
    :arg kwargs: Passed to annotate
    """
    if arrowprops is None:
        arrowprops = {}

    import matplotlib.pyplot as plt

    my_arrowprops = {
            "facecolor": "black",
            "edgecolor": "black",
            "arrowstyle": "->"}

    my_arrowprops.update(arrowprops)

    plt.gca().annotate(label, xy=to_pt, xytext=from_pt,
            arrowprops=my_arrowprops, **kwargs)


class SchematicVisitor:
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
                raise ValueError(f"unknown expn_style: {expn_style}")

        if psource.derived_from is None:
            return

        # Draw an annotation of the form
        #
        # ------> M

        text_kwargs = {
                "verticalalignment": "center",
                "horizontalalignment": "center"}

        label = "${}_{{{}}}$".format(
                type(psource).__name__[0].lower().replace("l", "\\ell"),
                psource.order)

        if psource.text_kwargs is not None:
            psource_text_kwargs_copy = psource.text_kwargs.copy()
            label = psource_text_kwargs_copy.pop("s", label)
            text_kwargs.update(psource_text_kwargs_copy)

        shrinkB = 0  # noqa
        if isinstance(psource.derived_from, ExpansionPotentialSource):
            # Avoid overlapping the tail of the arrow with any expansion labels that
            # are present at the tail.
            import matplotlib as mpl
            font_size = mpl.rcParams["font.size"]
            shrinkB = 7/8 * font_size  # noqa

        arrowprops = {"shrinkB": shrinkB, "arrowstyle": "<|-"}

        draw_annotation(psource.derived_from.center, psource.center, label,
                        arrowprops, **text_kwargs)
        self.rec(psource.derived_from)

    visit_localexpansion = visit_multipoleexpansion


def draw_schematic(psource, **kwargs):
    SchematicVisitor(**kwargs).rec(psource)
    import matplotlib.pyplot as plt
    plt.gca().set_aspect("equal")
    plt.tight_layout()

# }}}

# vim: foldmethod=marker
