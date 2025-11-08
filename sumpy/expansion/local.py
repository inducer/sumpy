from __future__ import annotations


__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from typing_extensions import override

from pytools import single_valued

import sumpy.symbolic as sym
from sumpy.expansion import (
    ExpansionBase,
    LinearPDEConformingVolumeTaylorExpansion,
    VolumeTaylorExpansion,
    VolumeTaylorExpansionMixin,
)
from sumpy.expansion.multipole import H2DMultipoleExpansion, Y2DMultipoleExpansion
from sumpy.tools import add_to_sac, mi_increment_axis


if TYPE_CHECKING:
    from collections.abc import Sequence

    import loopy as lp

    from sumpy.assignment_collection import SymbolicAssignmentCollection
    from sumpy.expansion.diff_op import MultiIndex
    from sumpy.expansion.m2l import M2LTranslationBase, TranslationClassesDepData
    from sumpy.expansion.multipole import (
        HankelBased2DMultipoleExpansion,
        MultipoleExpansionBase,
    )
    from sumpy.kernel import Kernel


logger = logging.getLogger(__name__)

__doc__ = """
.. autoclass:: LocalExpansionBase
.. autoclass:: VolumeTaylorLocalExpansion
.. autoclass:: H2DLocalExpansion
.. autoclass:: Y2DLocalExpansion
.. autoclass:: LineTaylorLocalExpansion
"""


@dataclass(frozen=True)
class LocalExpansionBase(ExpansionBase, ABC):
    """Base class for local expansions.

    .. automethod:: translate_from
    """

    @property
    @abstractmethod
    def m2l_translation(self) -> M2LTranslationBase:
        ...

    @abstractmethod
    def translate_from(self,
                src_expansion: LocalExpansionBase | MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                m2l_translation_classes_dependent_data: (
                       TranslationClassesDepData | None) = None
            ) -> Sequence[sym.Expr]:
        """Translate from a multipole or local expansion to a local expansion

        :arg src_expansion: The source expansion to translate from.
        :arg src_coeff_exprs: An iterable of symbolic expressions representing the
                coefficients of the source expansion.
        :arg src_rscale: scaling factor for the source expansion.
        :arg dvec: symbolic expression for the distance between target and
                source centers.
        :arg tgt_rscale: scaling factor for the target expansion.
        :arg sac: An object of type
                :class:`sumpy.assignment_collection.SymbolicAssignmentCollection`
                to collect common subexpressions or None.
        :arg m2l_translation_classes_dependent_data: An iterable of symbolic
                expressions representing the expressions returned by
                :func:`~sumpy.expansion.m2l.M2LTranslationBase.translation_classes_dependent_data`.
        """


# {{{ line taylor

class LineTaylorLocalExpansion(LocalExpansionBase, ABC):
    @property
    @override
    def m2l_translation(self) -> M2LTranslationBase:
        # FIXME: Um...
        raise NotImplementedError()

    @override
    def get_storage_index(self, mi: MultiIndex) -> int:
        ind, = mi
        return ind

    @override
    def get_coefficient_identifiers(self) -> Sequence[MultiIndex]:
        return [(i,) for i in range(self.order+1)]

    @override
    def coefficients_from_source(self,
                kernel: Kernel,
                avec: sym.Matrix,
                bvec: sym.Matrix | None,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None
            ) -> Sequence[sym.Expr]:
        # no point in heeding rscale here--just ignore it
        if bvec is None:
            raise RuntimeError("cannot use line-Taylor expansions in a setting "
                    "where the center-target vector is not known at coefficient "
                    "formation")

        tau = sym.Symbol("tau")
        avec_line = cast("sym.Matrix", avec + tau*bvec)
        line_kernel = kernel.get_expression(avec_line)

        from sumpy.symbolic import USE_SYMENGINE

        if USE_SYMENGINE:
            from sumpy.derivative_taker import ExprDerivativeTaker
            deriv_taker = ExprDerivativeTaker(line_kernel, (tau,), sac=sac,
                                              rscale=sym.sympify(1))

            return [kernel.postprocess_at_source(deriv_taker.diff(i), avec)
                    .subs(tau, 0)
                    for i in self.get_coefficient_identifiers()]
        else:
            # Workaround for sympy. The automatic distribution after
            # single-variable diff makes the expressions very large
            # (https://github.com/sympy/sympy/issues/4596), so avoid doing
            # single variable diff.
            #
            # See also https://gitlab.tiker.net/inducer/pytential/merge_requests/12

            return [kernel.postprocess_at_source(line_kernel.diff(tau, i), avec)
                    .subs(tau, 0)
                    for i, in self.get_coefficient_identifiers()]

    @override
    def evaluate(self,
                 kernel: Kernel,
                 coeffs: Sequence[sym.Expr],
                 bvec: sym.Matrix,
                 rscale: sym.Expr,
                 sac: SymbolicAssignmentCollection | None = None) -> sym.Expr:
        # no point in heeding rscale here--just ignore it

        # NOTE: We can't meaningfully apply target derivatives here.
        # Instead, this is handled in LayerPotentialBase._evaluate.
        return sym.Add(*(
                    coeffs[self.get_storage_index(i)] / math.factorial(i[0])
                    for i in self.get_coefficient_identifiers()))

    @override
    def translate_from(self,
                src_expansion: LocalExpansionBase | MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                m2l_translation_classes_dependent_data: (
                       TranslationClassesDepData | None) = None
            ) -> Sequence[sym.Expr]:
        raise NotImplementedError

# }}}


# {{{ volume taylor

@dataclass(frozen=True)
class VolumeTaylorLocalExpansionBase(VolumeTaylorExpansionMixin,
                                     LocalExpansionBase, ABC):
    """
    Coefficients represent derivative values of the kernel.
    """

    m2l_translation_override: M2LTranslationBase | None = \
        field(kw_only=True, default=None)

    @property
    @override
    def m2l_translation(self) -> M2LTranslationBase:
        if self.m2l_translation_override is not None:
            return self.m2l_translation_override
        else:
            from sumpy.expansion.m2l import DefaultM2LTranslationClassFactory
            factory = DefaultM2LTranslationClassFactory()
            return factory.get_m2l_translation_class(self.kernel, self.__class__)()

    @override
    def coefficients_from_source_vec(self,
                kernels: Sequence[Kernel],
                avec: sym.Matrix,
                bvec: sym.Matrix | None,
                rscale: sym.Expr,
                weights: Sequence[sym.Expr],
                sac: SymbolicAssignmentCollection | None = None
            ) -> Sequence[sym.Expr]:
        """Form an expansion with a linear combination of kernels and weights.
        Since all of the kernels share a base kernel, this method uses one
        derivative taker with one SymbolicAssignmentCollection object
        to remove redundant calculations.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.
        :arg sac: a symbolic assignment collection where temporary
            expressions are stored.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """
        if not self.use_rscale:
            rscale = sym.sympify(1)

        base_kernel = single_valued(knl.get_base_kernel() for knl in kernels)
        base_taker = base_kernel.get_derivative_taker(avec, rscale, sac)
        result: list[sym.Expr] = [sym.sympify(0)]*len(self)

        for knl, weight in zip(kernels, weights, strict=True):
            taker = knl.postprocess_at_source(base_taker, avec)
            # Following is a hack to make sure cse works.
            if 1:
                def save_temp(x: sym.Expr) -> sym.Expr:
                    return add_to_sac(sac, weight * x)  # noqa: B023

                for i, mi in enumerate(self.get_coefficient_identifiers()):
                    result[i] += taker.diff(mi, save_temp)
            else:
                def save_temp(x: sym.Expr) -> sym.Expr:
                    return add_to_sac(sac, x)

                for i, mi in enumerate(self.get_coefficient_identifiers()):
                    result[i] += weight * taker.diff(mi, save_temp)

        return result

    @override
    def coefficients_from_source(self,
                kernel: Kernel,
                avec: sym.Matrix,
                bvec: sym.Matrix | None,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None
            ) -> Sequence[sym.Expr]:
        return self.coefficients_from_source_vec((kernel,), avec, bvec,
                rscale=rscale, weights=(sym.sympify(1),), sac=sac)

    @override
    def evaluate(self,
                kernel: Kernel,
                coeffs: Sequence[sym.Expr],
                bvec: sym.Matrix,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> sym.Expr:
        rscale = sym.sympify(1 if not self.use_rscale else rscale)

        evaluated_coeffs = (
            self.expansion_terms_wrangler.get_full_kernel_derivatives_from_stored(
                coeffs, rscale, sac=sac))

        bvec_scaled = [cast("sym.Expr", b*rscale**-1) for b in bvec]
        from sumpy.tools import mi_factorial, mi_power

        result = sym.sympify(sum(
            coeff
            * mi_power(bvec_scaled, mi, evaluate=False)
            / mi_factorial(mi)
            for coeff, mi in zip(
                    evaluated_coeffs, self.get_full_coefficient_identifiers(),
                    strict=True)))

        return kernel.postprocess_at_target(result, bvec)

    @override
    def translate_from(self,
                src_expansion: LocalExpansionBase | MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                m2l_translation_classes_dependent_data: (
                       TranslationClassesDepData | None) = None,
                _fast_version: bool = True,
            ) -> Sequence[sym.Expr]:
        logger.info("building translation operator for %s: %s(%d) -> %s(%d): start",
                    src_expansion.kernel,
                    type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order)

        if not self.use_rscale:
            src_rscale = sym.sympify(1)
            tgt_rscale = sym.sympify(1)

        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase

        # {{{ M2L

        if isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
            m2l_result = self.m2l_translation.translate(self, src_expansion,
                src_coeff_exprs, src_rscale, dvec, tgt_rscale, sac,
                m2l_translation_classes_dependent_data)

            logger.info("building translation operator: done")
            return m2l_result

        # }}}

        assert isinstance(src_expansion, VolumeTaylorLocalExpansionBase)

        # {{{ L2L

        # not coming from a Taylor multipole: expand via derivatives
        # FIXME: this shouldn't need to be sympified, but many places still
        # pass in floats. removing it fails `test_m2m_and_l2l_exprs_simpler`
        rscale_ratio = add_to_sac(sac, sym.sympify(tgt_rscale/src_rscale))

        src_wrangler = src_expansion.expansion_terms_wrangler
        src_coeffs = (
            src_wrangler.get_full_kernel_derivatives_from_stored(
                src_coeff_exprs, src_rscale, sac=sac))

        src_mis = \
            src_expansion.expansion_terms_wrangler.get_full_coefficient_identifiers()
        src_mi_to_index = {mi: i for i, mi in enumerate(src_mis)}

        tgt_mis = self.expansion_terms_wrangler.get_coefficient_identifiers()
        tgt_mi_to_index = {mi: i for i, mi in enumerate(tgt_mis)}

        tgt_split = self.expansion_terms_wrangler._split_coeffs_into_hyperplanes()

        p = max(sum(mi) for mi in src_mis)
        result: list[sym.Expr] = [sym.sympify(0)] * len(tgt_mis)

        # Local expansion around the old center gives us that,
        #
        # $ u = \sum_{|q| \le p} (x-c_1)^q \frac{C_q}{q!} $
        #
        # where $c_1$ is the old center and $q$ is a multi-index,
        # $p$ is the order and $C_q$ is a coefficient of the local expansion around
        # the center $c_1$.
        #
        # Differentiating, we get,
        #
        # $ D_{r} u = \sum_{|q| \le p} \frac{C_{q}}{(q-r)!} (x - c_1)^{q - r}$.
        #
        # This algorithm uses the observation that L2L coefficients
        # have the following form in 2D
        #
        # $T_{m, n} = \sum_{i\le p-m, j\le p-n-m-i} C_{i+m, j+n}
        #                   d_x^i d_y^j \frac{1}{i! j!}$
        #
        # where $d$ is the distance between the centers and $T$ is the translated
        # coefficient. $T$ can be rewritten as follows.
        #
        # Let $Y1_{m, n} = \sum_{j\le p-m-n} C_{m, j+n} d_y^j \frac{1}{j!}$.
        #
        # Then, $T_{m, n} = \sum_{i\le p-m} Y1_{i+m, n} d_x^i \frac{1}{i!}$.
        #
        # Expanding this to 3D,
        # $T_{m, n, l} = \sum_{i \le p-m, j \le p-n-m-i, k \le p-n-m-l-i-j}
        #             C_{i+m, j+n, k+l} d_x^i d_y^j d_z^k \frac{1}{i! j! k!}$
        #
        # Let,
        # $Y1_{m, n, l} = \sum_{k\le p-m-n-l} C_{m, n, k+l} d_z^k \frac{1}{l!}$
        # and,
        # $Y2_{m, n, l} = \sum_{j\le p-m-n} Y1_{m, j+n, l} d_y^j \frac{1}{n!}$.
        #
        # Then,
        # $T_{m, n, l} = \sum_{i\le p-m} Y2_{i+m, n, l} d_x^i \frac{1}{m!}$.
        #
        # Cost of the above algorithm is $O(p^4)$ for full since each value needs
        # $O(p)$ work and there are $O(p^3)$ values for $T, Y1, Y2$.
        # For a hyperplane of coefficients with normal direction `l` fixed,
        # we need only $O(p^2)$ of $T, Y1, Y2$ and since there are only a constant
        # number of coefficient hyperplanes in compressed, the algorithm is
        # $O(p^3)$

        # We start by iterating through all the axes which is at most 3 iterations
        # (in <=3D).
        # The number of iterations is one for full because all the $O(p)$ hyperplanes
        # are parallel to each other.
        # The number of iterations is one for compressed expansions with
        # elliptic PDEs because the $O(1)$ hyperplanes are parallel to each other.
        for axis in {d for d, _ in tgt_split}:
            # Use the axis as the first dimension to vary so that the below
            # algorithm is O(p^{d+1}) for full and O(p^{d}) for compressed
            dims = [axis, *list(range(axis)), *list(range(axis + 1, self.dim))]

            # Start with source coefficients. Gets updated after each axis.
            cur_dim_input_coeffs = list(src_coeffs)
            cur_dim_output_coeffs: list[sym.Expr] = [sym.sympify(-1)] * len(src_mis)

            # O(1) iterations
            for d in dims:
                cur_dim_output_coeffs = [sym.sympify(0)] * len(src_mis)

                # Only O(p^{d-1}) operations are used in compressed
                # O(p^d) operations are used in full
                for out_i, out_mi in enumerate(src_mis):
                    # O(p) iterations
                    for q in range(p+1-sum(out_mi)):
                        src_mi = mi_increment_axis(out_mi, d, q)
                        if src_mi in src_mi_to_index:
                            dvec_d = cast("sym.Expr", dvec[d])

                            cur_dim_output_coeffs[out_i] += (
                                (dvec_d/src_rscale)**q
                                * cur_dim_input_coeffs[src_mi_to_index[src_mi]]
                                / math.factorial(q))

                # Y at the end of the iteration becomes the source coefficients
                # for the next iteration
                cur_dim_input_coeffs = cur_dim_output_coeffs

            for mi in tgt_mis:
                # In L2L, source level usually has same or higher order than target
                # level. If not, extra coeffs in target level are zero filled.
                if mi not in src_mi_to_index:
                    result[tgt_mi_to_index[mi]] = sym.sympify(0)
                else:
                    # Add to result after scaling
                    result[tgt_mi_to_index[mi]] += (
                        cur_dim_output_coeffs[src_mi_to_index[mi]]
                        * rscale_ratio ** sum(mi))

        # {{{ simpler, functionally equivalent code

        if not _fast_version:
            # Rscale/operand magnitude is fairly sensitive to the order of
            # operations--which is something we don't have fantastic control
            # over at the symbolic level. Scaling dvec, then differentiating,
            # and finally rescaling dvec leaves the expression needing a scaling
            # to compensate for differentiating which is done at the end.
            # This moves the two cancelling "rscales" closer to each other at
            # the end in the hope of helping rscale magnitude.
            from sumpy.derivative_taker import ExprDerivativeTaker

            dvec_scaled = sym.Matrix([d*src_rscale for d in dvec])
            expr = src_expansion.evaluate(src_expansion.kernel, src_coeff_exprs,
                        dvec_scaled, rscale=src_rscale, sac=sac)

            replace_dict = {d: d/src_rscale for d in dvec}
            taker = ExprDerivativeTaker(expr, dvec)
            rscale_ratio = sym.UnevaluatedExpr(tgt_rscale/src_rscale)

            result = [
                    (taker.diff(mi).xreplace(replace_dict) * rscale_ratio**sum(mi))
                    for mi in self.get_coefficient_identifiers()]

        # }}}

        logger.info("building translation operator: done")
        return result

    def loopy_translate_from(self, src_expansion: ExpansionBase) -> lp.TranslationUnit:
        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase

        if isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
            return self.m2l_translation.loopy_translate(self, src_expansion)

        raise NotImplementedError(
            f"a direct loopy kernel for translation from "
            f"{src_expansion} to {self} is not implemented.")


class VolumeTaylorLocalExpansion(
        VolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):
    pass


class LinearPDEConformingVolumeTaylorLocalExpansion(
        LinearPDEConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):
    pass

# }}}


# {{{ 2D Bessel-based-expansion

@dataclass(frozen=True)
class FourierBesselLocalExpansionMixin(LocalExpansionBase, ABC):

    m2l_translation_override: M2LTranslationBase | None = (
            field(kw_only=True, default=None))

    @property
    @abstractmethod
    def mpole_expn_class(self) -> type[HankelBased2DMultipoleExpansion]:
        ...

    @property
    @override
    def m2l_translation(self) -> M2LTranslationBase:
        if self.m2l_translation_override is not None:
            return self.m2l_translation_override
        else:
            from sumpy.expansion.m2l import DefaultM2LTranslationClassFactory

            factory = DefaultM2LTranslationClassFactory()
            return factory.get_m2l_translation_class(self.kernel, self.__class__)()

    @abstractmethod
    def get_bessel_arg_scaling(self) -> sym.Expr:
        ...

    @override
    def get_storage_index(self, mi: MultiIndex) -> int:
        ind, = mi
        return self.order+ind

    @override
    def get_coefficient_identifiers(self) -> Sequence[MultiIndex]:
        return [(i,) for i in range(-self.order, self.order+1)]

    @override
    def coefficients_from_source(self,
                kernel: Kernel,
                avec: sym.Matrix,
                bvec: sym.Matrix | None,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None
            ) -> Sequence[sym.Expr]:
        if not self.use_rscale:
            rscale = sym.sympify(1)

        from sumpy.symbolic import Hankel1, sym_real_norm_2

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        avec_len = sym_real_norm_2(avec)
        return [kernel.postprocess_at_source(
                    Hankel1(c, arg_scale * avec_len, 0)
                    * rscale ** abs(c)
                    * sym.exp(sym.I * c * source_angle_rel_center), avec)
                    for c, in self.get_coefficient_identifiers()]

    @override
    def evaluate(self,
                kernel: Kernel,
                coeffs: Sequence[sym.Expr],
                bvec: sym.Matrix,
                rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> sym.Expr:
        if not self.use_rscale:
            rscale = sym.sympify(1)

        from sumpy.symbolic import BesselJ, sym_real_norm_2
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sym.sympify(sum(coeffs[self.get_storage_index((c,))]
                   * kernel.postprocess_at_target(
                       BesselJ(c, arg_scale * bvec_len, 0)
                       / rscale ** abs(c)
                       * sym.exp(sym.I * c * -target_angle_rel_center), bvec)
                for c, in self.get_coefficient_identifiers()))

    @override
    def translate_from(self,
                src_expansion: LocalExpansionBase | MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                m2l_translation_classes_dependent_data: (
                       TranslationClassesDepData | None) = None
            ) -> Sequence[sym.Expr]:
        from sumpy.symbolic import BesselJ, sym_real_norm_2

        if not self.use_rscale:
            src_rscale = sym.sympify(1)
            tgt_rscale = sym.sympify(1)

        arg_scale = self.get_bessel_arg_scaling()

        if isinstance(src_expansion, type(self)):
            dvec_len = sym_real_norm_2(dvec)
            new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])

            translated_coeffs: list[sym.Expr] = []
            for j, in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum((src_coeff_exprs[src_expansion.get_storage_index((m,))]
                         * BesselJ(m - j, arg_scale * dvec_len, 0)
                         / src_rscale ** abs(m)
                         * tgt_rscale ** abs(j)
                         * sym.exp(sym.I * (m - j) * -new_center_angle_rel_old_center)
                    for m, in src_expansion.get_coefficient_identifiers()),
                    sym.sympify(0)))

            return translated_coeffs

        if isinstance(src_expansion, self.mpole_expn_class):
            return self.m2l_translation.translate(self, src_expansion,
                src_coeff_exprs, src_rscale, dvec, tgt_rscale, sac,
                m2l_translation_classes_dependent_data)

        raise RuntimeError(
            "do not know how to translate "
            f"{type(src_expansion).__name__} to {type(self).__name__}")

    def loopy_translate_from(self, src_expansion: ExpansionBase) -> lp.TranslationUnit:
        if isinstance(src_expansion, self.mpole_expn_class):
            return self.m2l_translation.loopy_translate(self, src_expansion)

        raise NotImplementedError(
            f"a direct loopy kernel for translation from "
            f"{src_expansion} to {self} is not implemented.")


class H2DLocalExpansion(FourierBesselLocalExpansionMixin):
    def __post_init__(self) -> None:
        from sumpy.kernel import HelmholtzKernel

        kernel = self.kernel.get_base_kernel()
        if not (isinstance(kernel, HelmholtzKernel) and kernel.dim == 2):
            raise TypeError(
                f"{type(self).__name__} can only be applied to 2D HelmholtzKernel: "
                f"{kernel!r}")

    @property
    @override
    def mpole_expn_class(self) -> type[HankelBased2DMultipoleExpansion]:
        return H2DMultipoleExpansion

    @override
    def get_bessel_arg_scaling(self) -> sym.Expr:
        from sumpy.kernel import HelmholtzKernel
        kernel = self.kernel.get_base_kernel()
        assert isinstance(kernel, HelmholtzKernel)

        return sym.Symbol(kernel.helmholtz_k_name)


class Y2DLocalExpansion(FourierBesselLocalExpansionMixin):
    def __post_init__(self) -> None:
        from sumpy.kernel import YukawaKernel

        kernel = self.kernel.get_base_kernel()
        if not (isinstance(kernel, YukawaKernel) and kernel.dim == 2):
            raise TypeError(
                f"{type(self).__name__} can only be applied to 2D YukawaKernel: "
                f"{kernel!r}")

    @property
    @override
    def mpole_expn_class(self) -> type[HankelBased2DMultipoleExpansion]:
        return Y2DMultipoleExpansion

    @override
    def get_bessel_arg_scaling(self) -> sym.Expr:
        from sumpy.kernel import YukawaKernel
        kernel = self.kernel.get_base_kernel()
        assert isinstance(kernel, YukawaKernel)

        return sym.I * sym.Symbol(kernel.yukawa_lambda_name)

# }}}

# vim: fdm=marker
