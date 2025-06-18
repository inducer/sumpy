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
from abc import abstractmethod

from pytools import single_valued

import sumpy.symbolic as sym
from sumpy.expansion import (
    ExpansionBase,
    LinearPDEConformingVolumeTaylorExpansion,
    VolumeTaylorExpansion,
    VolumeTaylorExpansionMixin,
)
from sumpy.tools import add_to_sac, mi_increment_axis


logger = logging.getLogger(__name__)

__doc__ = """

.. autoclass:: LocalExpansionBase
.. autoclass:: VolumeTaylorLocalExpansion
.. autoclass:: H2DLocalExpansion
.. autoclass:: Y2DLocalExpansion
.. autoclass:: LineTaylorLocalExpansion

"""


class LocalExpansionBase(ExpansionBase):
    """Base class for local expansions.

    .. automethod:: translate_from
    """

    init_arg_names = ("kernel", "order", "use_rscale", "m2l_translation")

    def __init__(self, kernel, order, use_rscale=None,
            m2l_translation=None):
        super().__init__(kernel, order, use_rscale)
        self.m2l_translation = m2l_translation

    def with_kernel(self, kernel):
        return type(self)(kernel, self.order, self.use_rscale,
            self.m2l_translation)

    def update_persistent_hash(self, key_hash, key_builder):
        super().update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, self.m2l_translation)

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.kernel == other.kernel
            and self.order == other.order
            and self.use_rscale == other.use_rscale
            and self.m2l_translation == other.m2l_translation
        )

    @abstractmethod
    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None, m2l_translation_classes_dependent_data=None):
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

class LineTaylorLocalExpansion(LocalExpansionBase):
    def __init__(self, kernel, order, tau=1, use_rscale=None, m2l_translation=None):
        super().__init__(kernel, order, use_rscale, m2l_translation)
        self.tau = tau
    

    def get_storage_index(self, k):
        return k

    def get_coefficient_identifiers(self):
        return list(range(self.order+1))

    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        # no point in heeding rscale here--just ignore it
        if bvec is None:
            raise RuntimeError("cannot use line-Taylor expansions in a setting "
                    "where the center-target vector is not known at coefficient "
                    "formation")

        tau = sym.Symbol("tau")

        avec_line = avec + tau*bvec

        line_kernel = kernel.get_expression(avec_line)

        from sumpy.symbolic import USE_SYMENGINE

        if USE_SYMENGINE:
            from sumpy.derivative_taker import ExprDerivativeTaker
            deriv_taker = ExprDerivativeTaker(line_kernel, (tau,), sac=sac, rscale=1)

            return [kernel.postprocess_at_source(
                        deriv_taker.diff(i), avec).subs(tau, 0)
                    for i in self.get_coefficient_identifiers()]
        else:
            # Workaround for sympy. The automatic distribution after
            # single-variable diff makes the expressions very large
            # (https://github.com/sympy/sympy/issues/4596), so avoid doing
            # single variable diff.
            #
            # See also https://gitlab.tiker.net/inducer/pytential/merge_requests/12

            return [kernel.postprocess_at_source(
                            line_kernel.diff(tau, i), avec)
                    .subs(tau, 0)
                    for i in self.get_coefficient_identifiers()]

    def evaluate(self, tgt_kernel, coeffs, bvec, rscale, sac=None):
        # no point in heeding rscale here--just ignore it

        return sym.Add(*(
                coeffs[self.get_storage_index(i)] / math.factorial(i) * self.tau**i
                for i in self.get_coefficient_identifiers()))

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None, m2l_translation_classes_dependent_data=None):
        raise NotImplementedError

# }}}


# {{{ Asymline taylor
class AsymLineTaylorLocalExpansion(LocalExpansionBase):
    def __init__(self, kernel, asymptotic, order, tau=1, use_rscale=None, m2l_translation=None):
        super().__init__(kernel, order, use_rscale, m2l_translation)
        self.asymptotic = asymptotic
        self.tau = tau
    
    
    def get_storage_index(self, k):
        return k

    def get_coefficient_identifiers(self):
        return list(range(self.order+1))

    def get_asymptotic_expression(self, scaled_dist_vec):
        from sumpy.symbolic import PymbolicToSympyMapperWithSymbols, Symbol

        expr = PymbolicToSympyMapperWithSymbols()(self.asymptotic)
        expr = expr.xreplace({Symbol(f"d{i}"): dist_vec_i for i, dist_vec_i in enumerate(scaled_dist_vec)})
        
        tau = sym.Symbol("tau")
    
        b = scaled_dist_vec.applyfunc(lambda expr: expr.coeff(tau))
        a = scaled_dist_vec - tau*b
        expr = expr.subs({Symbol(f"a{i}"): a_i for i, a_i in enumerate(a)})
        expr = expr.subs({Symbol(f"b{i}"): b_i for i, b_i in enumerate(b)})
        
        return expr
    
    
    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        # no point in heeding rscale here--just ignore it
        if bvec is None:
            raise RuntimeError("cannot use line-Taylor expansions in a setting "
                    "where the center-target vector is not known at coefficient "
                    "formation")

        tau = sym.Symbol("tau")

        avec_line = avec + tau*bvec
        line_kernel = kernel.get_expression(avec_line) / self.get_asymptotic_expression(avec_line)
        
        from sumpy.symbolic import USE_SYMENGINE
        if USE_SYMENGINE:
            
            from sumpy.derivative_taker import ExprDerivativeTaker
            deriv_taker = ExprDerivativeTaker(line_kernel, (tau,), sac=sac, rscale=1)

            return [kernel.postprocess_at_source(
                        deriv_taker.diff(i), avec).subs(tau, 0)
                    for i in self.get_coefficient_identifiers()]
        else:
            # Workaround for sympy. The automatic distribution after
            # single-variable diff makes the expressions very large
            # (https://github.com/sympy/sympy/issues/4596), so avoid doing
            # single variable diff.
            #
            # See also https://gitlab.tiker.net/inducer/pytential/merge_requests/12
       
            return [kernel.postprocess_at_source(
                            line_kernel.diff(tau, i), avec)
                    .subs(tau, 0)
                    for i in self.get_coefficient_identifiers()]


    def evaluate(self, tgt_kernel, coeffs, bvec, rscale, sac=None):
        # no point in heeding rscale here--just ignore it
        
        return sym.Add(*(
                coeffs[self.get_storage_index(i)] / math.factorial(i) * self.tau**i
                for i in self.get_coefficient_identifiers()))

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None, m2l_translation_classes_dependent_data=None):
        raise NotImplementedError

# }}}







# {{{ volume taylor

class VolumeTaylorLocalExpansionBase(VolumeTaylorExpansionMixin, LocalExpansionBase):
    """
    Coefficients represent derivative values of the kernel.
    """

    def __init__(self, kernel, order, use_rscale=None,
            m2l_translation=None):
        if not m2l_translation:
            from sumpy.expansion.m2l import DefaultM2LTranslationClassFactory
            factory = DefaultM2LTranslationClassFactory()
            m2l_translation = factory.get_m2l_translation_class(kernel,
                self.__class__)()
        super().__init__(kernel, order, use_rscale, m2l_translation)

    def coefficients_from_source_vec(self, kernels, avec, bvec, rscale, weights,
            sac=None):
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
            rscale = 1

        base_kernel = single_valued(knl.get_base_kernel() for knl in kernels)
        base_taker = base_kernel.get_derivative_taker(avec, rscale, sac)
        result = [0]*len(self)

        for knl, weight in zip(kernels, weights, strict=True):
            taker = knl.postprocess_at_source(base_taker, avec)
            # Following is a hack to make sure cse works.
            if 1:
                def save_temp(x):
                    return add_to_sac(sac, weight * x)  # noqa: B023

                for i, mi in enumerate(self.get_coefficient_identifiers()):
                    result[i] += taker.diff(mi, save_temp)
            else:
                def save_temp(x):
                    return add_to_sac(sac, x)

                for i, mi in enumerate(self.get_coefficient_identifiers()):
                    result[i] += weight * taker.diff(mi, save_temp)

        return result

    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        return self.coefficients_from_source_vec((kernel,), avec, bvec,
                rscale=rscale, weights=(1,), sac=sac)

    def evaluate(self, kernel, coeffs, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        evaluated_coeffs = (
            self.expansion_terms_wrangler.get_full_kernel_derivatives_from_stored(
                coeffs, rscale, sac=sac))

        bvec_scaled = [b*rscale**-1 for b in bvec]
        from sumpy.tools import mi_factorial, mi_power

        result = sum(
            coeff
            * mi_power(bvec_scaled, mi, evaluate=False)
            / mi_factorial(mi)
            for coeff, mi in zip(
                    evaluated_coeffs, self.get_full_coefficient_identifiers(),
                    strict=True))

        return kernel.postprocess_at_target(result, bvec)

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None, _fast_version=True,
            m2l_translation_classes_dependent_data=None):
        logger.info("building translation operator for %s: %s(%d) -> %s(%d): start",
                    src_expansion.kernel,
                    type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order)

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase

        # {{{ M2L

        if isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
            result = self.m2l_translation.translate(self, src_expansion,
                src_coeff_exprs, src_rscale, dvec, tgt_rscale, sac,
                m2l_translation_classes_dependent_data)

            logger.info("building translation operator: done")
            return result

        # }}}

        # {{{ L2L

        # not coming from a Taylor multipole: expand via derivatives
        rscale_ratio = add_to_sac(sac, tgt_rscale/src_rscale)

        src_wrangler = src_expansion.expansion_terms_wrangler
        src_coeffs = (
            src_wrangler.get_full_kernel_derivatives_from_stored(
                src_coeff_exprs, src_rscale, sac=sac))
        src_mis = \
            src_expansion.expansion_terms_wrangler.get_full_coefficient_identifiers()

        src_mi_to_index = {mi: i for i, mi in enumerate(src_mis)}

        tgt_mis = \
            self.expansion_terms_wrangler.get_coefficient_identifiers()
        tgt_mi_to_index = {mi: i for i, mi in enumerate(tgt_mis)}

        tgt_split = self.expansion_terms_wrangler._split_coeffs_into_hyperplanes()

        p = max(sum(mi) for mi in src_mis)
        result = [0] * len(tgt_mis)

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
            cur_dim_input_coeffs = src_coeffs
            # O(1) iterations
            for d in dims:
                cur_dim_output_coeffs = [0] * len(src_mis)
                # Only O(p^{d-1}) operations are used in compressed
                # O(p^d) operations are used in full
                for out_i, out_mi in enumerate(src_mis):
                    # O(p) iterations
                    for q in range(p+1-sum(out_mi)):
                        src_mi = mi_increment_axis(out_mi, d, q)
                        if src_mi in src_mi_to_index:
                            cur_dim_output_coeffs[out_i] += (dvec[d]/src_rscale)**q \
                                * cur_dim_input_coeffs[src_mi_to_index[src_mi]] \
                                / math.factorial(q)
                # Y at the end of the iteration becomes the source coefficients
                # for the next iteration
                cur_dim_input_coeffs = cur_dim_output_coeffs

            for mi in tgt_mis:
                # In L2L, source level usually has same or higher order than target
                # level. If not, extra coeffs in target level are zero filled.
                if mi not in src_mi_to_index:
                    result[tgt_mi_to_index[mi]] = 0
                else:
                    # Add to result after scaling
                    result[tgt_mi_to_index[mi]] += \
                        cur_dim_output_coeffs[src_mi_to_index[mi]] \
                        * rscale_ratio ** sum(mi)

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
            dvec_scaled = [d*src_rscale for d in dvec]
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

    def loopy_translate_from(self, src_expansion):
        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase

        if isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
            return self.m2l_translation.loopy_translate(self, src_expansion)

        raise NotImplementedError(
            f"A direct loopy kernel for translation from "
            f"{src_expansion} to {self} is not implemented.")


class VolumeTaylorLocalExpansion(
        VolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None, m2l_translation=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale,
            m2l_translation=m2l_translation)
        VolumeTaylorExpansion.__init__(self, kernel, order, use_rscale)


class LinearPDEConformingVolumeTaylorLocalExpansion(
        LinearPDEConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None, m2l_translation=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale,
            m2l_translation=m2l_translation)
        LinearPDEConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)


class LaplaceConformingVolumeTaylorLocalExpansion(
        LinearPDEConformingVolumeTaylorLocalExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("LaplaceConformingVolumeTaylorLocalExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorLocalExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class HelmholtzConformingVolumeTaylorLocalExpansion(
        LinearPDEConformingVolumeTaylorLocalExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("HelmholtzConformingVolumeTaylorLocalExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorLocalExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class BiharmonicConformingVolumeTaylorLocalExpansion(
        LinearPDEConformingVolumeTaylorLocalExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("BiharmonicConformingVolumeTaylorLocalExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorLocalExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

# }}}


# {{{ 2D Bessel-based-expansion

class _FourierBesselLocalExpansion(LocalExpansionBase):
    def __init__(self,
            kernel, order, mpole_expn_class,
            use_rscale=None, m2l_translation=None):
        if not m2l_translation:
            from sumpy.expansion.m2l import DefaultM2LTranslationClassFactory
            factory = DefaultM2LTranslationClassFactory()
            m2l_translation = (
                factory.get_m2l_translation_class(kernel, self.__class__)())

        super().__init__(kernel, order, use_rscale, m2l_translation)
        self.mpole_expn_class = mpole_expn_class

    @abstractmethod
    def get_bessel_arg_scaling(self):
        pass

    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_identifiers(self):
        return list(range(-self.order, self.order+1))

    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import Hankel1, sym_real_norm_2

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        avec_len = sym_real_norm_2(avec)
        return [kernel.postprocess_at_source(
                    Hankel1(c, arg_scale * avec_len, 0)
                    * rscale ** abs(c)
                    * sym.exp(sym.I * c * source_angle_rel_center), avec)
                    for c in self.get_coefficient_identifiers()]

    def evaluate(self, kernel, coeffs, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import BesselJ, sym_real_norm_2
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(c)]
                   * kernel.postprocess_at_target(
                       BesselJ(c, arg_scale * bvec_len, 0)
                       / rscale ** abs(c)
                       * sym.exp(sym.I * c * -target_angle_rel_center), bvec)
                for c in self.get_coefficient_identifiers())

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None, m2l_translation_classes_dependent_data=None):
        from sumpy.symbolic import BesselJ, sym_real_norm_2

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        arg_scale = self.get_bessel_arg_scaling()

        if isinstance(src_expansion, type(self)):
            dvec_len = sym_real_norm_2(dvec)
            new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])
            translated_coeffs = []

            for j in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum(src_coeff_exprs[src_expansion.get_storage_index(m)]
                        * BesselJ(m - j, arg_scale * dvec_len, 0)
                        / src_rscale ** abs(m)
                        * tgt_rscale ** abs(j)
                        * sym.exp(sym.I * (m - j) * -new_center_angle_rel_old_center)
                    for m in src_expansion.get_coefficient_identifiers()))
            return translated_coeffs

        if isinstance(src_expansion, self.mpole_expn_class):
            return self.m2l_translation.translate(self, src_expansion,
                src_coeff_exprs, src_rscale, dvec, tgt_rscale, sac,
                m2l_translation_classes_dependent_data)

        raise RuntimeError(
            "do not know how to translate "
            f"{type(src_expansion).__name__} to {type(self).__name__}")

    def loopy_translate_from(self, src_expansion):
        if isinstance(src_expansion, self.mpole_expn_class):
            return self.m2l_translation.loopy_translate(self, src_expansion)

        raise NotImplementedError(
            f"A direct loopy kernel for translation from "
            f"{src_expansion} to {self} is not implemented.")


class H2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None, m2l_translation=None):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dim == 2)

        from sumpy.expansion.multipole import H2DMultipoleExpansion
        super().__init__(kernel, order, H2DMultipoleExpansion,
            use_rscale=use_rscale,
            m2l_translation=m2l_translation)

    def get_bessel_arg_scaling(self):
        return sym.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)


class Y2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None, m2l_translation=None):
        from sumpy.kernel import YukawaKernel
        assert (isinstance(kernel.get_base_kernel(), YukawaKernel)
                and kernel.dim == 2)

        from sumpy.expansion.multipole import Y2DMultipoleExpansion
        super().__init__(kernel, order, Y2DMultipoleExpansion,
            use_rscale=use_rscale,
            m2l_translation=m2l_translation)

    def get_bessel_arg_scaling(self):
        return sym.I * sym.Symbol(self.kernel.get_base_kernel().yukawa_lambda_name)

# }}}

# vim: fdm=marker
