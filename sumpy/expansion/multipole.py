from __future__ import division, absolute_import

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

from six.moves import range, zip
import sumpy.symbolic as sym  # noqa

from sumpy.symbolic import vector_xreplace
from sumpy.expansion import (
    ExpansionBase, VolumeTaylorExpansion, LaplaceConformingVolumeTaylorExpansion,
    HelmholtzConformingVolumeTaylorExpansion,
    BiharmonicConformingVolumeTaylorExpansion)

import logging
logger = logging.getLogger(__name__)


__doc__ = """

.. autoclass:: VolumeTaylorMultipoleExpansion
.. autoclass:: H2DMultipoleExpansion
.. autoclass:: Y2DMultipoleExpansion

"""


class MultipoleExpansionBase(ExpansionBase):
    pass


# {{{ volume taylor

class VolumeTaylorMultipoleExpansionBase(MultipoleExpansionBase):
    """
    Coefficients represent the terms in front of the kernel derivatives.
    """

    def coefficients_from_source(self, avec, bvec, rscale):
        from sumpy.kernel import DirectionalSourceDerivative
        kernel = self.kernel

        from sumpy.tools import mi_power, mi_factorial

        if not self.use_rscale:
            rscale = 1

        coeff_identifiers = self.get_full_coefficient_identifiers()
        if isinstance(kernel, DirectionalSourceDerivative):
            result = [0] * len(coeff_identifiers)
            for i, mi in enumerate(coeff_identifiers):
                result[i] = self.kernel.postprocess_at_source(
                        mi_power(avec, mi), avec)
                result[i] /= (mi_factorial(mi) * rscale ** sum(mi))
        else:
            avec = [sym.UnevaluatedExpr(a * rscale**-1) for a in avec]
            result = [
                    mi_power(avec, mi) / mi_factorial(mi)
                    for mi in coeff_identifiers]
        return (
            self.expansion_terms_wrangler.get_stored_mpole_coefficients_from_full(
                result, rscale))

    def get_scaled_multipole(self, expr, bvec, rscale, nderivatives,
            nderivatives_for_scaling=None):
        if nderivatives_for_scaling is None:
            nderivatives_for_scaling = nderivatives

        if self.kernel.has_efficient_scale_adjustment:
            return (
                    self.kernel.adjust_for_kernel_scaling(
                        vector_xreplace(
                            expr,
                            bvec, bvec * rscale**-1),
                        rscale, nderivatives)
                    / rscale ** (nderivatives - nderivatives_for_scaling))
        else:
            return (rscale**nderivatives_for_scaling * expr)

    def evaluate(self, coeffs, bvec, rscale, knl=None):
        from sumpy.tools import MiDerivativeTakerWrapper
        from pytools import single_valued
        if not self.use_rscale:
            rscale = 1
        if knl is None:
            knl = self.kernel

        taker = self.get_kernel_derivative_taker(bvec)
        expr_dict = {(0,)*self.dim: 1}
        expr_dict = knl.get_derivative_transformation_at_target(expr_dict)
        pp_nderivatives = single_valued(sum(mi) for mi in expr_dict.keys())

        result = []
        for coeff, mi in zip(coeffs, self.get_coefficient_identifiers()):
            wrapper = MiDerivativeTakerWrapper(taker, mi)
            mi_expr = knl.postprocess_at_target(wrapper, bvec)
            expr = coeff * self.get_scaled_multipole(mi_expr, bvec, rscale,
                    sum(mi) + pp_nderivatives, sum(mi))
            result.append(expr)

        result = sym.Add(*tuple(result))
        #return knl.postprocess_at_target(result, bvec)
        return result

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale):
        if not isinstance(src_expansion, type(self)):
            raise RuntimeError("do not know how to translate %s to "
                    "Taylor multipole expansion"
                               % type(src_expansion).__name__)

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        logger.info("building translation operator: %s(%d) -> %s(%d): start"
                % (type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order))

        from sumpy.tools import mi_factorial

        src_mi_to_index = dict((mi, i) for i, mi in enumerate(
            src_expansion.get_full_coefficient_identifiers()))

        tgt_mi_to_index = dict((mi, i) for i, mi in enumerate(
            self.get_full_coefficient_identifiers()))

        src_coeff_exprs_full = \
            [0]*len(src_expansion.get_full_coefficient_identifiers())

        for i, mi in enumerate(src_expansion.get_coefficient_identifiers()):
            src_coeff_exprs_full[src_mi_to_index[mi]] = src_coeff_exprs[i] * \
                sym.UnevaluatedExpr(src_rscale/tgt_rscale)**sum(mi)

        # This algorithm uses the observation that M2M coefficients
        # have the following form in 2D
        #
        # $B_{m, n} = \sum_{i\le m, j\le n} A_{i, j}
        #             d_x^i d_y^j \binom{m}{i} \binom{n}{j}$
        # and can be rewritten as follows.
        #
        # Let $Y_{m, n} = \sum_{i\le m} A_{i, n} d_x^i \binom{m}{i}$.
        #
        # Then, $B_{m, n} = \sum_{j\le n} Y_{m, j} d_y^j \binom{n}{j}$.
        #
        # $Y_{m, n}$ are $p^2$ number of temporary variables that are
        # reused for different M2M coefficients and costs $p$ per variable.
        # Total cost for calculating $Y_{m, n}$ is $p^3$ and similar
        # for $B_{m, n}$

        # In other words, we're better off computing the translation
        # one dimension at a time. If the coefficients in the source
        # expansion have the form (0, m) and (m, 0), then we calculate
        # the output from (0, m) with the second dimension as the fastest
        # varying dimension and then calculate the output from (m, 0)
        # with the first dimension as the fastest varying dimension.
        # However, the contribution from (0, 0) is counted twice and we
        # need to subtract that.

        src_split = \
            src_expansion.expansion_terms_wrangler._get_coeff_identifier_split()
        result = [0] * len(self.get_full_coefficient_identifiers())

        non_zero_coeffs_per_dim = [[] for d in range(self.dim)]
        for d, src_mi in src_split:
            non_zero_coeffs_per_dim[d] += src_mi

        for const_dim in set(d for d, _ in src_split):
            dim_coeffs_to_translate = \
                [0] * len(src_expansion.get_full_coefficient_identifiers())
            for mi in non_zero_coeffs_per_dim[const_dim]:
                idx = src_mi_to_index[mi]
                dim_coeffs_to_translate[idx] = src_coeff_exprs_full[idx]

            # Use the const_dim as the last dimension to vary
            dims = list(range(const_dim)) + \
                   list(range(const_dim+1, self.dim)) + [const_dim]
            for d in dims:
                temp = [0] * len(src_expansion.get_full_coefficient_identifiers())
                for i, tgt_mi in enumerate(
                        src_expansion.get_full_coefficient_identifiers()):
                    src_mis_per_dim = []
                    for mi_i in range(tgt_mi[d]+1):
                        new_mi = list(tgt_mi)
                        new_mi[d] = mi_i
                        src_mis_per_dim.append(tuple(new_mi))

                    for src_mi in src_mis_per_dim:
                        try:
                            src_index = src_mi_to_index[src_mi]
                        except KeyError:
                            # Omitted coefficients: not life-threatening
                            continue

                        contrib = dim_coeffs_to_translate[src_index]
                        for idim in range(self.dim):
                            n = tgt_mi[idim]
                            k = src_mi[idim]
                            assert n >= k
                            contrib /= mi_factorial((n-k,))
                            contrib *= \
                                sym.UnevaluatedExpr(dvec[idim]/tgt_rscale)**(n-k)

                        temp[i] += contrib

                dim_coeffs_to_translate = temp[:]

            for i, mi in enumerate(src_expansion.get_full_coefficient_identifiers()):
                result[tgt_mi_to_index[mi]] += dim_coeffs_to_translate[i]

        # {{{ simpler, functionally equivalent code
        if 0:
            src_mi_to_index = dict((mi, i) for i, mi in enumerate(
                src_expansion.get_coefficient_identifiers()))
            result = [0] * len(self.get_full_coefficient_identifiers())

            for i, mi in enumerate(src_expansion.get_coefficient_identifiers()):
                src_coeff_exprs[i] *= mi_factorial(mi)

            from pytools import generate_nonnegative_integer_tuples_below as gnitb

            for i, tgt_mi in enumerate(
                    self.get_full_coefficient_identifiers()):

                tgt_mi_plus_one = tuple(mi_i + 1 for mi_i in tgt_mi)

                for src_mi in gnitb(tgt_mi_plus_one):
                    try:
                        src_index = src_mi_to_index[src_mi]
                    except KeyError:
                        # Omitted coefficients: not life-threatening
                        continue

                    contrib = src_coeff_exprs[src_index]

                    for idim in range(self.dim):
                        n = tgt_mi[idim]
                        k = src_mi[idim]
                        assert n >= k
                        from sympy import binomial
                        contrib *= (binomial(n, k)
                                * sym.UnevaluatedExpr(dvec[idim]/tgt_rscale)**(n-k))

                    result[i] += (contrib
                        * sym.UnevaluatedExpr(src_rscale/tgt_rscale)**sum(src_mi))

                result[i] /= mi_factorial(tgt_mi)
        # }}}

        logger.info("building translation operator: done")
        return (
            self.expansion_terms_wrangler.get_stored_mpole_coefficients_from_full(
                result, tgt_rscale))


class VolumeTaylorMultipoleExpansion(
        VolumeTaylorExpansion,
        VolumeTaylorMultipoleExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorMultipoleExpansionBase.__init__(self, kernel, order, use_rscale)
        VolumeTaylorExpansion.__init__(self, kernel, order, use_rscale)


class LaplaceConformingVolumeTaylorMultipoleExpansion(
        LaplaceConformingVolumeTaylorExpansion,
        VolumeTaylorMultipoleExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorMultipoleExpansionBase.__init__(self, kernel, order, use_rscale)
        LaplaceConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)


class HelmholtzConformingVolumeTaylorMultipoleExpansion(
        HelmholtzConformingVolumeTaylorExpansion,
        VolumeTaylorMultipoleExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorMultipoleExpansionBase.__init__(self, kernel, order, use_rscale)
        HelmholtzConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)


class BiharmonicConformingVolumeTaylorMultipoleExpansion(
        BiharmonicConformingVolumeTaylorExpansion,
        VolumeTaylorMultipoleExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorMultipoleExpansionBase.__init__(self, kernel, order, use_rscale)
        BiharmonicConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)

# }}}


# {{{ 2D Hankel-based expansions

class _HankelBased2DMultipoleExpansion(MultipoleExpansionBase):
    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_identifiers(self):
        return list(range(-self.order, self.order+1))

    def coefficients_from_source(self, avec, bvec, rscale):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        bessel_j = sym.Function("bessel_j")
        avec_len = sym_real_norm_2(avec)

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        return [
                self.kernel.postprocess_at_source(
                    bessel_j(l, arg_scale * avec_len)
                    / rscale ** abs(l)
                    * sym.exp(sym.I * l * -source_angle_rel_center),
                    avec)
                for l in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec, rscale, knl=None):
        if not self.use_rscale:
            rscale = 1
        if knl is None:
            knl = self.kernel

        from sumpy.symbolic import sym_real_norm_2
        hankel_1 = sym.Function("hankel_1")
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(l)]
                   * knl.postprocess_at_target(
                       hankel_1(l, arg_scale * bvec_len)
                       * rscale ** abs(l)
                       * sym.exp(sym.I * l * target_angle_rel_center), bvec)
                for l in self.get_coefficient_identifiers())

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale):
        if not isinstance(src_expansion, type(self)):
            raise RuntimeError("do not know how to translate %s to %s"
                               % (type(src_expansion).__name__,
                                   type(self).__name__))

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        dvec_len = sym_real_norm_2(dvec)
        bessel_j = sym.Function("bessel_j")
        new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        translated_coeffs = []
        for l in self.get_coefficient_identifiers():
            translated_coeffs.append(
                sum(src_coeff_exprs[src_expansion.get_storage_index(m)]
                    * bessel_j(m - l, arg_scale * dvec_len)
                    * src_rscale ** abs(m)
                    / tgt_rscale ** abs(l)
                    * sym.exp(sym.I * (m - l) * new_center_angle_rel_old_center)
                for m in src_expansion.get_coefficient_identifiers()))
        return translated_coeffs


class H2DMultipoleExpansion(_HankelBased2DMultipoleExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dim == 2)

        super(H2DMultipoleExpansion, self).__init__(
                kernel, order, use_rscale=use_rscale)

    def get_bessel_arg_scaling(self):
        return sym.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)


class Y2DMultipoleExpansion(_HankelBased2DMultipoleExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import YukawaKernel
        assert (isinstance(kernel.get_base_kernel(), YukawaKernel)
                and kernel.dim == 2)

        super(Y2DMultipoleExpansion, self).__init__(
                kernel, order, use_rscale=use_rscale)

    def get_bessel_arg_scaling(self):
        return sym.I * sym.Symbol(self.kernel.get_base_kernel().yukawa_lambda_name)

# }}}

# vim: fdm=marker
