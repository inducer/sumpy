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
import sumpy.symbolic as sym

from sumpy.expansion import (
    ExpansionBase, VolumeTaylorExpansion, LaplaceConformingVolumeTaylorExpansion,
    HelmholtzConformingVolumeTaylorExpansion,
    BiharmonicConformingVolumeTaylorExpansion)


class LocalExpansionBase(ExpansionBase):
    pass


import logging
logger = logging.getLogger(__name__)

__doc__ = """

.. autoclass:: VolumeTaylorLocalExpansion
.. autoclass:: H2DLocalExpansion
.. autoclass:: Y2DLocalExpansion
.. autoclass:: LineTaylorLocalExpansion

"""


# {{{ line taylor

class LineTaylorLocalExpansion(LocalExpansionBase):

    def get_storage_index(self, k):
        return k

    def get_coefficient_identifiers(self):
        return list(range(self.order+1))

    def coefficients_from_source(self, avec, bvec, rscale, sac=None):
        # no point in heeding rscale here--just ignore it
        if bvec is None:
            raise RuntimeError("cannot use line-Taylor expansions in a setting "
                    "where the center-target vector is not known at coefficient "
                    "formation")

        tau = sym.Symbol("tau")

        avec_line = avec + tau*bvec

        line_kernel = self.kernel.get_expression(avec_line)

        from sumpy.symbolic import USE_SYMENGINE

        if USE_SYMENGINE:
            from sumpy.tools import MiDerivativeTaker, my_syntactic_subs
            deriv_taker = MiDerivativeTaker(line_kernel, (tau,))

            return [my_syntactic_subs(
                        self.kernel.postprocess_at_target(
                            self.kernel.postprocess_at_source(
                                deriv_taker.diff(i),
                                avec), bvec),
                        {tau: 0})
                    for i in self.get_coefficient_identifiers()]
        else:
            # Workaround for sympy. The automatic distribution after
            # single-variable diff makes the expressions very large
            # (https://github.com/sympy/sympy/issues/4596), so avoid doing
            # single variable diff.
            #
            # See also https://gitlab.tiker.net/inducer/pytential/merge_requests/12

            return [self.kernel.postprocess_at_target(
                        self.kernel.postprocess_at_source(
                            line_kernel.diff("tau", i), avec),
                        bvec)
                    .subs("tau", 0)
                    for i in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec, rscale, sac=None):
        # no point in heeding rscale here--just ignore it
        from pytools import factorial
        return sym.Add(*(
                coeffs[self.get_storage_index(i)] / factorial(i)
                for i in self.get_coefficient_identifiers()))

# }}}


# {{{ volume taylor

class VolumeTaylorLocalExpansionBase(LocalExpansionBase):
    """
    Coefficients represent derivative values of the kernel.
    """

    def coefficients_from_source(self, avec, bvec, rscale, sac=None):
        from sumpy.tools import MiDerivativeTaker
        ppkernel = self.kernel.postprocess_at_source(
                self.kernel.get_expression(avec), avec)

        taker = MiDerivativeTaker(ppkernel, avec)
        return [
                taker.diff(mi) * rscale ** sum(mi)
                for mi in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec, rscale, sac=None):
        from pytools import factorial

        evaluated_coeffs = (
            self.expansion_terms_wrangler.get_full_kernel_derivatives_from_stored(
                coeffs, rscale))

        bvec = [b*rscale**-1 for b in bvec]
        mi_to_index = dict((mi, i) for i, mi in
                        enumerate(self.get_full_coefficient_identifiers()))

        # Sort multi-indices so that last dimension varies fastest
        sorted_target_mis = sorted(self.get_full_coefficient_identifiers())
        dim = self.dim

        # Start with an invalid "seen" multi-index
        seen_mi = [-1]*dim
        # Local sum keep the sum of the terms that differ by each dimension
        local_sum = [0]*dim
        # Local multiplier keep the scalar that the sum has to be multiplied by
        # when adding to the sum of the preceding dimension.
        local_multiplier = [0]*dim

        # For the multi-indices in 3D, local_sum looks like this:
        #
        # Multi-index | coef | local_sum                              | local_mult
        # (0, 0, 0)   |  c0  | 0, 0,                c0                | 0, 1, 1
        # (0, 0, 1)   |  c1  | 0, 0,                c0+c1*dz          | 0, 1, 1
        # (0, 0, 2)   |  c2  | 0, 0,                c0+c1*dz+c2*dz^2  | 0, 1, 1
        # (0, 1, 0)   |  c3  | 0, c0+c1*dz+c2*dz^2, c3                | 0, 1, dy
        # (0, 1, 1)   |  c4  | 0, c0+c1*dz+c2*dz^2, c3+c4*dz          | 0, 1, dy
        # (0, 1, 2)   |  c5  | 0, c0+c1*dz+c2*dz^2, c3+c4*dz+c5*dz^2  | 0, 1, dy
        # (0, 2, 0)   |  c6  | 0, c0+c1*dz+c2*dz^2, c6                | 0, 1, dy^2
        #             |      |    +dy*(c3+c4*dz+c5*dz^2)              |
        # (0, 2, 1)   |  c7  | 0, c0+c1*dz+c2*dz^2, c6+c7*dz          | 0, 1, dy^2
        #             |      |    +dy*(c3+c4*dz+c5*dz^2)              |
        # (0, 2, 2)   |  c8  | 0, c0+c1*dz+c2*dz^2, c6+c7*dz+x8*dz^2  | 0, 1, dy^2
        #             |      |    +dy*(c3+c4*dz+c5*dz^2)              |
        # (1, 0, 0)   |  c8  | c0+c1*dz+c2*dz^2,         0, 0         | 0, dx, 1
        #             |      |  +dy*(c3+c4*dz+c5*dz^2)                |
        #             |      |  +dy^2*(c6+c7*dz+c8*dz^2)              |

        for mi in sorted_target_mis:

            # {{{ handle the case where a not-last dimension "clicked over"

            # (where d will be that not-last dimension)

            # Iterate in reverse order of dimensions to properly handle a
            # "double click-over".

            for d in reversed(range(dim-1)):
                if seen_mi[d] != mi[d]:
                    # If the dimension d of mi changed from the previous value
                    # then the sum for dimension d+1 is complete, add it to
                    # dimension d after multiplying and restart.

                    local_sum[d] += local_sum[d+1]*local_multiplier[d+1]
                    local_sum[d+1] = 0
                    local_multiplier[d+1] = bvec[d]**mi[d] / factorial(mi[d])

            # }}}

            local_sum[dim-1] += evaluated_coeffs[mi_to_index[mi]] * \
                                    bvec[dim-1]**mi[dim-1] / factorial(mi[dim-1])
            seen_mi = mi

        for d in reversed(range(dim-1)):
            local_sum[d] += local_sum[d+1]*local_multiplier[d+1]

        # {{{ simpler, functionally equivalent code

        if 0:
            from sumpy.tools import mi_power, mi_factorial

            return sum(
                coeff
                * mi_power(bvec, mi, evaluate=False)
                / mi_factorial(mi)
                for coeff, mi in zip(
                        evaluated_coeffs, self.get_full_coefficient_identifiers()))

        # }}}

        return local_sum[0]

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale):
        logger.info("building translation operator: %s(%d) -> %s(%d): start"
                % (type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order))

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansionBase
        if isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase):
            # We know the general form of the multipole expansion is:
            #
            #    coeff0 * diff(kernel, mi0) + coeff1 * diff(kernel, mi1) + ...
            #
            # To get the local expansion coefficients, we take derivatives of
            # the multipole expansion.
            #
            # This code speeds up derivative taking by caching all kernel
            # derivatives.

            taker = src_expansion.get_kernel_derivative_taker(dvec)

            from sumpy.tools import add_mi

            max_mi = [0]*self.dim
            for i in range(self.dim):
                max_mi[i] = max(mi[i] for mi in
                                  src_expansion.get_coefficient_identifiers())
                max_mi[i] += max(mi[i] for mi in
                                  self.get_coefficient_identifiers())

            # Create a expansion terms wrangler for derivatives up to order
            # (tgt order)+(src order) including a corresponding reduction matrix
            tgtplusderiv_exp_terms_wrangler = \
                src_expansion.expansion_terms_wrangler.copy(
                        order=self.order + src_expansion.order, max_mi=tuple(max_mi))
            tgtplusderiv_coeff_ids = \
                tgtplusderiv_exp_terms_wrangler.get_coefficient_identifiers()
            tgtplusderiv_full_coeff_ids = \
                tgtplusderiv_exp_terms_wrangler.get_full_coefficient_identifiers()

            tgtplusderiv_ident_to_index = dict((ident, i) for i, ident in
                                enumerate(tgtplusderiv_full_coeff_ids))

            result = []
            for lexp_mi in self.get_coefficient_identifiers():
                lexp_mi_terms = []

                # Embed the source coefficients in the coefficient array
                # for the (tgt order)+(src order) wrangler, offset by lexp_mi.
                embedded_coeffs = [0] * len(tgtplusderiv_full_coeff_ids)
                for coeff, term in zip(
                        src_coeff_exprs,
                        src_expansion.get_coefficient_identifiers()):
                    embedded_coeffs[
                            tgtplusderiv_ident_to_index[add_mi(lexp_mi, term)]] \
                                    = coeff

                # Compress the embedded coefficient set
                stored_coeffs = tgtplusderiv_exp_terms_wrangler \
                        .get_stored_mpole_coefficients_from_full(
                                embedded_coeffs, src_rscale)

                # Sum the embedded coefficient set
                for i, coeff in enumerate(stored_coeffs):
                    if coeff == 0:
                        continue
                    nderivatives_for_scaling = \
                            sum(tgtplusderiv_coeff_ids[i])-sum(lexp_mi)
                    kernel_deriv = (
                            src_expansion.get_scaled_multipole(
                                taker.diff(tgtplusderiv_coeff_ids[i]),
                                dvec, src_rscale,
                                nderivatives=sum(tgtplusderiv_coeff_ids[i]),
                                nderivatives_for_scaling=nderivatives_for_scaling))

                    lexp_mi_terms.append(
                            coeff * kernel_deriv * tgt_rscale**sum(lexp_mi))
                result.append(sym.Add(*lexp_mi_terms))

        else:
            from sumpy.tools import MiDerivativeTaker
            # Rscale/operand magnitude is fairly sensitive to the order of
            # operations--which is something we don't have fantastic control
            # over at the symbolic level. Scaling dvec, then differentiating,
            # and finally rescaling dvec leaves the expression needing a scaling
            # to compensate for differentiating which is done at the end.
            # This moves the two cancelling "rscales" closer to each other at
            # the end in the hope of helping rscale magnitude.
            dvec_scaled = [d*src_rscale for d in dvec]
            expr = src_expansion.evaluate(src_coeff_exprs, dvec_scaled,
                        rscale=src_rscale)
            replace_dict = dict((d, d/src_rscale) for d in dvec)
            taker = MiDerivativeTaker(expr, dvec)
            rscale_ratio = sym.UnevaluatedExpr(tgt_rscale/src_rscale)
            result = [
                    (taker.diff(mi).xreplace(replace_dict) * rscale_ratio**sum(mi))
                    for mi in self.get_coefficient_identifiers()]

        logger.info("building translation operator: done")
        return result


class VolumeTaylorLocalExpansion(
        VolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale)
        VolumeTaylorExpansion.__init__(self, kernel, order, use_rscale)


class LaplaceConformingVolumeTaylorLocalExpansion(
        LaplaceConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale)
        LaplaceConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)


class HelmholtzConformingVolumeTaylorLocalExpansion(
        HelmholtzConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale)
        HelmholtzConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)


class BiharmonicConformingVolumeTaylorLocalExpansion(
        BiharmonicConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale)
        BiharmonicConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)

# }}}


# {{{ 2D Bessel-based-expansion

class _FourierBesselLocalExpansion(LocalExpansionBase):
    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_identifiers(self):
        return list(range(-self.order, self.order+1))

    def coefficients_from_source(self, avec, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        hankel_1 = sym.Function("hankel_1")

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        avec_len = sym_real_norm_2(avec)
        return [self.kernel.postprocess_at_source(
                    hankel_1(c, arg_scale * avec_len)
                    * rscale ** abs(c)
                    * sym.exp(sym.I * c * source_angle_rel_center), avec)
                    for c in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        bessel_j = sym.Function("bessel_j")
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(c)]
                   * self.kernel.postprocess_at_target(
                       bessel_j(c, arg_scale * bvec_len)
                       / rscale ** abs(c)
                       * sym.exp(sym.I * c * -target_angle_rel_center), bvec)
                for c in self.get_coefficient_identifiers())

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale):
        from sumpy.symbolic import sym_real_norm_2

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        arg_scale = self.get_bessel_arg_scaling()

        if isinstance(src_expansion, type(self)):
            dvec_len = sym_real_norm_2(dvec)
            bessel_j = sym.Function("bessel_j")
            new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])
            translated_coeffs = []
            for j in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum(src_coeff_exprs[src_expansion.get_storage_index(m)]
                        * bessel_j(m - j, arg_scale * dvec_len)
                        / src_rscale ** abs(m)
                        * tgt_rscale ** abs(j)
                        * sym.exp(sym.I * (m - j) * -new_center_angle_rel_old_center)
                    for m in src_expansion.get_coefficient_identifiers()))
            return translated_coeffs

        if isinstance(src_expansion, self.mpole_expn_class):
            dvec_len = sym_real_norm_2(dvec)
            hankel_1 = sym.Function("hankel_1")
            new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])
            translated_coeffs = []
            for j in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum(
                        (-1) ** j
                        * hankel_1(m + j, arg_scale * dvec_len)
                        * src_rscale ** abs(m)
                        * tgt_rscale ** abs(j)
                        * sym.exp(sym.I * (m + j) * new_center_angle_rel_old_center)
                        * src_coeff_exprs[src_expansion.get_storage_index(m)]
                        for m in src_expansion.get_coefficient_identifiers()))
            return translated_coeffs

        raise RuntimeError("do not know how to translate %s to %s"
                           % (type(src_expansion).__name__,
                               type(self).__name__))


class H2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dim == 2)

        super(H2DLocalExpansion, self).__init__(kernel, order, use_rscale)

        from sumpy.expansion.multipole import H2DMultipoleExpansion
        self.mpole_expn_class = H2DMultipoleExpansion

    def get_bessel_arg_scaling(self):
        return sym.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)


class Y2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import YukawaKernel
        assert (isinstance(kernel.get_base_kernel(), YukawaKernel)
                and kernel.dim == 2)

        super(Y2DLocalExpansion, self).__init__(kernel, order, use_rscale)

        from sumpy.expansion.multipole import Y2DMultipoleExpansion
        self.mpole_expn_class = Y2DMultipoleExpansion

    def get_bessel_arg_scaling(self):
        return sym.I * sym.Symbol(self.kernel.get_base_kernel().yukawa_lambda_name)

# }}}

# vim: fdm=marker
