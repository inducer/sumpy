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
    HelmholtzConformingVolumeTaylorExpansion)

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

        if isinstance(kernel, DirectionalSourceDerivative):
            if kernel.get_base_kernel() is not kernel.inner_kernel:
                raise NotImplementedError("more than one source derivative "
                        "not supported at present")

            from sumpy.symbolic import make_sym_vector
            dir_vec = make_sym_vector(kernel.dir_vec_name, kernel.dim)

            coeff_identifiers = self.get_full_coefficient_identifiers()
            result = [0] * len(coeff_identifiers)

            for idim in range(kernel.dim):
                for i, mi in enumerate(coeff_identifiers):
                    if mi[idim] == 0:
                        continue

                    derivative_mi = tuple(mi_i - 1 if iaxis == idim else mi_i
                            for iaxis, mi_i in enumerate(mi))

                    result[i] += (
                        - mi_power(avec, derivative_mi) * mi[idim]
                        * dir_vec[idim])
            for i, mi in enumerate(coeff_identifiers):
                result[i] /= (mi_factorial(mi) * rscale ** sum(mi))
        else:
            avec = avec * rscale**-1

            result = [
                    mi_power(avec, mi) / mi_factorial(mi)
                    for mi in self.get_full_coefficient_identifiers()]
        return (
            self.derivative_wrangler.get_stored_mpole_coefficients_from_full(
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

    def evaluate(self, coeffs, bvec, rscale):
        if not self.use_rscale:
            rscale = 1

        taker = self.get_kernel_derivative_taker(bvec)

        result = sym.Add(*tuple(
                coeff
                * self.get_scaled_multipole(taker.diff(mi), bvec, rscale, sum(mi))
                for coeff, mi in zip(coeffs, self.get_coefficient_identifiers())))

        return result

    def get_kernel_derivative_taker(self, bvec):
        return (self.derivative_wrangler.get_derivative_taker(
            self.kernel.get_expression(bvec), bvec))

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
            src_expansion.get_coefficient_identifiers()))

        for i, mi in enumerate(src_expansion.get_coefficient_identifiers()):
            src_coeff_exprs[i] *= mi_factorial(mi)

        result = [0] * len(self.get_full_coefficient_identifiers())
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
                            * dvec[idim]**(n-k))

                result[i] += (
                        contrib
                        * (src_rscale**sum(src_mi) / tgt_rscale**sum(tgt_mi)))

            result[i] /= mi_factorial(tgt_mi)

        logger.info("building translation operator: done")
        return (
            self.derivative_wrangler.get_stored_mpole_coefficients_from_full(
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

    def evaluate(self, coeffs, bvec, rscale):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        hankel_1 = sym.Function("hankel_1")
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(l)]
                   * self.kernel.postprocess_at_target(
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
