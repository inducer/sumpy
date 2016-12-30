from __future__ import division, absolute_import
from six.moves import range, zip

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

import sympy as sp

from sumpy.expansion import (
    ExpansionBase, VolumeTaylorExpansion, LaplaceConformingVolumeTaylorExpansion,
    HelmholtzConformingVolumeTaylorExpansion)


class LocalExpansionBase(ExpansionBase):
    pass


import logging
logger = logging.getLogger(__name__)

__doc__ = """

.. autoclass:: VolumeTaylorLocalExpansion
.. autoclass:: H2DLocalExpansion
.. autoclass:: LineTaylorLocalExpansion

"""


# {{{ line taylor

class LineTaylorLocalExpansion(LocalExpansionBase):

    def get_storage_index(self, k):
        return k

    def get_coefficient_identifiers(self):
        return list(range(self.order+1))

    def coefficients_from_source(self, avec, bvec):
        if bvec is None:
            raise RuntimeError("cannot use line-Taylor expansions in a setting "
                    "where the center-target vector is not known at coefficient "
                    "formation")
        avec_line = avec + sp.Symbol("tau")*bvec

        line_kernel = self.kernel.get_expression(avec_line)

        return [
                self.kernel.postprocess_at_target(
                    self.kernel.postprocess_at_source(
                        line_kernel.diff("tau", i),
                        avec),
                    bvec)
                .subs("tau", 0)
                for i in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec):
        from pytools import factorial
        return sum(
                coeffs[self.get_storage_index(i)] / factorial(i)
                for i in self.get_coefficient_identifiers())

# }}}


# {{{ volume taylor

class VolumeTaylorLocalExpansionBase(LocalExpansionBase):
    """
    Coefficients represent derivative values of the kernel.
    """

    def coefficients_from_source(self, avec, bvec):
        from sumpy.tools import MiDerivativeTaker
        ppkernel = self.kernel.postprocess_at_source(
                self.kernel.get_expression(avec), avec)
        taker = MiDerivativeTaker(ppkernel, avec)
        return [taker.diff(mi) for mi in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec):
        from sumpy.tools import mi_power, mi_factorial
        evaluated_coeffs = self.stored_to_full(coeffs)
        result = sum(
                coeff
                * self.kernel.postprocess_at_target(mi_power(bvec, mi), bvec)
                / mi_factorial(mi)
                for coeff, mi in zip(
                        evaluated_coeffs, self.get_full_coefficient_identifiers()))
        return result

    def translate_from(self, src_expansion, src_coeff_exprs, dvec):
        logger.info("building translation operator: %s(%d) -> %s(%d): start"
                % (type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order))

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

            def mi_sum(a, b):
                return tuple(aval + bval for aval, bval in zip(a, b))

            result = []
            for deriv in self.get_coefficient_identifiers():
                local_result = []
                for coeff, term in zip(
                        src_coeff_exprs,
                        src_expansion.get_coefficient_identifiers()):
                    kernel_deriv = taker.diff(mi_sum(deriv, term))
                    local_result.append(coeff * kernel_deriv)
                result.append(sp.Add(*local_result))
        else:
            from sumpy.tools import MiDerivativeTaker
            expr = src_expansion.evaluate(src_coeff_exprs, dvec)
            taker = MiDerivativeTaker(expr, dvec)
            result = [taker.diff(mi) for mi in self.get_coefficient_identifiers()]

        logger.info("building translation operator: done")
        return result


class VolumeTaylorLocalExpansion(
        VolumeTaylorLocalExpansionBase,
        VolumeTaylorExpansion):

    def __init__(self, kernel, order):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order)
        VolumeTaylorExpansion.__init__(self)


class LaplaceConformingVolumeTaylorLocalExpansion(
        VolumeTaylorLocalExpansionBase,
        LaplaceConformingVolumeTaylorExpansion):

    def __init__(self, kernel, order):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order)
        LaplaceConformingVolumeTaylorExpansion.__init__(self)


class HelmholtzConformingVolumeTaylorLocalExpansion(
        VolumeTaylorLocalExpansionBase,
        HelmholtzConformingVolumeTaylorExpansion):

    def __init__(self, kernel, order):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order)
        HelmholtzConformingVolumeTaylorExpansion.__init__(self)

# }}}


# {{{ 2D J-expansion

class H2DLocalExpansion(LocalExpansionBase):
    def __init__(self, kernel, order):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dim == 2)

        LocalExpansionBase.__init__(self, kernel, order)

    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_identifiers(self):
        return list(range(-self.order, self.order+1))

    def coefficients_from_source(self, avec, bvec):
        from sumpy.symbolic import sympy_real_norm_2
        hankel_1 = sp.Function("hankel_1")

        k = sp.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sp.atan2(-avec[1], -avec[0])
        avec_len = sympy_real_norm_2(avec)
        return [self.kernel.postprocess_at_source(
                    hankel_1(l, k * avec_len)
                    * sp.exp(sp.I * l * source_angle_rel_center), avec)
                    for l in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec):
        from sumpy.symbolic import sympy_real_norm_2
        bessel_j = sp.Function("bessel_j")
        bvec_len = sympy_real_norm_2(bvec)
        target_angle_rel_center = sp.atan2(bvec[1], bvec[0])

        k = sp.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)

        return sum(coeffs[self.get_storage_index(l)]
                   * self.kernel.postprocess_at_target(
                       bessel_j(l, k * bvec_len)
                       * sp.exp(sp.I * l * -target_angle_rel_center), bvec)
                for l in self.get_coefficient_identifiers())

    def translate_from(self, src_expansion, src_coeff_exprs, dvec):
        from sumpy.symbolic import sympy_real_norm_2

        k = sp.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)

        if isinstance(src_expansion, H2DLocalExpansion):
            dvec_len = sympy_real_norm_2(dvec)
            bessel_j = sp.Function("bessel_j")
            new_center_angle_rel_old_center = sp.atan2(dvec[1], dvec[0])
            translated_coeffs = []
            for l in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum(src_coeff_exprs[src_expansion.get_storage_index(m)]
                        * bessel_j(m - l, k * dvec_len)
                        * sp.exp(sp.I * (m - l) * -new_center_angle_rel_old_center)
                    for m in src_expansion.get_coefficient_identifiers()))
            return translated_coeffs

        from sumpy.expansion.multipole import H2DMultipoleExpansion
        if isinstance(src_expansion, H2DMultipoleExpansion):
            dvec_len = sympy_real_norm_2(dvec)
            hankel_1 = sp.Function("hankel_1")
            new_center_angle_rel_old_center = sp.atan2(dvec[1], dvec[0])
            translated_coeffs = []
            for l in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum((-1) ** l * hankel_1(m + l, k * dvec_len)
                        * sp.exp(sp.I * (m + l) * new_center_angle_rel_old_center)
                        * src_coeff_exprs[src_expansion.get_storage_index(m)]
                    for m in src_expansion.get_coefficient_identifiers()))
            return translated_coeffs

        raise RuntimeError("do not know how to translate %s to "
                           "local 2D Helmholtz Bessel expansion"
                           % type(src_expansion).__name__)

# }}}

# vim: fdm=marker
