from __future__ import division

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

import sympy as sp  # noqa

from sumpy.expansion import ExpansionBase, VolumeTaylorExpansionBase

import logging
logger = logging.getLogger(__name__)


class MultipoleExpansionBase(ExpansionBase):
    pass


# {{{ volume taylor

class VolumeTaylorMultipoleExpansion(
        MultipoleExpansionBase, VolumeTaylorExpansionBase):
    def coefficients_from_source(self, avec, bvec):
        from sumpy.kernel import DirectionalSourceDerivative
        kernel = self.kernel

        from sumpy.tools import mi_power

        if isinstance(kernel, DirectionalSourceDerivative):
            if kernel.get_base_kernel() is not kernel.inner_kernel:
                raise NotImplementedError("more than one source derivative "
                        "not supported at present")

            from sumpy.symbolic import make_sympy_vector
            dir_vec = make_sympy_vector(kernel.dir_vec_name, kernel.dim)

            coeff_identifiers = self.get_coefficient_identifiers()
            result = [0] * len(coeff_identifiers)

            for idim in xrange(kernel.dim):
                for i, mi in enumerate(coeff_identifiers):
                    if mi[idim] == 0:
                        continue

                    derivative_mi = tuple(mi_i - 1 if iaxis == idim else mi_i
                            for iaxis, mi_i in enumerate(mi))

                    result[i] += (
                        - mi_power(avec, derivative_mi) * mi[idim]
                        * dir_vec[idim])

            return result
        else:
            return [
                    mi_power(avec, mi)
                    for mi in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec):
        ppkernel = self.kernel.postprocess_at_target(
                self.kernel.get_expression(bvec), bvec)

        from sumpy.tools import mi_derivative, mi_factorial
        return sum(
                coeff / mi_factorial(mi)
                * mi_derivative(ppkernel, bvec, mi)
                for coeff, mi in zip(coeffs, self.get_coefficient_identifiers()))

    def translate_from(self, src_expansion, src_coeff_exprs, dvec):
        if not isinstance(src_expansion, type(self)):
            raise RuntimeError("do not know how to translate %s to "
                    "Taylor multipole expansion"
                               % type(src_expansion).__name__)

        logger.info("building translation operator: %s(%d) -> %s(%d): start"
                % (type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order))

        src_mi_to_index = dict((mi, i) for i, mi in enumerate(
            src_expansion.get_coefficient_identifiers()))

        result = [0] * len(self.get_coefficient_identifiers())
        from pytools import generate_nonnegative_integer_tuples_below as gnitb

        for i, tgt_mi in enumerate(
                self.get_coefficient_identifiers()):

            tgt_mi_plus_one = tuple(mi_i + 1 for mi_i in tgt_mi)

            for src_mi in gnitb(tgt_mi_plus_one):
                try:
                    src_index = src_mi_to_index[src_mi]
                except KeyError:
                    # tgt higher-order than source--dumb, but not life-threatening
                    continue

                contrib = src_coeff_exprs[src_index]

                for idim in xrange(self.dim):
                    n = tgt_mi[idim]
                    k = src_mi[idim]
                    assert n >= k
                    contrib *= (sp.binomial(n, k)
                            * dvec[idim]**(n-k))

                result[i] += contrib

        logger.info("building translation operator: done")
        return result

# }}}


# {{{ 2D J-expansion

class H2DMultipoleExpansion(MultipoleExpansionBase):
    def __init__(self, kernel, order):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dim == 2)

        MultipoleExpansionBase.__init__(self, kernel, order)

    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_identifiers(self):
        return range(-self.order, self.order+1)

    def coefficients_from_source(self, avec, bvec):
        from sumpy.symbolic import sympy_real_norm_2
        bessel_j = sp.Function("bessel_j")
        avec_len = sympy_real_norm_2(avec)
        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sp.atan2(-avec[1], -avec[0])
        # FIXME: Figure out why compiling some kernels fails without the cast
        # to complex values.
        return [sp.Function("complex")(self.kernel.postprocess_at_source(
                    bessel_j(l, sp.Symbol("k") * avec_len) *
                    sp.exp(sp.I * l * -source_angle_rel_center), avec))
                for l in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec):
        from sumpy.symbolic import sympy_real_norm_2
        hankel_1 = sp.Function("hankel_1")
        bvec_len = sympy_real_norm_2(bvec)
        target_angle_rel_center = sp.atan2(bvec[1], bvec[0])
        return sum(coeffs[self.get_storage_index(l)]
                   * self.kernel.postprocess_at_target(
                       hankel_1(l, sp.Symbol("k") * bvec_len)
                       * sp.exp(sp.I * l * target_angle_rel_center), bvec)
                for l in self.get_coefficient_identifiers())

    def translate_from(self, src_expansion, src_coeff_exprs, dvec):
        if not isinstance(src_expansion, type(self)):
            raise RuntimeError("do not know how to translate %s to "
                               "multipole 2D Helmholtz Bessel expansion"
                               % type(src_expansion).__name__)
        from sumpy.symbolic import sympy_real_norm_2
        dvec_len = sympy_real_norm_2(dvec)
        bessel_j = sp.Function("bessel_j")
        new_center_angle_rel_old_center = sp.atan2(dvec[1], dvec[0])
        translated_coeffs = []
        for l in self.get_coefficient_identifiers():
            translated_coeffs.append(
                sum(src_coeff_exprs[src_expansion.get_storage_index(m)]
                    * bessel_j(m - l, sp.Symbol("k") * dvec_len)
                    * sp.exp(sp.I * (m - l) * new_center_angle_rel_old_center)
                for m in src_expansion.get_coefficient_identifiers()))
        return translated_coeffs

# }}}

# vim: fdm=marker
