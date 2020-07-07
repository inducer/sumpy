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
from pytools import single_valued

from sumpy.expansion import (
    ExpansionBase, VolumeTaylorExpansion, LaplaceConformingVolumeTaylorExpansion,
    HelmholtzConformingVolumeTaylorExpansion,
    BiharmonicConformingVolumeTaylorExpansion)

from sumpy.tools import (matvec_toeplitz_upper_triangular,
    fft_toeplitz_upper_triangular, add_to_sac, fft)


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

    def coefficients_from_source(self, avec, bvec, rscale, sac):
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

    def evaluate(self, coeffs, bvec, rscale, sac, knl=None):
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

    def coefficients_from_source(self, avec, bvec, rscale, sac):
        from sumpy.tools import MiDerivativeTakerWrapper

        result = []
        taker = self.get_kernel_derivative_taker(avec, rscale, sac)
        expr_dict = {(0,)*self.dim: 1}
        expr_dict = self.kernel.get_derivative_transformation_at_source(expr_dict)
        pp_nderivatives = single_valued(sum(mi) for mi in expr_dict.keys())

        for mi in self.get_coefficient_identifiers():
            wrapper = MiDerivativeTakerWrapper(taker, mi)
            mi_expr = self.kernel.postprocess_at_source(wrapper, avec)
            # By passing `rscale` to the derivative taker we are taking a scaled
            # version of the derivative which is `expr.diff(mi)*rscale**sum(mi)`
            # which might be implemented efficiently for kernels like Laplace.
            # One caveat is that `postprocess_at_source` might take more
            # derivatives which would multiply the expression by more `rscale`s
            # than necessary as the derivative taker does not know about
            # `postprocess_at_source`. This is corrected by dividing by `rscale`.
            expr = mi_expr / rscale ** pp_nderivatives
            result.append(expr)

        return result

    def evaluate(self, coeffs, bvec, rscale, sac, knl=None):
        evaluated_coeffs = (
            self.expansion_terms_wrangler.get_full_kernel_derivatives_from_stored(
                coeffs, rscale))

        bvec_scaled = [b*rscale**-1 for b in bvec]
        from sumpy.tools import mi_power, mi_factorial

        result = sum(
            coeff
            * mi_power(bvec_scaled, mi, evaluate=False)
            / mi_factorial(mi)
            for coeff, mi in zip(
                    evaluated_coeffs, self.get_full_coefficient_identifiers()))

        if knl is None:
            knl = self.kernel
        return knl.postprocess_at_target(result, bvec)

    def m2l_global_precompute_nexpr(self, src_expansion, use_fft=False):
        from sumpy.tools import fft_toeplitz_upper_triangular_lwork
        if use_fft:
            nexpr = len(self._m2l_global_precompute_mis(src_expansion)[0])
            nexpr = fft_toeplitz_upper_triangular_lwork(nexpr)
        else:
            nexpr = len(self._m2l_global_precompute_mis(src_expansion)[1])
        return nexpr

    def _m2l_global_precompute_mis(self, src_expansion, use_fft=False):
        from pytools import generate_nonnegative_integer_tuples_below as gnitb
        from sumpy.tools import add_mi

        max_mi = [0]*self.dim
        for i in range(self.dim):
            max_mi[i] = max(mi[i] for mi in
                              src_expansion.get_coefficient_identifiers())
            max_mi[i] += max(mi[i] for mi in
                              self.get_coefficient_identifiers())

        toeplitz_matrix_coeffs = list(gnitb([m + 1 for m in max_mi]))

        needed_vector_terms = []
        # For eg: 2D full Taylor Laplace, we only need kernel derivatives
        # (n1+n2, m1+m2), n1+m1<=p, n2+m2<=p
        for tgt_deriv in self.get_coefficient_identifiers():
            for src_deriv in src_expansion.get_coefficient_identifiers():
                needed = add_mi(src_deriv, tgt_deriv)
                if needed not in needed_vector_terms:
                    needed_vector_terms.append(needed)

        return toeplitz_matrix_coeffs, needed_vector_terms, max_mi

    def _fft(self, x, sac):
        return fft(x, sac=sac)

    def m2l_global_precompute_exprs(self, src_expansion, src_rscale,
            dvec, tgt_rscale, sac, use_fft=False):
        # We know the general form of the multipole expansion is:
        #
        #  coeff0 * diff(kernel(src - c1), mi0) +
        #    coeff1 * diff(kernel(src - c1, mi1) + ...
        #
        # To get the local expansion coefficients, we take derivatives of
        # the multipole expansion. For eg: the coefficient w.r.t mir is
        #
        #  coeff0 * diff(kernel(c2 - c1), mi0 + mir) +
        #    coeff1 * diff(kernel(c2 - c1, mi1 + mir) + ...
        #
        # The derivatives above depends only on `c2 - c1` and can be precomputed
        # globally as there are only a finite number of values for `c2 - c1` for
        # m2l.

        if not self.use_rscale:
            src_rscale = 1

        toeplitz_matrix_coeffs, needed_vector_terms, max_mi = \
            self._m2l_global_precompute_mis(src_expansion)

        toeplitz_matrix_ident_to_index = dict((ident, i) for i, ident in
                                enumerate(toeplitz_matrix_coeffs))

        # Create a expansion terms wrangler for derivatives up to order
        # (tgt order)+(src order) including a corresponding reduction matrix
        # For eg: 2D full Taylor Laplace, this is (n, m),
        # n+m<=2*p, n<=2*p, m<=2*p
        srcplusderiv_terms_wrangler = \
            src_expansion.expansion_terms_wrangler.copy(
                    order=self.order + src_expansion.order, max_mi=tuple(max_mi))
        srcplusderiv_full_coeff_ids = \
            srcplusderiv_terms_wrangler.get_full_coefficient_identifiers()
        srcplusderiv_ident_to_index = dict((ident, i) for i, ident in
                            enumerate(srcplusderiv_full_coeff_ids))

        # The vector has the kernel derivatives and depends only on the distance
        # between the two centers
        taker = src_expansion.get_kernel_derivative_taker(dvec, src_rscale, sac)
        vector_stored = []
        # Calculate the kernel derivatives for the compressed set
        for term in \
                srcplusderiv_terms_wrangler.get_coefficient_identifiers():
            kernel_deriv = taker.diff(term)
            vector_stored.append(kernel_deriv)
        # Calculate the kernel derivatives for the full set
        vector_full = \
            srcplusderiv_terms_wrangler.get_full_kernel_derivatives_from_stored(
                        vector_stored, src_rscale)

        for term in srcplusderiv_full_coeff_ids:
            assert term in needed_vector_terms

        vector = [0]*len(needed_vector_terms)
        for i, term in enumerate(needed_vector_terms):
            vector[i] = add_to_sac(sac,
                        vector_full[srcplusderiv_ident_to_index[term]])

        if use_fft:
            # Add zero values needed to make the translation matrix toeplitz
            derivatives_full = [0]*len(toeplitz_matrix_coeffs)
            for expr, mi in zip(vector, needed_vector_terms):
                derivatives_full[toeplitz_matrix_ident_to_index[mi]] = expr
            return self._fft(list(reversed(derivatives_full)), sac=sac)

        return vector

    def m2l_preprocess_exprs(self, src_expansion, src_coeff_exprs, sac,
            src_rscale, use_fft=False):
        toeplitz_matrix_coeffs, needed_vector_terms, max_mi = \
                self._m2l_global_precompute_mis(src_expansion)
        toeplitz_matrix_ident_to_index = dict((ident, i) for i, ident in
                            enumerate(toeplitz_matrix_coeffs))

        # Calculate the first row of the upper triangular Toeplitz matrix
        toeplitz_first_row = [0] * len(toeplitz_matrix_coeffs)
        for coeff, term in zip(
                src_coeff_exprs,
                src_expansion.get_coefficient_identifiers()):
            toeplitz_first_row[toeplitz_matrix_ident_to_index[term]] = \
                    add_to_sac(sac, coeff)

        if use_fft:
            return self._fft(toeplitz_first_row, sac=sac)
        else:
            return toeplitz_first_row

    def m2l_postprocess_exprs(self, src_expansion, m2l_result, src_rscale,
            tgt_rscale, sac, use_fft=False):
        toeplitz_matrix_coeffs, needed_vector_terms, max_mi = \
                self._m2l_global_precompute_mis(src_expansion)
        toeplitz_matrix_ident_to_index = dict((ident, i) for i, ident in
                            enumerate(toeplitz_matrix_coeffs))

        if use_fft:
            n = len(toeplitz_matrix_coeffs)
            m2l_result = fft(m2l_result, inverse=True, sac=sac)
            m2l_result = list(reversed(m2l_result[:n]))

        # Filter out the dummy rows and scale them for target
        result = []
        rscale_ratio = add_to_sac(sac, tgt_rscale/src_rscale)
        for term in self.get_coefficient_identifiers():
            index = toeplitz_matrix_ident_to_index[term]
            result.append(m2l_result[index]*rscale_ratio**sum(term))

        return result

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac, precomputed_exprs=None, use_fft=False):
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
            toeplitz_matrix_coeffs, needed_vector_terms, max_mi = \
                self._m2l_global_precompute_mis(src_expansion)
            toeplitz_matrix_ident_to_index = dict((ident, i) for i, ident in
                                enumerate(toeplitz_matrix_coeffs))

            if precomputed_exprs is None:
                derivatives = self.m2l_global_precompute_exprs(src_expansion,
                        src_rscale, dvec, tgt_rscale, sac, use_fft=use_fft)
            else:
                derivatives = precomputed_exprs

            if use_fft:
                assert precomputed_exprs is not None
                assert len(src_coeff_exprs) == len(precomputed_exprs)
                result = []
                for i in range(len(precomputed_exprs)):
                    a = precomputed_exprs[i]
                    b = src_coeff_exprs[i]
                    result.append(a*b)
                return result

            derivatives_full = [0]*len(toeplitz_matrix_coeffs)
            for expr, mi in zip(derivatives, needed_vector_terms):
                derivatives_full[toeplitz_matrix_ident_to_index[mi]] = expr

            toeplitz_first_row = self.m2l_preprocess_exprs(src_expansion,
                src_coeff_exprs, sac, use_fft)

            # Do the matvec
            if 0:
                output = fft_toeplitz_upper_triangular(toeplitz_first_row,
                                derivatives_full, sac=sac)
            else:
                output = matvec_toeplitz_upper_triangular(toeplitz_first_row,
                                derivatives_full)

            result = self.m2l_postprocess_exprs(src_expansion, output, src_rscale,
                tgt_rscale, sac, use_fft)

            logger.info("building translation operator: done")
            return result

        rscale_ratio = tgt_rscale/src_rscale
        rscale_ratio = add_to_sac(sac, rscale_ratio)

        from sumpy.tools import MiDerivativeTaker
        from math import factorial
        src_wrangler = src_expansion.expansion_terms_wrangler
        src_coeffs = (
            src_wrangler.get_full_kernel_derivatives_from_stored(
                src_coeff_exprs, src_rscale))
        src_mis = \
            src_expansion.expansion_terms_wrangler.get_full_coefficient_identifiers()

        src_mi_to_index = dict((mi, i) for i, mi in enumerate(src_mis))

        tgt_mis_all = \
            self.expansion_terms_wrangler.get_coefficient_identifiers()
        tgt_mi_to_index = dict((mi, i) for i, mi in enumerate(tgt_mis_all))

        tgt_split = self.expansion_terms_wrangler._get_coeff_identifier_split()

        p = max(sum(mi) for mi in src_mis)
        result = [0] * len(tgt_mis_all)

        # O(1) iterations
        for const_dim in set(d for d, _ in tgt_split):
            # Use the const_dim as the first dimension to vary so that the below
            # algorithm is O(p^{d+1}) for full and O(p^{d}) for compressed
            dims = [const_dim] + list(range(const_dim)) + \
                    list(range(const_dim+1, self.dim))
            # Start with source coefficients
            Y = src_coeffs   # noqa: N806
            # O(1) iterations
            for d in dims:
                C = Y        # noqa: N806
                Y = [0] * len(src_mis)   # noqa: N806
                # Only O(p^{d-1}) operations are used in compressed
                # All of them are used in full.
                for i, s in enumerate(src_mis):
                    # O(p) iterations
                    for q in range(p+1-sum(s)):
                        src_mi = list(s)
                        src_mi[d] += q
                        src_mi = tuple(src_mi)
                        if src_mi in src_mi_to_index:
                            Y[i] += (dvec[d]/src_rscale) ** q * \
                                    C[src_mi_to_index[src_mi]] / factorial(q)

            # This is O(p) in full and O(1) in compressed
            for d, tgt_mis in tgt_split:
                if d != const_dim:
                    continue
                # O(p^{d-1}) iterations
                for mi in tgt_mis:
                    if mi not in src_mi_to_index:
                        continue
                    result[tgt_mi_to_index[mi]] = Y[src_mi_to_index[mi]] \
                                                    * rscale_ratio ** sum(mi)

        # {{{ simpler, functionally equivalent code
        if 0:
            # Rscale/operand magnitude is fairly sensitive to the order of
            # operations--which is something we don't have fantastic control
            # over at the symbolic level. Scaling dvec, then differentiating,
            # and finally rescaling dvec leaves the expression needing a scaling
            # to compensate for differentiating which is done at the end.
            # This moves the two cancelling "rscales" closer to each other at
            # the end in the hope of helping rscale magnitude.
            dvec_scaled = [d*src_rscale for d in dvec]
            expr = src_expansion.evaluate(src_coeff_exprs, dvec_scaled,
                        rscale=src_rscale, sac=sac)
            replace_dict = dict((d, d/src_rscale) for d in dvec)
            taker = MiDerivativeTaker(expr, dvec)
            rscale_ratio = sym.UnevaluatedExpr(tgt_rscale/src_rscale)
            result = [
                    (taker.diff(mi).xreplace(replace_dict) * rscale_ratio**sum(mi))
                    for mi in self.get_coefficient_identifiers()]
        # }}}
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

    def coefficients_from_source(self, avec, bvec, rscale, sac):
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

    def evaluate(self, coeffs, bvec, rscale, sac, knl=None):
        if not self.use_rscale:
            rscale = 1
        if knl is None:
            knl = self.kernel

        from sumpy.symbolic import sym_real_norm_2
        bessel_j = sym.Function("bessel_j")
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(c)]
                   * knl.postprocess_at_target(
                       bessel_j(c, arg_scale * bvec_len)
                       / rscale ** abs(c)
                       * sym.exp(sym.I * c * -target_angle_rel_center), bvec)
                for c in self.get_coefficient_identifiers())

    def m2l_global_precompute_nexpr(self, src_expansion, use_fft=False):
        from sumpy.tools import fft_toeplitz_upper_triangular_lwork
        nexpr = 4 * self.order + 1
        if use_fft:
            nexpr = fft_toeplitz_upper_triangular_lwork(nexpr)
        print("nexpr", nexpr, 4 * self.order + 1)
        return nexpr

    def m2l_global_precompute_exprs(self, src_expansion, src_rscale,
            dvec, tgt_rscale, sac, use_fft=False):

        from sumpy.symbolic import sym_real_norm_2
        from sumpy.tools import fft

        dvec_len = sym_real_norm_2(dvec)
        hankel_1 = sym.Function("hankel_1")
        new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])
        arg_scale = self.get_bessel_arg_scaling()
        assert self.order == src_expansion.order
        precomputed_exprs = [0] * (4*self.order + 1)
        for j in self.get_coefficient_identifiers():
            for m in self.get_coefficient_identifiers():
                precomputed_exprs[m + j + 2 * self.order] = (
                    hankel_1(m + j, arg_scale * dvec_len)
                    * sym.exp(sym.I * (m + j) * new_center_angle_rel_old_center))

        if use_fft:
            order = self.order
            first, last = precomputed_exprs[:2*order], precomputed_exprs[2*order:]
            return fft(list(last)+list(first), sac)

        return precomputed_exprs

    def m2l_preprocess_exprs(self, src_expansion, src_coeff_exprs, sac,
            src_rscale, use_fft=False):

        from sumpy.tools import fft
        src_coeff_exprs = list(src_coeff_exprs)
        for m in src_expansion.get_coefficient_identifiers():
            src_coeff_exprs[src_expansion.get_storage_index(m)] *= src_rscale**abs(m)

        if use_fft:
            src_coeff_exprs = list(reversed(src_coeff_exprs))
            src_coeff_exprs += [0] * (len(src_coeff_exprs) - 1)
            res = fft(src_coeff_exprs, sac=sac)
            return res
        else:
            return src_coeff_exprs

    def m2l_postprocess_exprs(self, src_expansion, m2l_result, src_rscale,
            tgt_rscale, sac, use_fft=False):

        if use_fft:
            n = 2 * self.order + 1
            m2l_result = fft(m2l_result, inverse=True, sac=sac)
            m2l_result = m2l_result[:2*self.order+1]

        # Filter out the dummy rows and scale them for target
        result = []
        for j in self.get_coefficient_identifiers():
            result.append(m2l_result[j + self.order] * tgt_rscale**(abs(j)) * (-1)**j)

        return result

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac, precomputed_exprs=None, use_fft=False):
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
            if precomputed_exprs is None:
                derivatives = self.m2l_global_precompute_exprs(src_expansion,
                    src_rscale, dvec, tgt_rscale, sac, use_fft)
            else:
                derivatives = precomputed_exprs

            translated_coeffs = []
            if use_fft:
                assert precomputed_exprs is not None
                assert len(derivatives) == len(src_coeff_exprs)
                for a, b in zip(derivatives, src_coeff_exprs):
                    translated_coeffs.append(a * b)
                return translated_coeffs

            src_coeff_exprs = self.m2l_preprocess_exprs(src_expansion, src_coeff_exprs, sac,
                src_rscale, use_fft=False)

            import numpy as np
            n = len(src_coeff_exprs)
            print(n, self.order)
            src = np.array(list(reversed(src_coeff_exprs))+[0]*(n-1), dtype=object)

            order = self.order
            first, last = precomputed_exprs[:2*order], precomputed_exprs[2*order:]
            derivatives_vec = np.array(list(last)+list(first), dtype=object)
            deriv_fft = fft(list(derivatives_vec), sac=None)
            src_fft = fft(list(src), sac=None)
            n = (len(derivatives_vec)+1)//2

            translated_coeffs = []
            assert len(deriv_fft) == len(src_fft)
            for a, b in zip(deriv_fft, src_fft):
                translated_coeffs.append(a * b)

            translated_coeffs = fft(translated_coeffs, sac=None, inverse=True)

            def simp(expr):
                #expr = expr.expand()
                rep = {}
                for i in expr.atoms(sym.Float):
                    j = i
                    if abs(complex(j))<1e-8:
                        j = 0
                    rep[i] = j
                rep[1.0] = 1
                expr = expr.xreplace(rep)
                return expr

            translated_coeffs = translated_coeffs[:n]
            for i in range(n):
                #print(translated_coeffs[i], simp(translated_coeffs[i]))
                translated_coeffs[i] = simp(translated_coeffs[i])

            for j in self.get_coefficient_identifiers():
                x1 = (
                      sum(derivatives[m + j + 2*self.order]
                        * src_coeff_exprs[src_expansion.get_storage_index(m)]
                        for m in src_expansion.get_coefficient_identifiers()))
                print("=================")
                x3 = translated_coeffs[j+self.order]
                if simp((x1-x3).expand()) != 0:
                    print((x1-x3).expand())
                    raise RuntimeError("")

            translated_coeffs = self.m2l_postprocess_exprs(src_expansion, translated_coeffs, src_rscale,
                tgt_rscale, sac, use_fft=False)
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
