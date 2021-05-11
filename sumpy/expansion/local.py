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

import sumpy.symbolic as sym
from sumpy.tools import add_to_sac

from sumpy.expansion import (
    ExpansionBase, VolumeTaylorExpansion, LaplaceConformingVolumeTaylorExpansion,
    HelmholtzConformingVolumeTaylorExpansion,
    BiharmonicConformingVolumeTaylorExpansion)

from sumpy.tools import mi_increment_axis
from pytools import single_valued


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
            from sumpy.tools import ExprDerivativeTaker, my_syntactic_subs
            deriv_taker = ExprDerivativeTaker(line_kernel, (tau,), sac=sac, rscale=1)

            return [my_syntactic_subs(
                        kernel.postprocess_at_source(
                            deriv_taker.diff(i),
                            avec),
                    {tau: 0})
                    for i in self.get_coefficient_identifiers()]
        else:
            # Workaround for sympy. The automatic distribution after
            # single-variable diff makes the expressions very large
            # (https://github.com/sympy/sympy/issues/4596), so avoid doing
            # single variable diff.
            #
            # See also https://gitlab.tiker.net/inducer/pytential/merge_requests/12

            return [kernel.postprocess_at_source(
                            line_kernel.diff("tau", i), avec)
                    .subs("tau", 0)
                    for i in self.get_coefficient_identifiers()]

    def evaluate(self, tgt_kernel, coeffs, bvec, rscale, sac=None):
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

        for knl, weight in zip(kernels, weights):
            taker = knl.postprocess_at_source(base_taker, avec)
            # Following is a hack to make sure cse works.
            if 1:
                def save_temp(x):
                    return add_to_sac(sac, weight * x)

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
        from sumpy.tools import mi_power, mi_factorial

        result = sum(
            coeff
            * mi_power(bvec_scaled, mi, evaluate=False)
            / mi_factorial(mi)
            for coeff, mi in zip(
                    evaluated_coeffs, self.get_full_coefficient_identifiers()))

        return kernel.postprocess_at_target(result, bvec)

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None, _fast_version=True):
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

            taker = src_expansion.kernel.get_derivative_taker(dvec=dvec, sac=sac,
                rscale=src_rscale)

            from sumpy.tools import add_mi

            # Calculate a elementwise maximum multi-index because the number
            # of multi-indices needed is much less than
            # gnitstam(src_order + tgt order) when PDE conforming expansions
            # are used. For full Taylor, there's no difference.

            def calc_max_mi(mis):
                return (max(mi[i] for mi in mis) for i in range(self.dim))

            src_max_mi = calc_max_mi(src_expansion.get_coefficient_identifiers())
            tgt_max_mi = calc_max_mi(self.get_coefficient_identifiers())
            max_mi = add_mi(src_max_mi, tgt_max_mi)

            # Create a expansion terms wrangler for derivatives up to order
            # (tgt order)+(src order) including a corresponding reduction matrix
            tgtplusderiv_exp_terms_wrangler = \
                src_expansion.expansion_terms_wrangler.copy(
                        order=self.order + src_expansion.order, max_mi=tuple(max_mi))
            tgtplusderiv_coeff_ids = \
                tgtplusderiv_exp_terms_wrangler.get_coefficient_identifiers()
            tgtplusderiv_full_coeff_ids = \
                tgtplusderiv_exp_terms_wrangler.get_full_coefficient_identifiers()

            tgtplusderiv_ident_to_index = {ident: i for i, ident in
                                enumerate(tgtplusderiv_full_coeff_ids)}

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
                                embedded_coeffs, src_rscale, sac=sac)

                # Sum the embedded coefficient set
                for tgtplusderiv_coeff_id, coeff in zip(tgtplusderiv_coeff_ids,
                                                        stored_coeffs):
                    if coeff == 0:
                        continue
                    kernel_deriv = taker.diff(tgtplusderiv_coeff_id)
                    rscale_ratio = tgt_rscale / src_rscale
                    lexp_mi_terms.append(
                            coeff * kernel_deriv * rscale_ratio**sum(lexp_mi))
                result.append(sym.Add(*lexp_mi_terms))
            logger.info("building translation operator: done")
            return result

        rscale_ratio = sym.UnevaluatedExpr(tgt_rscale/src_rscale)

        from math import factorial
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
            dims = [axis] + list(range(axis)) + \
                    list(range(axis+1, self.dim))
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
                                / factorial(q)
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
            from sumpy.tools import ExprDerivativeTaker
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

    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        hankel_1 = sym.Function("hankel_1")

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        avec_len = sym_real_norm_2(avec)
        return [kernel.postprocess_at_source(
                    hankel_1(c, arg_scale * avec_len)
                    * rscale ** abs(c)
                    * sym.exp(sym.I * c * source_angle_rel_center), avec)
                    for c in self.get_coefficient_identifiers()]

    def evaluate(self, kernel, coeffs, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2
        bessel_j = sym.Function("bessel_j")
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(c)]
                   * kernel.postprocess_at_target(
                       bessel_j(c, arg_scale * bvec_len)
                       / rscale ** abs(c)
                       * sym.exp(sym.I * c * -target_angle_rel_center), bvec)
                for c in self.get_coefficient_identifiers())

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None):
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
                        sym.Integer(-1) ** j
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

        super().__init__(kernel, order, use_rscale)

        from sumpy.expansion.multipole import H2DMultipoleExpansion
        self.mpole_expn_class = H2DMultipoleExpansion

    def get_bessel_arg_scaling(self):
        return sym.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)


class Y2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import YukawaKernel
        assert (isinstance(kernel.get_base_kernel(), YukawaKernel)
                and kernel.dim == 2)

        super().__init__(kernel, order, use_rscale)

        from sumpy.expansion.multipole import Y2DMultipoleExpansion
        self.mpole_expn_class = Y2DMultipoleExpansion

    def get_bessel_arg_scaling(self):
        return sym.I * sym.Symbol(self.kernel.get_base_kernel().yukawa_lambda_name)

# }}}

# vim: fdm=marker
