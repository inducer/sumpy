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

import math
from abc import abstractmethod

import sumpy.symbolic as sym
from sumpy.expansion import (
    ExpansionBase,
    VolumeTaylorExpansion,
    VolumeTaylorExpansionMixin,
    LinearPDEConformingVolumeTaylorExpansion)
from sumpy.tools import mi_set_axis, add_to_sac, mi_power, mi_factorial

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

class VolumeTaylorMultipoleExpansionBase(
        VolumeTaylorExpansionMixin, MultipoleExpansionBase):
    """
    Coefficients represent the terms in front of the kernel derivatives.
    """

    def coefficients_from_source_vec(self, kernels, avec, bvec, rscale, weights,
            sac=None):
        """This method calculates the full coefficients, sums them up and
        compresses them. This is more efficient that calculating full
        coefficients, compressing and then summing.
        """
        from sumpy.kernel import KernelWrapper

        if not self.use_rscale:
            rscale = 1

        result = [0]*len(self.get_full_coefficient_identifiers())
        for kernel, weight in zip(kernels, weights):
            if isinstance(kernel, KernelWrapper):
                coeffs = [
                        kernel.postprocess_at_source(mi_power(avec, mi), avec)
                        / rscale ** sum(mi)
                        for mi in self.get_full_coefficient_identifiers()]
            else:
                avec_scaled = [sym.UnevaluatedExpr(a * rscale**-1) for a in avec]
                coeffs = [mi_power(avec_scaled, mi)
                          for mi in self.get_full_coefficient_identifiers()]

            for i, mi in enumerate(self.get_full_coefficient_identifiers()):
                result[i] += coeffs[i] * weight / mi_factorial(mi)
        return (
            self.expansion_terms_wrangler.get_stored_mpole_coefficients_from_full(
                result, rscale, sac=sac))

    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        return self.coefficients_from_source_vec((kernel,), avec, bvec,
                rscale, (1,), sac=sac)

    def evaluate(self, kernel, coeffs, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        base_taker = kernel.get_derivative_taker(bvec, rscale, sac)
        taker = kernel.postprocess_at_target(base_taker, bvec)

        result = []
        for coeff, mi in zip(coeffs, self.get_coefficient_identifiers()):
            result.append(coeff * taker.diff(mi, lambda x: add_to_sac(sac, x)))

        result = sym.Add(*tuple(result))
        return result

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None, _fast_version=True):
        if not isinstance(src_expansion, type(self)):
            raise RuntimeError(
                f"do not know how to translate {type(src_expansion).__name__} to "
                "a Taylor multipole expansion")

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        logger.info("building translation operator for %s: %s(%d) -> %s(%d): start",
                    src_expansion.kernel,
                    type(src_expansion).__name__,
                    src_expansion.order,
                    type(self).__name__,
                    self.order)

        src_mi_to_index = {mi: i for i, mi in enumerate(
            src_expansion.get_coefficient_identifiers())}

        tgt_mi_to_index = {mi: i for i, mi in enumerate(
            self.get_full_coefficient_identifiers())}

        # This algorithm uses the observation that M2M coefficients
        # have the following form in 2D
        #
        # $T_{m, n} = \sum_{i\le m, j\le n} C_{i, j}
        #             d_x^i d_y^j \binom{m}{i} \binom{n}{j}$
        # and can be rewritten as follows.
        #
        # Let $Y_{m, n} = \sum_{i\le m} C_{i, n} d_x^i \binom{m}{i}$.
        #
        # Then, $T_{m, n} = \sum_{j\le n} Y_{m, j} d_y^j \binom{n}{j}$.
        #
        # $Y_{m, n}$ are $p^2$ temporary variables that are
        # reused for different M2M coefficients and costs $p$ per variable.
        # Total cost for calculating $Y_{m, n}$ is $p^3$ and similar
        # for $T_{m, n}$. For compressed Taylor series this can be done
        # more efficiently.

        # Let's take the example u_xy + u_x + u_y = 0.
        # In the diagram below, C depicts a non zero source coefficient.
        # We divide these into two hyperplanes.
        #
        #  C              C             0
        #  C 0            C 0           0 0
        #  C 0 0       =  C 0 0      +  0 0 0
        #  C 0 0 0        C 0 0 0       0 0 0 0
        #  C C C C C      C 0 0 0 0     0 C C C C
        #
        # The calculations done when naively translating first hyperplane of the
        # source coefficients (C) to target coefficients (T) are shown
        # below in the graph. Each connection represents a O(1) calculation,
        # and the arrows go "up and to the right".
        #
        #  ┌─→C             T
        #  │  ↑
        #  │┌→C→0←─────┐->  T T
        #  ││ ↑ ↑      │
        #  ││ ┌─┘┌────┐│
        #  ││↱C→0↲0←─┐││    T T T
        #  │││└───⬏  │││
        #  └└└C→0 0 0│││    T T T T
        #     └───⬏ ↑│││
        #     └─────┘│││
        #     └──────┘││
        #     └───────┘│
        #     └────────┘
        #
        # By using temporaries (Y), this can be reduced as shown below.
        #
        #  ┌→C           Y             T
        #  │ ↑
        #  │↱C 0     ->  Y→0       ->  T T
        #  ││↑
        #  ││C 0 0       Y→0 0         T T T
        #  ││↑           └───⬏
        #  └└C 0 0 0     Y 0 0 0       T T T T
        #                └───⬏ ↑
        #                └─────┘
        #
        # Note that in the above calculation data is propagated upwards
        # in the first pass and then rightwards in the second pass.
        # Data propagation with zeros are not shown as they are not calculated.
        # If the propagation was done rightwards first and upwards second
        # number of calculations are higher as shown below.
        #
        #    C             ┌→Y           T
        #                  │ ↑
        #    C→0       ->  │↱Y↱Y     ->  T T
        #                  ││↑│↑
        #    C→0 0         ││Y│Y Y       T T T
        #    └───⬏         ││↑│↑ ↑
        #    C→0 0 0       └└Y└Y Y Y     T T T T
        #    └───⬏ ↑
        #    └─────┘
        #
        # For the second hyperplane, data is propagated rightwards first
        # and then upwards second which is opposite to that of the first
        # hyperplane.
        #
        #    0              0            0
        #
        #    0 0       ->   0↱0      ->  0 T
        #                    │↑
        #    0 0 0          0│0 0        0 T T
        #                    │↑ ↑
        #    0 C→C→C        0└Y Y Y      0 T T T
        #      └───⬏
        #
        # In other words, we're better off computing the translation
        # one dimension at a time. If the coefficient-identifying multi-indices
        # in the source expansion have the form (0, m) and (n, 0), where m>=0, n>=1,
        # then we calculate the output from (0, m) with the second
        # dimension as the fastest varying dimension and then calculate
        # the output from (n, 0) with the first dimension as the fastest
        # varying dimension.

        tgt_hyperplanes = \
            self.expansion_terms_wrangler._split_coeffs_into_hyperplanes()
        result = [0] * len(self.get_full_coefficient_identifiers())

        # axis morally iterates over 'hyperplane directions'
        for axis in range(self.dim):
            # {{{ index gymnastics

            # First, let's write source coefficients in target coefficient
            # indices. If target order is lower than source order, then
            # we will discard higher order terms from source coefficients.
            cur_dim_input_coeffs = \
                [0] * len(self.get_full_coefficient_identifiers())
            for d, mis in tgt_hyperplanes:
                # Only consider hyperplanes perpendicular to *axis*.
                if d != axis:
                    continue
                for mi in mis:
                    # When target order is higher than source order, we assume
                    # that the higher order source coefficients were zero.
                    if mi not in src_mi_to_index:
                        continue

                    src_idx = src_mi_to_index[mi]
                    tgt_idx = tgt_mi_to_index[mi]
                    cur_dim_input_coeffs[tgt_idx] = src_coeff_exprs[src_idx] * \
                            sym.UnevaluatedExpr(src_rscale/tgt_rscale)**sum(mi)

            if all(coeff == 0 for coeff in cur_dim_input_coeffs):
                continue

            # }}}

            # {{{ translation

            # As explained above using the unicode art, we use the orthogonal axis
            # as the last dimension to vary to reduce the number of operations.
            dims = list(range(axis)) + \
                   list(range(axis+1, self.dim)) + [axis]

            # d is the axis along which we translate.
            for d in dims:
                # We build the full target multipole and then compress it
                # at the very end.
                cur_dim_output_coeffs = \
                    [0] * len(self.get_full_coefficient_identifiers())
                for i, tgt_mi in enumerate(
                        self.get_full_coefficient_identifiers()):

                    # Calling this input_mis instead of src_mis because we
                    # converted the source coefficients to target coefficient
                    # indices beforehand.
                    for mi_i in range(tgt_mi[d]+1):
                        input_mi = mi_set_axis(tgt_mi, d, mi_i)
                        contrib = cur_dim_input_coeffs[tgt_mi_to_index[input_mi]]
                        for n, k, dist in zip(tgt_mi, input_mi, dvec):
                            assert n >= k
                            contrib /= math.factorial(n-k)
                            contrib *= \
                                sym.UnevaluatedExpr(dist/tgt_rscale)**(n-k)

                        cur_dim_output_coeffs[i] += contrib
                # cur_dim_output_coeffs is the input in the next iteration
                cur_dim_input_coeffs = cur_dim_output_coeffs

            # }}}

            for i in range(len(cur_dim_output_coeffs)):
                result[i] += cur_dim_output_coeffs[i]

        # {{{ simpler, functionally equivalent code
        if not _fast_version:
            src_mi_to_index = {mi: i for i, mi in enumerate(
                src_expansion.get_coefficient_identifiers())}
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
                result, tgt_rscale, sac=sac))


class VolumeTaylorMultipoleExpansion(
        VolumeTaylorExpansion,
        VolumeTaylorMultipoleExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorMultipoleExpansionBase.__init__(self, kernel, order, use_rscale)
        VolumeTaylorExpansion.__init__(self, kernel, order, use_rscale)


class LinearPDEConformingVolumeTaylorMultipoleExpansion(
        LinearPDEConformingVolumeTaylorExpansion,
        VolumeTaylorMultipoleExpansionBase):

    def __init__(self, kernel, order, use_rscale=None):
        VolumeTaylorMultipoleExpansionBase.__init__(self, kernel, order, use_rscale)
        LinearPDEConformingVolumeTaylorExpansion.__init__(
                self, kernel, order, use_rscale)


class LaplaceConformingVolumeTaylorMultipoleExpansion(
        LinearPDEConformingVolumeTaylorMultipoleExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("LaplaceConformingVolumeTaylorMultipoleExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorMultipoleExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class HelmholtzConformingVolumeTaylorMultipoleExpansion(
        LinearPDEConformingVolumeTaylorMultipoleExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("HelmholtzConformingVolumeTaylorMultipoleExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorMultipoleExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class BiharmonicConformingVolumeTaylorMultipoleExpansion(
        LinearPDEConformingVolumeTaylorMultipoleExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("BiharmonicConformingVolumeTaylorMultipoleExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorMultipoleExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

# }}}


# {{{ 2D Hankel-based expansions

class _HankelBased2DMultipoleExpansion(MultipoleExpansionBase):
    @abstractmethod
    def get_bessel_arg_scaling(self):
        return

    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_identifiers(self):
        return list(range(-self.order, self.order+1))

    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        if kernel is None:
            kernel = self.kernel

        from sumpy.symbolic import sym_real_norm_2, BesselJ
        avec_len = sym_real_norm_2(avec)

        arg_scale = self.get_bessel_arg_scaling()

        # The coordinates are negated since avec points from source to center.
        source_angle_rel_center = sym.atan2(-avec[1], -avec[0])
        return [
                kernel.postprocess_at_source(
                    BesselJ(c, arg_scale * avec_len, 0)
                    / rscale ** abs(c)
                    * sym.exp(sym.I * c * -source_angle_rel_center),
                    avec)
                for c in self.get_coefficient_identifiers()]

    def evaluate(self, kernel, coeffs, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2, Hankel1
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(c)]
                   * kernel.postprocess_at_target(
                       Hankel1(c, arg_scale * bvec_len, 0)
                       * rscale ** abs(c)
                       * sym.exp(sym.I * c * target_angle_rel_center), bvec)
                for c in self.get_coefficient_identifiers())

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None):
        if not isinstance(src_expansion, type(self)):
            raise RuntimeError(
                "do not know how to translate "
                f"{type(src_expansion).__name__} to {type(self).__name__}")

        if not self.use_rscale:
            src_rscale = 1
            tgt_rscale = 1

        from sumpy.symbolic import sym_real_norm_2, BesselJ
        dvec_len = sym_real_norm_2(dvec)
        new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        translated_coeffs = []
        for j in self.get_coefficient_identifiers():
            translated_coeffs.append(
                sum(src_coeff_exprs[src_expansion.get_storage_index(m)]
                    * BesselJ(m - j, arg_scale * dvec_len, 0)
                    * src_rscale ** abs(m)
                    / tgt_rscale ** abs(j)
                    * sym.exp(sym.I * (m - j) * new_center_angle_rel_old_center)
                for m in src_expansion.get_coefficient_identifiers()))
        return translated_coeffs


class H2DMultipoleExpansion(_HankelBased2DMultipoleExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dim == 2)

        super().__init__(
                kernel, order, use_rscale=use_rscale)

    def get_bessel_arg_scaling(self):
        return sym.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)


class Y2DMultipoleExpansion(_HankelBased2DMultipoleExpansion):
    def __init__(self, kernel, order, use_rscale=None):
        from sumpy.kernel import YukawaKernel
        assert (isinstance(kernel.get_base_kernel(), YukawaKernel)
                and kernel.dim == 2)

        super().__init__(
                kernel, order, use_rscale=use_rscale)

    def get_bessel_arg_scaling(self):
        return sym.I * sym.Symbol(self.kernel.get_base_kernel().yukawa_lambda_name)

# }}}

# vim: fdm=marker
