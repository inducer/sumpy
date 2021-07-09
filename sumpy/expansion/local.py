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
from sumpy.tools import add_to_sac, fft

from sumpy.expansion import (
    ExpansionBase, VolumeTaylorExpansion, LinearPDEConformingVolumeTaylorExpansion)

from sumpy.tools import mi_increment_axis, matvec_toeplitz_upper_triangular
from pytools import single_valued
from typing import Tuple, Any

import logging
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

    .. attribute:: kernel
    .. attribute:: order
    .. attribute:: use_rscale
    .. attribute:: use_preprocessing_for_m2l

    .. automethod:: m2l_translation_classes_dependent_data
    .. automethod:: m2l_translation_classes_dependent_ndata
    .. automethod:: m2l_preprocess_multipole_exprs
    .. automethod:: m2l_preprocess_multipole_nexprs
    .. automethod:: m2l_postprocess_local_exprs
    .. automethod:: m2l_postprocess_local_nexprs
    .. automethod:: translate_from
    """
    init_arg_names = ("kernel", "order", "use_rscale", "use_preprocessing_for_m2l")

    def __init__(self, kernel, order, use_rscale=None,
            use_preprocessing_for_m2l=False):
        super().__init__(kernel, order, use_rscale)
        self.use_preprocessing_for_m2l = use_preprocessing_for_m2l

    def with_kernel(self, kernel):
        return type(self)(kernel, self.order, self.use_rscale,
                use_preprocessing_for_m2l=self.use_preprocessing_for_m2l)

    def update_persistent_hash(self, key_hash, key_builder):
        super().update_persistent_hash(key_hash, key_builder)
        key_builder.rec(key_hash, self.use_preprocessing_for_m2l)

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.kernel == other.kernel
            and self.order == other.order
            and self.use_rscale == other.use_rscale
            and self.use_preprocessing_for_m2l == other.use_preprocessing_for_m2l
        )

    def m2l_translation_classes_dependent_data(self, src_expansion, src_rscale,
            dvec, tgt_rscale, sac) -> Tuple[Any]:
        """Return an iterable of expressions that needs to be precomputed
        for multipole-to-local translations that depend only on the
        distance between the multipole center and the local center which
        is given as *dvec*.

        Since there are only a finite number of different values for the
        distance between per level, these can be precomputed for the tree.
        In :mod:`boxtree`, these distances are referred to as translation
        classes.
        """
        return tuple()

    def m2l_translation_classes_dependent_ndata(self, src_expansion):
        """Return the number of expressions returned by
        :func:`~sumpy.expansion.local.LocalExpansionBase.m2l_translation_classes_dependent_data`.
        This method exists because calculating the number of expressions using
        the above method might be costly and
        :func:`~sumpy.expansion.local.LocalExpansionBase.m2l_translation_classes_dependent_data`
        cannot be memoized due to it having side effects through the argument
        *sac*.
        """
        return 0

    def m2l_preprocess_multipole_exprs(self, src_expansion, src_coeff_exprs, sac,
            src_rscale):
        """Return the preprocessed multipole expansion for an optimized M2L.
        Preprocessing happens once per source box before M2L translation is done.

        When FFT is turned on, the input expressions are transformed into Fourier
        space. These expressions are used in a separate :mod:`loopy` kernel
        to avoid having to transform for each target and source box pair.
        When FFT is turned off, the expressions are equal to the multipole
        expansion coefficients with zeros added
        to make the M2L computation a circulant matvec.
        """
        raise NotImplementedError

    def m2l_preprocess_multipole_nexprs(self, src_expansion):
        """Return the number of expressions returned by
        :func:`~sumpy.expansion.local.LocalExpansionBase.m2l_preprocess_multipole_exprs`.
        This method exists because calculating the number of expressions using
        the above method might be costly and it cannot be memoized due to it having
        side effects through the argument *sac*.
        """
        # For all use-cases we have right now, this is equal to the number of
        # translation classes dependent exprs. Use that as a default.
        return self.m2l_translation_classes_dependent_ndata(src_expansion)

    def m2l_postprocess_local_exprs(self, src_expansion, m2l_result, src_rscale,
            tgt_rscale, sac):
        """Return postprocessed local expansion for an optimized M2L.
        Postprocessing happens once per target box just after the M2L translation
        is done and before storing the expansion coefficients for the local
        expansion.

        When FFT is turned on, the output expressions are transformed from Fourier
        space back to the original space.
        """
        raise NotImplementedError

    def m2l_postprocess_local_nexprs(self, src_expansion):
        """Return the number of expressions given as input to
        :func:`~sumpy.expansion.local.LocalExpansionBase.m2l_postprocess_local_exprs`.
        This method exists because calculating the number of expressions using
        the above method might be costly and it cannot be memoized due to it
        having side effects through the argument *sac*.
        """
        # For all use-cases we have right now, this is equal to the number of
        # translation classes dependent exprs. Use that as a default.
        return self.m2l_translation_classes_dependent_ndata(src_expansion)

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
                :func:`~sumpy.expansion.local.LocalExpansionBase.m2l_translation_classes_dependent_data`.
        """
        raise NotImplementedError


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
            from sumpy.tools import ExprDerivativeTaker
            deriv_taker = ExprDerivativeTaker(line_kernel, (tau,), sac=sac, rscale=1)

            return [kernel.postprocess_at_source(
                        deriv_taker.diff(i), avec).subs("tau", 0)
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

    def m2l_translation_classes_dependent_ndata(self, src_expansion):
        """Returns number of expressions in M2L global precomputation step.
        """
        mis_with_dummy_rows, mis_without_dummy_rows, _ = \
            self._m2l_translation_classes_dependent_data_mis(src_expansion)

        if self.use_preprocessing_for_m2l:
            return len(mis_with_dummy_rows)
        else:
            return len(mis_without_dummy_rows)

    def _m2l_translation_classes_dependent_data_mis(self, src_expansion):
        """We would like to compute the M2L by way of a circulant matrix below.
        To get the matrix representing the M2L into circulant form, a certain
        numbering of rows and columns (as identified by multi-indices) is
        required. This routine returns that numbering.

        .. note::

            The set of multi-indices returned may be a superset of the
            coefficients used by the expansion. On the input end, those
            coefficients are taken as zero. On output, they are simply
            dropped from the computed result.

        This method returns the multi-indices representing the rows
        of the circulant matrix, the multi-indices representing the rows
        of the M2L translation matrix and the maximum multi-index of the
        latter.
        """
        from pytools import generate_nonnegative_integer_tuples_below as gnitb
        from sumpy.tools import add_mi

        # max_mi is the multi-index which is the sum of the
        # element-wise maximum of source multi-indices and the
        # element-wise maximum of target multi-indices.
        max_mi = [0]*self.dim
        for i in range(self.dim):
            max_mi[i] = max(mi[i] for mi in
                              src_expansion.get_coefficient_identifiers())
            max_mi[i] += max(mi[i] for mi in
                              self.get_coefficient_identifiers())

        # These are the multi-indices representing the rows
        # in the circulant matrix.  Note that to get the circulant
        # matrix structure some multi-indices that is not in the
        # M2L translation matrix are added.
        # This corresponds to adding $\mathcal{O}(p^{d-1})$
        # additional rows and columns in the case of some PDEs
        # like Laplace and $\mathcal{O}(p^d)$ in other cases.
        circulant_matrix_mis = list(gnitb([m + 1 for m in max_mi]))

        # These are the multi-indices representing the rows
        # in the M2L translation matrix without the additional
        # multi-indices in the circulant matrix
        needed_vector_terms = set()
        # For eg: 2D full Taylor Laplace, we only need kernel derivatives
        # (n1+n2, m1+m2), n1+m1<=p, n2+m2<=p
        for tgt_deriv in self.get_coefficient_identifiers():
            for src_deriv in src_expansion.get_coefficient_identifiers():
                needed = add_mi(src_deriv, tgt_deriv)
                if needed not in needed_vector_terms:
                    needed_vector_terms.add(needed)

        return circulant_matrix_mis, tuple(needed_vector_terms), max_mi

    def m2l_translation_classes_dependent_data(self, src_expansion, src_rscale,
            dvec, tgt_rscale, sac):

        # We know the general form of the multipole expansion is:
        #
        #  coeff0 * diff(kernel(src - c1), mi0) +
        #    coeff1 * diff(kernel(src - c1), mi1) + ...
        #
        # To get the local expansion coefficients, we take derivatives of
        # the multipole expansion. For eg: the coefficient w.r.t mir is
        #
        #  coeff0 * diff(kernel(c2 - c1), mi0 + mir) +
        #    coeff1 * diff(kernel(c2 - c1), mi1 + mir) + ...
        #
        # The derivatives above depends only on `c2 - c1` and can be precomputed
        # globally as there are only a finite number of values for `c2 - c1` for
        # m2l.

        if not self.use_rscale:
            src_rscale = 1

        circulant_matrix_mis, needed_vector_terms, max_mi = \
            self._m2l_translation_classes_dependent_data_mis(src_expansion)

        circulant_matrix_ident_to_index = dict((ident, i) for i, ident in
                                enumerate(circulant_matrix_mis))

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
        taker = src_expansion.kernel.get_derivative_taker(dvec, src_rscale, sac)
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

        if self.use_preprocessing_for_m2l:
            # Add zero values needed to make the translation matrix circulant
            derivatives_full = [0]*len(circulant_matrix_mis)
            for expr, mi in zip(vector, needed_vector_terms):
                derivatives_full[circulant_matrix_ident_to_index[mi]] = expr

            # Note that the matrix we have now is a mirror image of a
            # circulant matrix. We reverse the first column to get the
            # first column for the circulant matrix and then finally
            # use the FFT for convolution represented by the circulant
            # matrix.
            return fft(list(reversed(derivatives_full)), sac=sac)

        return vector

    def m2l_preprocess_multipole_exprs(self, src_expansion, src_coeff_exprs, sac,
            src_rscale):
        circulant_matrix_mis, needed_vector_terms, max_mi = \
                self._m2l_translation_classes_dependent_data_mis(src_expansion)
        circulant_matrix_ident_to_index = dict((ident, i) for i, ident in
                            enumerate(circulant_matrix_mis))

        # Calculate the input vector for the circulant matrix
        input_vector = [0] * len(circulant_matrix_mis)
        for coeff, term in zip(
                src_coeff_exprs,
                src_expansion.get_coefficient_identifiers()):
            input_vector[circulant_matrix_ident_to_index[term]] = \
                    add_to_sac(sac, coeff)

        if self.use_preprocessing_for_m2l:
            return fft(input_vector, sac=sac)
        else:
            # When FFT is turned off, there is no preprocessing needed
            # Therefore no copying is done and the multipole expansion is sent to
            # the main M2L routine as it is. This method is used internally in the
            # the main M2l routine to avoid code duplication.
            return input_vector

    def m2l_postprocess_local_exprs(self, src_expansion, m2l_result, src_rscale,
            tgt_rscale, sac):
        circulant_matrix_mis, needed_vector_terms, max_mi = \
                self._m2l_translation_classes_dependent_data_mis(src_expansion)
        circulant_matrix_ident_to_index = dict((ident, i) for i, ident in
                            enumerate(circulant_matrix_mis))

        if self.use_preprocessing_for_m2l:
            n = len(circulant_matrix_mis)
            m2l_result = fft(m2l_result, inverse=True, sac=sac)
            # since we reversed the M2L matrix, we reverse the result
            # to get the correct result
            m2l_result = list(reversed(m2l_result[:n]))

        # Filter out the dummy rows and scale them for target
        rscale_ratio = add_to_sac(sac, tgt_rscale/src_rscale)
        result = [
            m2l_result[circulant_matrix_ident_to_index[term]]
            * rscale_ratio**sum(term)
            for term in self.get_coefficient_identifiers()]

        return result

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
            circulant_matrix_mis, needed_vector_terms, max_mi = \
                self._m2l_translation_classes_dependent_data_mis(src_expansion)
            circulant_matrix_ident_to_index = {ident: i for i, ident in
                                enumerate(circulant_matrix_mis)}

            if not m2l_translation_classes_dependent_data:
                derivatives = self.m2l_translation_classes_dependent_data(
                        src_expansion, src_rscale, dvec, tgt_rscale, sac)
            else:
                derivatives = m2l_translation_classes_dependent_data

            if self.use_preprocessing_for_m2l:
                assert m2l_translation_classes_dependent_data is not None
                assert len(src_coeff_exprs) == len(
                        m2l_translation_classes_dependent_data)
                return [a*b for a, b in zip(m2l_translation_classes_dependent_data,
                    src_coeff_exprs)]

            derivatives_full = [0]*len(circulant_matrix_mis)
            for expr, mi in zip(derivatives, needed_vector_terms):
                derivatives_full[circulant_matrix_ident_to_index[mi]] = expr

            input_vector = self.m2l_preprocess_multipole_exprs(src_expansion,
                src_coeff_exprs, sac, src_rscale)

            # Do the matvec
            output = matvec_toeplitz_upper_triangular(input_vector,
                derivatives_full)

            result = self.m2l_postprocess_local_exprs(src_expansion, output,
                src_rscale, tgt_rscale, sac)

            logger.info("building translation operator: done")
            return result

        # }}}

        # {{{ L2L

        # not coming from a Taylor multipole: expand via derivatives
        rscale_ratio = add_to_sac(sac, tgt_rscale/src_rscale)

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
        # }}}
        logger.info("building translation operator: done")
        return result


class VolumeTaylorLocalExpansion(
        VolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None,
            use_preprocessing_for_m2l=False):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale,
                use_preprocessing_for_m2l)
        VolumeTaylorExpansion.__init__(self, kernel, order, use_rscale)


class LinearPDEConformingVolumeTaylorLocalExpansion(
        LinearPDEConformingVolumeTaylorExpansion,
        VolumeTaylorLocalExpansionBase):

    def __init__(self, kernel, order, use_rscale=None,
            use_preprocessing_for_m2l=False):
        VolumeTaylorLocalExpansionBase.__init__(self, kernel, order, use_rscale,
                use_preprocessing_for_m2l)
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
    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_identifiers(self):
        return list(range(-self.order, self.order+1))

    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        if not self.use_rscale:
            rscale = 1

        from sumpy.symbolic import sym_real_norm_2, Hankel1

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

        from sumpy.symbolic import sym_real_norm_2, BesselJ
        bvec_len = sym_real_norm_2(bvec)
        target_angle_rel_center = sym.atan2(bvec[1], bvec[0])

        arg_scale = self.get_bessel_arg_scaling()

        return sum(coeffs[self.get_storage_index(c)]
                   * kernel.postprocess_at_target(
                       BesselJ(c, arg_scale * bvec_len, 0)
                       / rscale ** abs(c)
                       * sym.exp(sym.I * c * -target_angle_rel_center), bvec)
                for c in self.get_coefficient_identifiers())

    def m2l_translation_classes_dependent_ndata(self, src_expansion):
        nexpr = 2 * self.order + 2 * src_expansion.order + 1
        return nexpr

    def m2l_translation_classes_dependent_data(self, src_expansion, src_rscale,
            dvec, tgt_rscale, sac):

        from sumpy.symbolic import sym_real_norm_2, Hankel1
        from sumpy.tools import fft

        dvec_len = sym_real_norm_2(dvec)
        new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])
        arg_scale = self.get_bessel_arg_scaling()
        # [-(src_order+tgt_order), ..., 0, ..., (src_order + tgt_order)]
        m2l_translation_classes_dependent_data = \
                [0] * (2*self.order + 2 * src_expansion.order + 1)

        # The M2L is a mirror image of a Toeplitz matvec with Hankel function
        # evaluations. https://dlmf.nist.gov/10.23.F1
        # This loop computes the first row and the last column vector sufficient
        # to specify the matrix entries.
        for j in self.get_coefficient_identifiers():
            idx_j = self.get_storage_index(j)
            for m in src_expansion.get_coefficient_identifiers():
                idx_m = src_expansion.get_storage_index(m)
                m2l_translation_classes_dependent_data[idx_j + idx_m] = (
                    Hankel1(m + j, arg_scale * dvec_len, 0)
                    * sym.exp(sym.I * (m + j) * new_center_angle_rel_old_center))

        if self.use_preprocessing_for_m2l:
            order = src_expansion.order
            # For this expansion, we have a mirror image of a Toeplitz matrix.
            # First, we have to take the mirror image of the M2L matrix.
            #
            # After that the Toeplitz matrix has to be embedded in a circulant
            # matrix. In this cicrcular matrix the first part of the first
            # column is the first column of the Toeplitz matrix which is
            # the last column of the M2L matrix. The second part is the
            # reverse of the first row of the Toeplitz matrix which
            # is the reverse of the first row of the M2L matrix.
            first_row_m2l, last_column_m2l = \
                m2l_translation_classes_dependent_data[:2*order], \
                    m2l_translation_classes_dependent_data[2*order:]
            first_column_toeplitz = last_column_m2l
            first_row_toeplitz = list(reversed(first_row_m2l))

            first_column_circulant = list(first_column_toeplitz) + \
                    list(reversed(first_row_toeplitz))
            return fft(first_column_circulant, sac)

        return m2l_translation_classes_dependent_data

    def m2l_preprocess_multipole_exprs(self, src_expansion, src_coeff_exprs, sac,
            src_rscale):

        from sumpy.tools import fft
        src_coeff_exprs = list(src_coeff_exprs)
        for m in src_expansion.get_coefficient_identifiers():
            src_coeff_exprs[src_expansion.get_storage_index(m)] *= src_rscale**abs(m)

        if self.use_preprocessing_for_m2l:
            src_coeff_exprs = list(reversed(src_coeff_exprs))
            src_coeff_exprs += [0] * (len(src_coeff_exprs) - 1)
            res = fft(src_coeff_exprs, sac=sac)
            return res
        else:
            return src_coeff_exprs

    def m2l_postprocess_local_exprs(self, src_expansion, m2l_result, src_rscale,
            tgt_rscale, sac):

        if self.use_preprocessing_for_m2l:
            m2l_result = fft(m2l_result, inverse=True, sac=sac)
            m2l_result = m2l_result[:2*self.order+1]

        # Filter out the dummy rows and scale them for target
        result = []
        for j in self.get_coefficient_identifiers():
            result.append(m2l_result[self.get_storage_index(j)]
                    * tgt_rscale**(abs(j)) * sym.Integer(-1)**j)

        return result

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None, m2l_translation_classes_dependent_data=None):
        from sumpy.symbolic import sym_real_norm_2, BesselJ

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
            if m2l_translation_classes_dependent_data is None:
                derivatives = self.m2l_translation_classes_dependent_data(
                    src_expansion, src_rscale, dvec, tgt_rscale, sac=sac)
            else:
                derivatives = m2l_translation_classes_dependent_data

            translated_coeffs = []
            if self.use_preprocessing_for_m2l:
                assert m2l_translation_classes_dependent_data is not None
                assert len(derivatives) == len(src_coeff_exprs)
                for a, b in zip(derivatives, src_coeff_exprs):
                    translated_coeffs.append(a * b)
                return translated_coeffs

            src_coeff_exprs = self.m2l_preprocess_multipole_exprs(src_expansion,
                    src_coeff_exprs, sac, src_rscale)

            for j in self.get_coefficient_identifiers():
                translated_coeffs.append(
                    sum(derivatives[m + j + self.order + src_expansion.order]
                        * src_coeff_exprs[src_expansion.get_storage_index(m)]
                        for m in src_expansion.get_coefficient_identifiers()))

            translated_coeffs = self.m2l_postprocess_local_exprs(src_expansion,
                translated_coeffs, src_rscale, tgt_rscale, sac)
            return translated_coeffs

        raise RuntimeError("do not know how to translate %s to %s"
                           % (type(src_expansion).__name__,
                               type(self).__name__))


class H2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None,
            use_preprocessing_for_m2l=False):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dim == 2)

        super().__init__(kernel, order, use_rscale,
                use_preprocessing_for_m2l=use_preprocessing_for_m2l)

        if use_preprocessing_for_m2l:
            raise ValueError("H2DLocalExpansion with FFT is not implemented yet.")

        from sumpy.expansion.multipole import H2DMultipoleExpansion
        self.mpole_expn_class = H2DMultipoleExpansion

    def get_bessel_arg_scaling(self):
        return sym.Symbol(self.kernel.get_base_kernel().helmholtz_k_name)


class Y2DLocalExpansion(_FourierBesselLocalExpansion):
    def __init__(self, kernel, order, use_rscale=None,
            use_preprocessing_for_m2l=False):
        from sumpy.kernel import YukawaKernel
        assert (isinstance(kernel.get_base_kernel(), YukawaKernel)
                and kernel.dim == 2)

        super().__init__(kernel, order, use_rscale,
                use_preprocessing_for_m2l=use_preprocessing_for_m2l)

        if use_preprocessing_for_m2l:
            raise ValueError("Y2DLocalExpansion with FFT is not implemented yet.")

        from sumpy.expansion.multipole import Y2DMultipoleExpansion
        self.mpole_expn_class = Y2DMultipoleExpansion

    def get_bessel_arg_scaling(self):
        return sym.I * sym.Symbol(self.kernel.get_base_kernel().yukawa_lambda_name)

# }}}

# vim: fdm=marker
