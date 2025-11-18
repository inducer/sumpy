from __future__ import annotations


__copyright__ = "Copyright (C) 2022 Isuru Fernando"

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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, TypeAlias, cast

import numpy as np
from typing_extensions import override

import loopy as lp
import pymbolic.primitives as p

import sumpy.symbolic as sym
from sumpy.expansion.local import (
    FourierBesselLocalExpansionMixin,
    LocalExpansionBase,
    VolumeTaylorLocalExpansionBase,
)
from sumpy.expansion.multipole import (
    HankelBased2DMultipoleExpansion,
    MultipoleExpansionBase,
    VolumeTaylorMultipoleExpansionBase,
)
from sumpy.tools import add_to_sac, matvec_toeplitz_upper_triangular


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import DTypeLike

    from pymbolic.typing import ArithmeticExpression
    from pytools import Hash
    from pytools.persistent_dict import KeyBuilder

    from sumpy.assignment_collection import SymbolicAssignmentCollection
    from sumpy.expansion.diff_op import MultiIndex
    from sumpy.kernel import Kernel


logger = logging.getLogger(__name__)

__doc__ = """
.. class:: TranslationClassesDepData
.. autoclass:: M2LTranslationClassFactoryBase
.. autoclass:: NonFFTM2LTranslationClassFactory
.. autoclass:: FFTM2LTranslationClassFactory
.. autoclass:: DefaultM2LTranslationClassFactory

.. autoclass:: M2LTranslationBase
.. autoclass:: VolumeTaylorM2LTranslation
.. autoclass:: VolumeTaylorM2LWithFFT
.. autoclass:: FourierBesselM2LTranslation
"""


# {{{ M2L translation factory

class M2LTranslationClassFactoryBase(ABC):
    """
    .. automethod:: get_m2l_translation_class
    """

    @abstractmethod
    def get_m2l_translation_class(
                self,
                base_kernel: Kernel,
                local_expansion_class: type[LocalExpansionBase]
            ) -> type[M2LTranslationBase]:
        """
        :returns: a subclass of :class:`M2LTranslationBase` suitable for
            *base_kernel* and *local_expansion_class*.
        """


class NonFFTM2LTranslationClassFactory(M2LTranslationClassFactoryBase):
    """An implementation of :class:`M2LTranslationClassFactoryBase` that uses
    non FFT M2L translation class.
    """

    @override
    def get_m2l_translation_class(
                self,
                base_kernel: Kernel,
                local_expansion_class: type[LocalExpansionBase]
            ) -> type[M2LTranslationBase]:
        from sumpy.expansion.local import (
            FourierBesselLocalExpansionMixin,
            VolumeTaylorLocalExpansionBase,
        )
        if issubclass(local_expansion_class, VolumeTaylorLocalExpansionBase):
            return VolumeTaylorM2LTranslation
        elif issubclass(local_expansion_class, FourierBesselLocalExpansionMixin):
            return FourierBesselM2LTranslation
        else:
            raise RuntimeError(
                f"unknown local_expansion_class: {local_expansion_class}")


class FFTM2LTranslationClassFactory(M2LTranslationClassFactoryBase):
    """An implementation of :class:`M2LTranslationClassFactoryBase` that uses
    FFT M2L translation class.
    """

    @override
    def get_m2l_translation_class(
                self,
                base_kernel: Kernel,
                local_expansion_class: type[LocalExpansionBase]
            ) -> type[M2LTranslationBase]:
        from sumpy.expansion.local import (
            FourierBesselLocalExpansionMixin,
            VolumeTaylorLocalExpansionBase,
        )
        if issubclass(local_expansion_class, VolumeTaylorLocalExpansionBase):
            return VolumeTaylorM2LWithFFT
        elif issubclass(local_expansion_class, FourierBesselLocalExpansionMixin):
            return FourierBesselM2LWithFFT
        else:
            raise RuntimeError(
                f"unknown local_expansion_class: {local_expansion_class}")


class DefaultM2LTranslationClassFactory(M2LTranslationClassFactoryBase):
    """An implementation of :class:`M2LTranslationClassFactoryBase` that gives the
    'best known' translation type for each kernel and local expansion class.
    """

    @override
    def get_m2l_translation_class(
                self,
                base_kernel: Kernel,
                local_expansion_class: type[LocalExpansionBase]
            ) -> type[M2LTranslationBase]:
        from sumpy.expansion.local import (
            FourierBesselLocalExpansionMixin,
            VolumeTaylorLocalExpansionBase,
        )
        if issubclass(local_expansion_class, VolumeTaylorLocalExpansionBase):
            return VolumeTaylorM2LWithFFT
        elif issubclass(local_expansion_class, FourierBesselLocalExpansionMixin):
            return FourierBesselM2LTranslation
        else:
            raise RuntimeError(
                f"unknown local_expansion_class: {local_expansion_class}")

# }}}


# {{{ M2LTranslationBase

TranslationClassesDepData: TypeAlias = tuple[sym.Expr, ...]
OptimizationCallable: TypeAlias = "Callable[[lp.TranslationUnit], lp.TranslationUnit]"


class M2LTranslationBase(ABC):
    """Base class for Multipole to Local Translation

    .. automethod:: translate
    .. automethod:: loopy_translate
    .. automethod:: translation_classes_dependent_data
    .. automethod:: translation_classes_dependent_ndata
    .. automethod:: preprocess_multipole_exprs
    .. automethod:: preprocess_multipole_nexprs
    .. automethod:: postprocess_local_exprs
    .. automethod:: postprocess_local_nexprs
    .. autoattribute:: use_fft
    .. autoattribute:: use_preprocessing
    """

    use_fft: ClassVar[bool] = False
    use_preprocessing: ClassVar[bool] = False

    # Don't convert to frozen dataclass: plain (non-dc) subclasses wind up mutable.
    @override
    def __setattr__(self, name: str, value: object) -> None:
        # These are intended to be stateless.
        raise AttributeError(
            f"{type(self)} is stateless and does not permit attribute modification")

    @override
    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    @abstractmethod
    def translate(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                translation_classes_dependent_data:
                    TranslationClassesDepData | None = None
            ) -> Sequence[sym.Expr]:
        ...

    def loopy_translate(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> lp.TranslationUnit:
        raise NotImplementedError(
            f"A direct loopy kernel for translation from "
            f"{src_expansion} to {tgt_expansion} using {self} is not implemented.")

    def translation_classes_dependent_data(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> TranslationClassesDepData:
        """Return an iterable of expressions that needs to be precomputed
        for multipole-to-local translations that depend only on the
        distance between the multipole center and the local center which
        is given as *dvec*.

        Since there are only a finite number of different values for the
        distance between per level, these can be precomputed for the tree.
        In :mod:`boxtree`, these distances are referred to as translation
        classes.

        When FFT is turned on, the output expressions are assumed to be
        transformed into Fourier space at the end by the caller.
        """
        return ()

    def translation_classes_dependent_ndata(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> int:
        """Return the number of expressions returned by
        :func:`~sumpy.expansion.m2l.M2LTranslationBase.translation_classes_dependent_data`.
        This method exists because calculating the number of expressions using
        the above method might be costly and
        :func:`~sumpy.expansion.m2l.M2LTranslationBase.translation_classes_dependent_data`
        cannot be memoized due to it having side effects through the argument
        *sac*.
        """
        return 0

    def loopy_translation_classes_dependent_data(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                result_dtype: DTypeLike) -> lp.TranslationUnit:
        """
        :arg result_dtype: the :mod:`numpy` type of the result.
        :returns: a :mod:`loopy` kernel that calculates the data described by
            :func:`~sumpy.expansion.m2l.M2LTranslationBase.translation_classes_dependent_data`.
        """
        return loopy_translation_classes_dependent_data(
                tgt_expansion, src_expansion, result_dtype
        )

    @abstractmethod
    def preprocess_multipole_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                sac: SymbolicAssignmentCollection | None,
                src_rscale: sym.Expr,
            ) -> Sequence[sym.Expr]:
        """Preprocess the multipole expansion for an optimized M2L.

        Preprocessing happens once per source box before M2L translation is done.

        These expressions are used in a separate :mod:`loopy` kernel to avoid
        having to process for each target and source box pair. When FFT is
        turned on, the output expressions are assumed to be transformed into
        Fourier space at the end by the caller. When FFT is turned off, the
        output expressions are equal to the multipole expansion coefficients
        with zeros added to make the M2L computation a circulant matrix.
        """

    def preprocess_multipole_nexprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> int:
        """Return the number of expressions returned by
        :func:`~sumpy.expansion.m2l.M2LTranslationBase.preprocess_multipole_exprs`.

        This method exists because calculating the number of expressions using
        the above method might be costly and it cannot be memoized due to it having
        side effects through the argument *sac*.
        """
        # For all use-cases we have right now, this is equal to the number of
        # translation classes dependent exprs. Use that as a default.
        return self.translation_classes_dependent_ndata(tgt_expansion, src_expansion)

    @abstractmethod
    def postprocess_local_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                m2l_result: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None) -> Sequence[sym.Expr]:
        """Postprocess the local expansion for an optimized M2L.

        Postprocessing happens once per target box just after the M2L translation
        is done and before storing the expansion coefficients for the local
        expansion.

        When FFT is turned on, the output expressions are assumed to have been
        transformed from Fourier space back to the original space by the caller.
        """

    def postprocess_local_nexprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase
            ) -> int:
        """Return the number of expressions given as input to
        :func:`~sumpy.expansion.m2l.M2LTranslationBase.postprocess_local_exprs`.

        This method exists because calculating the number of expressions using
        the above method might be costly and it cannot be memoized due to it
        having side effects through the argument *sac*.
        """
        # For all use-cases we have right now, this is equal to the number of
        # translation classes dependent exprs. Use that as a default.
        return self.translation_classes_dependent_ndata(tgt_expansion, src_expansion)

    def update_persistent_hash(self, key_hash: Hash, key_builder: KeyBuilder) -> None:
        key_hash.update(type(self).__name__.encode("utf8"))

    def optimize_loopy_kernel(self,
                knl: lp.TranslationUnit,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> lp.TranslationUnit:
        return lp.tag_inames(knl, {"itgt_box": "g.0"})


# }}} M2LTranslationBase

# {{{ VolumeTaylorM2LTranslation

class VolumeTaylorM2LTranslation(M2LTranslationBase):
    @override
    def translate(self,
                  tgt_expansion: LocalExpansionBase,
                  src_expansion: MultipoleExpansionBase,
                  src_coeff_exprs: Sequence[sym.Expr],
                  src_rscale: sym.Expr,
                  dvec: sym.Matrix,
                  tgt_rscale: sym.Expr,
                  sac: SymbolicAssignmentCollection | None = None,
                  translation_classes_dependent_data: (
                    TranslationClassesDepData | None) = None) -> Sequence[sym.Expr]:
        if translation_classes_dependent_data:
            derivatives = translation_classes_dependent_data
        else:
            derivatives = self.translation_classes_dependent_data(
                tgt_expansion, src_expansion, src_rscale, dvec, sac=sac)

        src_coeff_exprs = self.preprocess_multipole_exprs(
            tgt_expansion, src_expansion, src_coeff_exprs, sac, src_rscale)

        # Returns a big symbolic sum of matrix entries
        # (FIXME? Though this is just the correctness-checking
        # fallback for the FFT anyhow)
        result = matvec_toeplitz_upper_triangular(src_coeff_exprs, derivatives)
        result = self.postprocess_local_exprs(tgt_expansion, src_expansion,
            result, src_rscale, tgt_rscale, sac)

        return result

    @override
    def translation_classes_dependent_ndata(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> int:
        """Returns number of expressions in M2L global precomputation step.
        """
        mis_with_dummy_rows, _, _ = self._translation_classes_dependent_data_mis(
            tgt_expansion, src_expansion)

        return len(mis_with_dummy_rows)

    def _translation_classes_dependent_data_mis(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> tuple[Sequence[MultiIndex], Sequence[MultiIndex], MultiIndex]:
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

        dim = tgt_expansion.dim
        # max_mi is the multi-index which is the sum of the
        # element-wise maximum of source multi-indices and the
        # element-wise maximum of target multi-indices.
        max_mi = [0]*dim
        for i in range(dim):
            max_mi[i] = max(mi[i] for mi in src_expansion.get_coefficient_identifiers())
            max_mi[i] += max(mi[i] for mi in tgt_expansion.get_coefficient_identifiers())  # noqa: E501

        # These are the multi-indices representing the rows
        # in the circulant matrix.  Note that to get the circulant
        # matrix structure some multi-indices that are not in the
        # M2L translation matrix are added.
        # This corresponds to adding O(p^(d-1))
        # additional rows and columns in the case of some PDEs
        # like Laplace and O(p^d) in other cases.
        circulant_matrix_mis = list(gnitb([m + 1 for m in max_mi]))

        # These are the multi-indices representing the rows
        # in the M2L translation matrix without the additional
        # multi-indices in the circulant matrix
        needed_vector_terms: set[MultiIndex] = set()

        # For eg: 2D full Taylor Laplace, we only need kernel derivatives
        # (n1+n2, m1+m2), n1+m1<=p, n2+m2<=p
        for tgt_deriv in tgt_expansion.get_coefficient_identifiers():
            for src_deriv in src_expansion.get_coefficient_identifiers():
                needed = add_mi(src_deriv, tgt_deriv)
                if needed not in needed_vector_terms:
                    needed_vector_terms.add(needed)

        return circulant_matrix_mis, tuple(needed_vector_terms), tuple(max_mi)

    @override
    def translation_classes_dependent_data(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> TranslationClassesDepData:
        assert isinstance(tgt_expansion, VolumeTaylorLocalExpansionBase)
        assert isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase)

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

        if not tgt_expansion.use_rscale:
            src_rscale = sym.sympify(1)

        circulant_matrix_mis, needed_vector_terms, max_mi = (
                self._translation_classes_dependent_data_mis(tgt_expansion,
                                                             src_expansion))

        circulant_matrix_ident_to_index = {
            ident: i for i, ident in enumerate(circulant_matrix_mis)}

        # Create a expansion terms wrangler for derivatives up to order
        # (tgt order)+(src order) including a corresponding reduction matrix
        # For eg: 2D full Taylor Laplace, this is (n, m),
        # n+m<=2*p, n<=2*p, m<=2*p
        srcplusderiv_terms_wrangler = (
                src_expansion.expansion_terms_wrangler.copy(
                    order=tgt_expansion.order + src_expansion.order,
                    max_mi=tuple(max_mi)))
        srcplusderiv_full_coeff_ids = (
            srcplusderiv_terms_wrangler.get_full_coefficient_identifiers())
        srcplusderiv_ident_to_index = {
                ident: i for i, ident in enumerate(srcplusderiv_full_coeff_ids)}

        # The vector has the kernel derivatives and depends only on the distance
        # between the two centers
        taker = src_expansion.kernel.get_derivative_taker(dvec, src_rscale, sac)
        vector_stored: list[sym.Expr] = []

        # Calculate the kernel derivatives for the compressed set
        for term in srcplusderiv_terms_wrangler.get_coefficient_identifiers():
            kernel_deriv = taker.diff(term)
            vector_stored.append(kernel_deriv)

        # Calculate the kernel derivatives for the full set
        vector_full = (
            srcplusderiv_terms_wrangler.get_full_kernel_derivatives_from_stored(
                vector_stored, src_rscale))

        for term in srcplusderiv_full_coeff_ids:
            assert term in needed_vector_terms

        vector: list[sym.Expr] = [sym.sympify(0)] * len(needed_vector_terms)
        for i, term in enumerate(needed_vector_terms):
            vector[i] = add_to_sac(sac, vector_full[srcplusderiv_ident_to_index[term]])

        # Add zero values needed to make the translation matrix circulant
        derivatives_full: list[sym.Expr] = [sym.sympify(0)] * len(circulant_matrix_mis)
        for expr, mi in zip(vector, needed_vector_terms, strict=True):
            derivatives_full[circulant_matrix_ident_to_index[mi]] = expr

        return tuple(derivatives_full)

    @override
    def preprocess_multipole_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                sac: SymbolicAssignmentCollection | None,
                src_rscale: sym.Expr) -> Sequence[sym.Expr]:
        circulant_matrix_mis, _, _ = self._translation_classes_dependent_data_mis(
            tgt_expansion, src_expansion)
        circulant_matrix_ident_to_index = {
            ident: i for i, ident in enumerate(circulant_matrix_mis)}

        # Calculate the input vector for the circulant matrix
        input_vector: list[sym.Expr] = [sym.sympify(0)] * len(circulant_matrix_mis)
        for coeff, term in zip(
                src_coeff_exprs,
                src_expansion.get_coefficient_identifiers(), strict=True):
            input_vector[circulant_matrix_ident_to_index[term]] = add_to_sac(sac, coeff)

        return input_vector

    @override
    def preprocess_multipole_nexprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase) -> int:
        circulant_matrix_mis, _, _ = self._translation_classes_dependent_data_mis(
            tgt_expansion, src_expansion)

        return len(circulant_matrix_mis)

    def loopy_preprocess_multipole(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                # FIXME: why is result_dtype unused here?
                result_dtype: DTypeLike,
            ) -> tuple[lp.TranslationUnit, Sequence[OptimizationCallable]]:
        assert isinstance(tgt_expansion, VolumeTaylorLocalExpansionBase)
        assert isinstance(src_expansion, VolumeTaylorMultipoleExpansionBase)

        _, _, max_mi = self._translation_classes_dependent_data_mis(
            tgt_expansion, src_expansion)

        ncoeff_src = len(src_expansion.get_coefficient_identifiers())
        ncoeff_preprocessed = self.preprocess_multipole_nexprs(
            tgt_expansion, src_expansion)
        order = src_expansion.order

        output_coeffs = p.Variable("output_coeffs")
        input_coeffs = p.Variable("input_coeffs")
        output_icoeff = p.Variable("output_icoeff")
        input_icoeff = p.Variable("input_icoeff")
        input_coeffs_copy = p.Variable("input_coeffs_copy")

        dim = tgt_expansion.dim
        v = tuple(p.Variable(f"x{i}") for i in range(dim))

        from sumpy.expansion import (
            FullExpansionTermsWrangler,
            LinearPDEBasedExpansionTermsWrangler,
        )

        wrangler = src_expansion.expansion_terms_wrangler
        assert isinstance(wrangler, (FullExpansionTermsWrangler,
                                     LinearPDEBasedExpansionTermsWrangler))

        _, axis_permutation = wrangler._get_mi_ordering_key_and_axis_permutation()
        slowest_idx = axis_permutation[0]

        # max_mi[slowest_idx] = 2*(c - 1)
        c = max_mi[slowest_idx] // 2 + 1
        noutput_coeffs = cast("int", c * (2*order + 1) ** (dim - 1))

        domains = [
            "{[output_icoeff]: 0<=output_icoeff<noutput_coeffs}",
            "{[input_icoeff]: 0<=input_icoeff<ninput_coeffs}",
        ]

        insns = [
            lp.Assignment(
                assignee=input_coeffs_copy[input_icoeff],
                expression=input_coeffs[input_icoeff],
                id="input_copy",
                temp_var_type=lp.Optional(None),
            ),
        ]

        idx = output_icoeff
        for i in range(dim - 1, -1, -1):
            new_idx = idx % (max_mi[i] + 1) if i > 0 else idx
            insns.append(lp.Assignment(
                    assignee=v[i],
                    expression=new_idx,
                    id=f"set_x{i}",
                    temp_var_type=lp.Optional(None),
            ))
            idx = idx // (max_mi[i] + 1)

        input_idx = wrangler.get_storage_index(v)
        output_idx = 0
        mult = 1
        for i in range(dim - 1, -1, -1):
            output_idx += mult*v[i]
            mult *= (max_mi[i] + 1)

        insns += [
            lp.Assignment(
                assignee=output_coeffs[output_icoeff],
                expression=input_coeffs_copy[input_idx],
                predicates=frozenset([
                    p.Comparison(sum(v), "<=", order),
                    p.Comparison(v[slowest_idx], "<", c),
                ]),
                happens_after=frozenset(
                    [f"set_x{i}" for i in range(dim)] + ["input_copy"]
                ),
            )
        ]

        knl = lp.make_function(domains, insns,
            kernel_data=[
                lp.ValueArg("src_rscale", None),
                lp.GlobalArg("output_coeffs", None, shape=ncoeff_preprocessed,
                    is_input=False, is_output=True),
                lp.GlobalArg("input_coeffs", None, shape=ncoeff_src),
                ...],
            name="m2l_preprocess_inner",
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
            fixed_parameters={"noutput_coeffs": noutput_coeffs,
                              "ninput_coeffs": ncoeff_src},
        )

        from functools import partial
        optimizations = [
            partial(lp.split_iname, split_iname="m2l__input_icoeff",
                    inner_length=32, inner_tag="l.0"),
            partial(lp.split_iname, split_iname="m2l__output_icoeff",
                    inner_length=32, inner_tag="l.0"),
        ]

        return knl, optimizations

    @override
    def postprocess_local_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                m2l_result: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> Sequence[sym.Expr]:
        circulant_matrix_mis, _, _ = self._translation_classes_dependent_data_mis(
            tgt_expansion, src_expansion)
        circulant_matrix_ident_to_index = {
            ident: i for i, ident in enumerate(circulant_matrix_mis)}

        # Filter out the dummy rows and scale them for target
        rscale_ratio = add_to_sac(sac, tgt_rscale/src_rscale)
        result = [
            m2l_result[circulant_matrix_ident_to_index[term]]
            * rscale_ratio**sum(term)
            for term in tgt_expansion.get_coefficient_identifiers()]

        return result

    @override
    def postprocess_local_nexprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> int:
        return self.translation_classes_dependent_ndata(tgt_expansion, src_expansion)

    def loopy_postprocess_local(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                result_dtype: DTypeLike,
            ) -> tuple[lp.TranslationUnit, Sequence[OptimizationCallable]]:
        circulant_matrix_mis, _, _ = self._translation_classes_dependent_data_mis(
            tgt_expansion, src_expansion)
        circulant_matrix_ident_to_index = {
            ident: i for i, ident in enumerate(circulant_matrix_mis)}

        ncoeff_tgt = len(tgt_expansion.get_coefficient_identifiers())
        ncoeff_before_postprocessed = self.postprocess_local_nexprs(
            tgt_expansion, src_expansion)
        order = tgt_expansion.order

        fixed_parameters = {
            "ncoeff_tgt": ncoeff_tgt,
            "ncoeff_before_postprocessed": ncoeff_before_postprocessed,
            "order": order,
        }

        domains = [
            "{[iorder]: 0<iorder<=order}"
        ]

        insns = ["<> rscale_ratio = tgt_rscale / src_rscale {id=rscale_ratio}"]

        rscale_arr = p.Variable("rscale_arr")
        rscale_ratio = p.Variable("rscale_ratio")
        iorder = p.Variable("iorder")

        insns += [
            lp.Assignment(
                assignee=rscale_arr[0],
                expression=1,
                id="rscale_arr0",
                happens_after=frozenset(["rscale_ratio"]),
            ),
            lp.Assignment(
                assignee=rscale_arr[iorder],
                expression=rscale_arr[iorder - 1]*rscale_ratio,
                id="rscale_arr",
                happens_after=frozenset(["rscale_arr0"]),
            ),
        ]

        if self.use_fft and result_dtype in (np.float64, np.float32):
            result_func = p.Variable("real")
        else:
            def result_func(x: ArithmeticExpression) -> ArithmeticExpression:
                return x

        output_coeffs = p.Variable("output_coeffs")
        input_coeffs = p.Variable("input_coeffs")
        src_idx_sym = p.Variable("src_idx")
        rscale_idx_arr_sym = p.Variable("rscale_idx_arr")
        output_icoeff_sym = p.Variable("output_icoeff")

        src_idx = np.full(ncoeff_tgt, -1, dtype=np.int32)
        for output_icoeff, term in enumerate(
                tgt_expansion.get_coefficient_identifiers()):
            if self.use_fft:
                # since we reversed the M2L matrix, we reverse the result
                # to get the correct result
                n = len(circulant_matrix_mis)
                input_icoeff = n - 1 - circulant_matrix_ident_to_index[term]
            else:
                input_icoeff = circulant_matrix_ident_to_index[term]
            src_idx[output_icoeff] = input_icoeff

        rscale_idx_arr = np.full(ncoeff_tgt, -1, dtype=np.int32)
        for output_icoeff, term in enumerate(
                tgt_expansion.get_coefficient_identifiers()):
            rscale_idx_arr[output_icoeff] = sum(term)

        insns += [
            lp.Assignment(
                assignee=output_coeffs[output_icoeff_sym],
                expression=(
                    result_func(input_coeffs[src_idx_sym[output_icoeff_sym]])
                    * rscale_arr[rscale_idx_arr_sym[output_icoeff_sym]]),
                id="coeff_insn",
                happens_after=frozenset(["rscale_arr"]),
            )
        ]

        domains += [
            "{[output_icoeff]: 0<=output_icoeff<ncoeff_tgt}"
        ]

        knl = lp.make_function(
            domains,
            insns,
            kernel_data=[
                lp.ValueArg("src_rscale", None),
                lp.ValueArg("tgt_rscale", None),
                lp.GlobalArg("output_coeffs", None,
                    shape=ncoeff_tgt, is_input=False,
                    is_output=True),
                lp.GlobalArg("input_coeffs", None,
                    shape=ncoeff_before_postprocessed,
                    is_output=False, is_input=True),
                lp.TemporaryVariable("rscale_arr",
                    None,
                    shape=(order + 1,)),
                lp.TemporaryVariable(
                    src_idx_sym.name, initializer=src_idx,
                    address_space=lp.AddressSpace.GLOBAL, read_only=True),
                lp.TemporaryVariable(
                    rscale_idx_arr_sym.name, initializer=rscale_idx_arr,
                    address_space=lp.AddressSpace.GLOBAL, read_only=True),
                ...],
            name="m2l_postprocess_inner",
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
            fixed_parameters=fixed_parameters,
        )

        from functools import partial
        optimizations = [
            partial(lp.split_iname, split_iname="m2l__output_icoeff",
                    inner_length=32, inner_tag="l.0"),
        ]

        return knl, optimizations

# }}} VolumeTaylorM2LTranslation


# {{{ VolumeTaylorM2LWithPreprocessedMultipoles

class VolumeTaylorM2LWithPreprocessedMultipoles(VolumeTaylorM2LTranslation):
    use_preprocessing: ClassVar[bool] = True

    @override
    def translate(self,
                  tgt_expansion: LocalExpansionBase,
                  src_expansion: MultipoleExpansionBase,
                  src_coeff_exprs: Sequence[sym.Expr],
                  src_rscale: sym.Expr,
                  dvec: sym.Matrix,
                  tgt_rscale: sym.Expr,
                  sac: SymbolicAssignmentCollection | None = None,
                  translation_classes_dependent_data: (
                    TranslationClassesDepData | None) = None,
            ) -> Sequence[sym.Expr]:
        assert translation_classes_dependent_data

        derivatives = translation_classes_dependent_data
        # Returns a big symbolic sum of matrix entries
        # (FIXME? Though this is just the correctness-checking
        # fallback for the FFT anyhow)
        result = matvec_toeplitz_upper_triangular(src_coeff_exprs, derivatives)

        return result

    @override
    def loopy_translate(self,
                        tgt_expansion: LocalExpansionBase,
                        src_expansion: MultipoleExpansionBase) -> lp.TranslationUnit:
        ncoeff_src = self.preprocess_multipole_nexprs(tgt_expansion, src_expansion)
        ncoeff_tgt = self.postprocess_local_nexprs(tgt_expansion, src_expansion)

        icoeff_src = p.Variable("icoeff_src")
        icoeff_tgt = p.Variable("icoeff_tgt")
        domains = [f"{{[icoeff_tgt]: 0<=icoeff_tgt<{ncoeff_tgt} }}"]

        tgt_coeffs = p.Variable("tgt_coeffs")
        src_coeffs = p.Variable("src_coeffs")
        translation_classes_dependent_data = p.Variable("data")

        if self.use_fft:
            expr = src_coeffs[icoeff_tgt]*translation_classes_dependent_data[icoeff_tgt]
        else:
            toeplitz_first_row = src_coeffs[icoeff_src-icoeff_tgt]
            vector = translation_classes_dependent_data[icoeff_src]
            expr = toeplitz_first_row * vector

            domains.append(f"{{[icoeff_src]: icoeff_tgt<=icoeff_src<{ncoeff_src} }}")

        expr = src_coeffs[icoeff_tgt] * translation_classes_dependent_data[icoeff_tgt]

        insns = [
            lp.Assignment(
                assignee=tgt_coeffs[icoeff_tgt],
                expression=tgt_coeffs[icoeff_tgt] + expr
            ),
        ]

        knl = lp.make_function(
            domains,
            insns,
            kernel_data=[
                lp.GlobalArg("tgt_coeffs",
                             shape=lp.auto, is_input=True, is_output=True),
                lp.GlobalArg("src_coeffs, data",
                             shape=lp.auto, is_input=True, is_output=False),
                lp.ValueArg("src_rscale, tgt_rscale", is_input=True),
                ...],
            name="e2e",
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        )

        return knl


# }}} VolumeTaylorM2LWithPreprocessedMultipoles


# {{{ VolumeTaylorM2LWithFFT

class VolumeTaylorM2LWithFFT(VolumeTaylorM2LWithPreprocessedMultipoles):
    use_fft: ClassVar[bool] = True

    @override
    def translate(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
                translation_classes_dependent_data: (
                  TranslationClassesDepData | None) = None,
            ) -> Sequence[sym.Expr]:
        assert translation_classes_dependent_data

        derivatives = translation_classes_dependent_data
        assert len(src_coeff_exprs) == len(derivatives)

        result = [a*b for a, b in zip(derivatives, src_coeff_exprs, strict=True)]
        return result

    @override
    def translation_classes_dependent_data(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> TranslationClassesDepData:
        """Return an iterable of expressions that needs to be precomputed
        for multipole-to-local translations that depend only on the
        distance between the multipole center and the local center which
        is given as *dvec*.

        The final result should be transformed using an FFT.
        """
        derivatives_full = super().translation_classes_dependent_data(
            tgt_expansion, src_expansion, src_rscale, dvec, sac)

        # Note that the matrix we have now is a mirror image of a
        # circulant matrix. We reverse the first column to get the
        # first column for the circulant matrix and then finally
        # use the FFT for convolution represented by the circulant
        # matrix.
        return tuple(reversed(derivatives_full))

    @override
    def postprocess_local_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                m2l_result: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> Sequence[sym.Expr]:
        circulant_matrix_mis, _, _ = self._translation_classes_dependent_data_mis(
            tgt_expansion, src_expansion)
        n = len(circulant_matrix_mis)

        # since we reversed the M2L matrix, we reverse the result
        # to get the correct result
        m2l_result = list(reversed(m2l_result[:n]))

        return super().postprocess_local_exprs(tgt_expansion,
            src_expansion, m2l_result, src_rscale, tgt_rscale, sac)

    @override
    def optimize_loopy_kernel(self,
                knl: lp.TranslationUnit,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> lp.TranslationUnit:
        # Transform the kernel so that icoeff_tgt and its duplicates
        # become the outermost iname
        inames = knl.default_entrypoint.all_inames()
        knl = lp.rename_inames(knl,
            [iname for iname in inames if "icoeff_tgt" in iname],
            "icoeff_tgt", existing_ok=True)
        knl = lp.add_inames_to_insn(knl, "icoeff_tgt", None)

        # unprivatize icoeff_tgt because it is the outermost iname
        knl = lp.unprivatize_temporaries_with_inames(
                knl,
                frozenset({"icoeff_tgt"}), frozenset({"tgt_expansion"}))

        knl = lp.split_iname(knl, "icoeff_tgt", 64, inner_iname="inner",
                             inner_tag="l.0", outer_tag="g.1")
        knl = lp.tag_inames(knl, {"itgt_box": "g.0"})

        return knl


# }}} VolumeTaylorM2LWithFFT

# {{{ FourierBesselM2LTranslation

class FourierBesselM2LTranslation(M2LTranslationBase):
    @override
    def translate(self,
                  tgt_expansion: LocalExpansionBase,
                  src_expansion: MultipoleExpansionBase,
                  src_coeff_exprs: Sequence[sym.Expr],
                  src_rscale: sym.Expr,
                  dvec: sym.Matrix,
                  tgt_rscale: sym.Expr,
                  sac: SymbolicAssignmentCollection | None = None,
                  translation_classes_dependent_data: (
                    TranslationClassesDepData | None) = None) -> Sequence[sym.Expr]:
        if translation_classes_dependent_data is None:
            derivatives = self.translation_classes_dependent_data(tgt_expansion,
                src_expansion, src_rscale, dvec, sac=sac)
        else:
            derivatives = translation_classes_dependent_data

        src_coeff_exprs = self.preprocess_multipole_exprs(tgt_expansion,
            src_expansion, src_coeff_exprs, sac, src_rscale)

        translated_coeffs = [
            sum((derivatives[m + j + tgt_expansion.order + src_expansion.order]
                 * src_coeff_exprs[src_expansion.get_storage_index((m,))]
                 for m, in src_expansion.get_coefficient_identifiers()),
                sym.sympify(0))
            for j, in tgt_expansion.get_coefficient_identifiers()]

        translated_coeffs = self.postprocess_local_exprs(tgt_expansion,
                src_expansion, translated_coeffs, src_rscale, tgt_rscale,
                sac)

        return translated_coeffs

    @override
    def translation_classes_dependent_ndata(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
            ) -> int:
        nexpr = 2 * tgt_expansion.order + 2 * src_expansion.order + 1
        return nexpr

    @override
    def translation_classes_dependent_data(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> TranslationClassesDepData:
        assert isinstance(tgt_expansion, FourierBesselLocalExpansionMixin)
        assert isinstance(src_expansion, HankelBased2DMultipoleExpansion)

        dvec_len = sym.sym_real_norm_2(dvec)
        new_center_angle_rel_old_center = sym.atan2(dvec[1], dvec[0])
        arg_scale = tgt_expansion.get_bessel_arg_scaling()

        # [-(src_order+tgt_order), ..., 0, ..., (src_order + tgt_order)]
        translation_classes_dependent_data: list[sym.Expr] = (
            [sym.sympify(0)] * (2*tgt_expansion.order + 2 * src_expansion.order + 1))

        # The M2L is a mirror image of a Toeplitz matvec with Hankel function
        # evaluations. https://dlmf.nist.gov/10.23.F1
        # This loop computes the first row and the last column vector sufficient
        # to specify the matrix entries.
        for j, in tgt_expansion.get_coefficient_identifiers():
            idx_j = tgt_expansion.get_storage_index((j,))
            for m, in src_expansion.get_coefficient_identifiers():
                idx_m = src_expansion.get_storage_index((m,))
                translation_classes_dependent_data[idx_j + idx_m] = (
                    sym.Hankel1(m + j, arg_scale * dvec_len, 0)
                    * sym.exp(sym.I * (m + j) * new_center_angle_rel_old_center))

        return tuple(translation_classes_dependent_data)

    @override
    def preprocess_multipole_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                sac: SymbolicAssignmentCollection | None,
                src_rscale: sym.Expr) -> Sequence[sym.Expr]:
        src_coeff_exprs = list(src_coeff_exprs)
        for m, in src_expansion.get_coefficient_identifiers():
            src_coeff_exprs[src_expansion.get_storage_index((m,))] *= src_rscale**abs(m)

        return src_coeff_exprs

    @override
    def preprocess_multipole_nexprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase) -> int:
        return 2*src_expansion.order + 1

    @override
    def postprocess_local_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                m2l_result: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None) -> Sequence[sym.Expr]:
        # Filter out the dummy rows and scale them for target
        result: list[sym.Expr] = []
        for j, in tgt_expansion.get_coefficient_identifiers():
            result.append(
                    m2l_result[tgt_expansion.get_storage_index((j,))]
                    * tgt_rscale**(abs(j)) * sym.Integer(-1)**j)

        return result

    @override
    def postprocess_local_nexprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase) -> int:
        return 2*tgt_expansion.order + 1

# }}} FourierBesselM2LTranslation


# {{{ FourierBesselM2LWithPreprocessedMultipoles

class FourierBesselM2LWithPreprocessedMultipoles(FourierBesselM2LTranslation):
    use_preprocessing: ClassVar[bool] = True

    @override
    def translate(self,
                  tgt_expansion: LocalExpansionBase,
                  src_expansion: MultipoleExpansionBase,
                  src_coeff_exprs: Sequence[sym.Expr],
                  src_rscale: sym.Expr,
                  dvec: sym.Matrix,
                  tgt_rscale: sym.Expr,
                  sac: SymbolicAssignmentCollection | None = None,
                  translation_classes_dependent_data: (
                    TranslationClassesDepData | None) = None) -> Sequence[sym.Expr]:
        assert translation_classes_dependent_data
        derivatives = translation_classes_dependent_data

        translated_coeffs = [
            sum((derivatives[m + j + tgt_expansion.order + src_expansion.order]
                 * src_coeff_exprs[src_expansion.get_storage_index((m,))]
                 for m, in src_expansion.get_coefficient_identifiers()),
                sym.sympify(0))
            for j, in tgt_expansion.get_coefficient_identifiers()
        ]

        return translated_coeffs

    @override
    def loopy_translate(self,
                        tgt_expansion: LocalExpansionBase,
                        src_expansion: MultipoleExpansionBase) -> lp.TranslationUnit:
        ncoeff_src = self.preprocess_multipole_nexprs(tgt_expansion, src_expansion)
        ncoeff_tgt = self.postprocess_local_nexprs(tgt_expansion, src_expansion)

        icoeff_src = p.Variable("icoeff_src")
        icoeff_tgt = p.Variable("icoeff_tgt")
        domains = [f"{{[icoeff_tgt]: 0<=icoeff_tgt<{ncoeff_tgt} }}"]

        tgt_coeffs = p.Variable("tgt_coeffs")
        src_coeffs = p.Variable("src_coeffs")
        translation_classes_dependent_data = p.Variable("data")

        if self.use_fft:
            expr = (src_coeffs[icoeff_tgt]
                    * translation_classes_dependent_data[icoeff_tgt])
        else:
            expr = (src_coeffs[icoeff_src]
                   * translation_classes_dependent_data[icoeff_tgt + icoeff_src])
            domains.append(f"{{[icoeff_src]: 0<=icoeff_src<{ncoeff_src} }}")

        insns = [
            lp.Assignment(
                assignee=tgt_coeffs[icoeff_tgt],
                expression=tgt_coeffs[icoeff_tgt] + expr),
        ]

        knl = lp.make_function(domains, insns,
            kernel_data=[
                lp.GlobalArg("tgt_coeffs", shape=lp.auto, is_input=True,
                    is_output=True),
                lp.GlobalArg("src_coeffs, data",
                    shape=lp.auto, is_input=True, is_output=False),
                lp.ValueArg("src_rscale, tgt_rscale", is_input=True),
                ...],
            name="e2e",
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
        )

        return knl

# }}} FourierBesselM2LWithPreprocessedMultipoles


# {{{ FourierBesselM2LWithFFT

class FourierBesselM2LWithFFT(FourierBesselM2LWithPreprocessedMultipoles):
    use_fft: ClassVar[bool] = True

    def __init__(self) -> None:
        # FIXME: expansion with FFT is correct symbolically and can be verified
        # with sympy. However there are numerical issues that we have to deal
        # with. Greengard and Rokhlin 1988 attributes this to numerical
        # instability but gives rscale as a possible solution. Sumpy's rscale
        # choice is slightly different from Greengard and Rokhlin and that
        # might be the reason for this numerical issue.
        raise ValueError("Bessel based expansions with FFT are not supported yet.")

    @override
    def translate(self,
                  tgt_expansion: LocalExpansionBase,
                  src_expansion: MultipoleExpansionBase,
                  src_coeff_exprs: Sequence[sym.Expr],
                  src_rscale: sym.Expr,
                  dvec: sym.Matrix,
                  tgt_rscale: sym.Expr,
                  sac: SymbolicAssignmentCollection | None = None,
                  translation_classes_dependent_data: (
                    TranslationClassesDepData | None) = None) -> Sequence[sym.Expr]:
        assert translation_classes_dependent_data is not None

        derivatives = translation_classes_dependent_data
        assert len(derivatives) == len(src_coeff_exprs)

        return [a * b for a, b in zip(derivatives, src_coeff_exprs, strict=True)]

    @override
    def loopy_translate(self,
                        tgt_expansion: LocalExpansionBase,
                        src_expansion: MultipoleExpansionBase) -> lp.TranslationUnit:
        raise NotImplementedError

    @override
    def translation_classes_dependent_data(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_rscale: sym.Expr,
                dvec: sym.Matrix,
                sac: SymbolicAssignmentCollection | None = None,
            ) -> TranslationClassesDepData:
        translation_classes_dependent_data = (
                super().translation_classes_dependent_data(
                    tgt_expansion, src_expansion, src_rscale, dvec, sac))
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
        first_row_m2l, last_column_m2l = (
            translation_classes_dependent_data[:2*order],
            translation_classes_dependent_data[2*order:])

        first_column_toeplitz = last_column_m2l
        first_row_toeplitz = list(reversed(first_row_m2l))

        first_column_circulant = (
                list(first_column_toeplitz)
                + list(reversed(first_row_toeplitz)))

        return tuple(first_column_circulant)

    @override
    def preprocess_multipole_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                src_coeff_exprs: Sequence[sym.Expr],
                sac: SymbolicAssignmentCollection | None,
                src_rscale: sym.Expr) -> Sequence[sym.Expr]:
        result = super().preprocess_multipole_exprs(
            tgt_expansion, src_expansion, src_coeff_exprs, sac, src_rscale)

        result = list(reversed(result))
        result += [sym.sympify(0)] * (len(result) - 1)

        return result

    @override
    def postprocess_local_exprs(self,
                tgt_expansion: LocalExpansionBase,
                src_expansion: MultipoleExpansionBase,
                m2l_result: Sequence[sym.Expr],
                src_rscale: sym.Expr,
                tgt_rscale: sym.Expr,
                sac: SymbolicAssignmentCollection | None = None) -> Sequence[sym.Expr]:
        m2l_result = m2l_result[:2*tgt_expansion.order + 1]
        return super().postprocess_local_exprs(
            tgt_expansion, src_expansion, m2l_result, src_rscale, tgt_rscale, sac)

# }}} FourierBesselM2LWithFFT


# {{{ loopy_translation_classes_dependent_data

def loopy_translation_classes_dependent_data(
            tgt_expansion: LocalExpansionBase,
            src_expansion: MultipoleExpansionBase,
            result_dtype: DTypeLike) -> lp.TranslationUnit:
    """
    This is a helper function to create a loopy kernel to generate translation
    classes dependent data. This function uses symbolic expressions given by the
    M2L translation, converts them to pymboltc expressions and generates a loopy
    kernel. Note that the loopy kernel returned has lots of expressions in it and
    takes a long time. Therefore, this function should be used only as a fallback
    when there is no "loop-y" kernel to calculate the data.
    """
    src_rscale = sym.Symbol("src_rscale")
    dvec = sym.make_sym_vector("d", tgt_expansion.dim)

    from sumpy.assignment_collection import SymbolicAssignmentCollection

    sac = SymbolicAssignmentCollection()
    derivatives = tgt_expansion.m2l_translation.translation_classes_dependent_data(
        tgt_expansion, src_expansion, src_rscale, dvec, sac)

    vec_name = "m2l_translation_classes_dependent_data"
    tgt_coeff_names = [
            sac.assign_unique(f"m2l_translation_classes_dependent_data{i}", coeff_i)
            for i, coeff_i in enumerate(derivatives)]
    sac.run_global_cse()

    from sumpy.codegen import to_loopy_insns
    from sumpy.tools import to_complex_dtype

    insns = to_loopy_insns(
            sac.assignments.items(),
            vector_names=frozenset(["d"]),
            pymbolic_expr_maps=[tgt_expansion.get_code_transformer()],
            retain_names=frozenset(tgt_coeff_names),
            complex_dtype=to_complex_dtype(result_dtype),
            )
    insns = list(insns)

    data = p.Variable("m2l_translation_classes_dependent_data")
    happens_after = None
    for i in range(len(insns)):
        insn = insns[i]
        if isinstance(insn, lp.Assignment) and \
                cast("p.Variable", insn.assignee).name.startswith(vec_name):
            idx = int(cast("p.Variable", insn.assignee).name[len(vec_name):])
            insns[i] = lp.Assignment(
                assignee=data[idx],
                expression=insn.expression,
                id=f"data_{idx}",
                happens_after=happens_after,
            )
            happens_after = frozenset([f"data_{idx}"])

    knl = lp.make_function([], insns,
        kernel_data=[
            lp.ValueArg("src_rscale", None),
            lp.GlobalArg("d", None, shape=tgt_expansion.dim),
            lp.GlobalArg(data.name, None,
                shape=len(derivatives), is_input=False,
                is_output=True),
        ],
        name="m2l_data",
        lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
    )

    return knl

# }}} loopy_translation_classes_dependent_data

# vim: fdm=marker
