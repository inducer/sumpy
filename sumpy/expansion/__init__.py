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

import logging
from pytools import memoize_method
import sumpy.symbolic as sym
from sumpy.tools import add_mi
from typing import List, Tuple

__doc__ = """
.. autoclass:: ExpansionBase
.. autoclass:: LinearPDEBasedExpansionTermsWrangler

Expansion Factories
^^^^^^^^^^^^^^^^^^^

.. autoclass:: ExpansionFactoryBase
.. autoclass:: DefaultExpansionFactory
.. autoclass:: VolumeTaylorExpansionFactory
"""

logger = logging.getLogger(__name__)


# {{{ base class

class ExpansionBase:
    """
    .. automethod:: with_kernel
    .. automethod:: __len__
    .. automethod:: get_coefficient_identifiers
    .. automethod:: coefficients_from_source
    .. automethod:: translate_from
    .. automethod:: __eq__
    .. automethod:: __ne__
    """
    init_arg_names = ("kernel", "order", "use_rscale")

    def __init__(self, kernel, order, use_rscale=None):
        # Don't be tempted to remove target derivatives here.
        # Line Taylor QBX can't do without them, because it can't
        # apply those derivatives to the expanded quantity.

        self.kernel = kernel
        self.order = order

        if use_rscale is None:
            use_rscale = True

        self.use_rscale = use_rscale

    # {{{ propagate kernel interface

    # This is here to conform this to enough of the kernel interface
    # to make it fit into sumpy.qbx.LayerPotential.

    @property
    def dim(self):
        return self.kernel.dim

    @property
    def is_complex_valued(self):
        return self.kernel.is_complex_valued

    def get_code_transformer(self):
        return self.kernel.get_code_transformer()

    def get_global_scaling_const(self):
        return self.kernel.get_global_scaling_const()

    def get_args(self):
        return self.kernel.get_args()

    def get_source_args(self):
        return self.kernel.get_source_args()

    # }}}

    def with_kernel(self, kernel):
        return type(self)(kernel, self.order, self.use_rscale)

    def __len__(self):
        return len(self.get_coefficient_identifiers())

    def get_coefficient_identifiers(self):
        """
        Returns the identifiers of the coefficients that actually get stored.
        """
        raise NotImplementedError

    def coefficients_from_source(self, kernel, avec, bvec, rscale, sac=None):
        """Form an expansion from a source point.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.
        :arg sac: a symbolic assignment collection where temporary
            expressions are stored.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """
        raise NotImplementedError

    def coefficients_from_source_vec(self, kernels, avec, bvec, rscale, weights,
            sac=None):
        """Form an expansion with a linear combination of kernels and weights.

        :arg avec: vector from source to center.
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.
        :arg sac: a symbolic assignment collection where temporary
            expressions are stored.

        :returns: a list of :mod:`sympy` expressions representing
            the coefficients of the expansion.
        """
        result = [0]*len(self)
        for knl, weight in zip(kernels, weights):
            coeffs = self.coefficients_from_source(knl, avec, bvec, rscale, sac=sac)
            for i in range(len(result)):
                result[i] += weight * coeffs[i]
        return result

    def evaluate(self, kernel, coeffs, bvec, rscale, sac=None):
        """
        :return: a :mod:`sympy` expression corresponding
            to the evaluated expansion with the coefficients
            in *coeffs*.
        """

        raise NotImplementedError

    def translate_from(self, src_expansion, src_coeff_exprs, src_rscale,
            dvec, tgt_rscale, sac=None):
        raise NotImplementedError

    def update_persistent_hash(self, key_hash, key_builder):
        key_hash.update(type(self).__name__.encode("utf8"))
        key_builder.rec(key_hash, self.kernel)
        key_builder.rec(key_hash, self.order)
        key_builder.rec(key_hash, self.use_rscale)

    def __eq__(self, other):
        return (
                type(self) == type(other)
                and self.kernel == other.kernel
                and self.order == other.order
                and self.use_rscale == other.use_rscale)

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self, **kwargs):
        new_kwargs = {
                name: getattr(self, name)
                for name in self.init_arg_names}

        for name in self.init_arg_names:
            new_kwargs[name] = kwargs.pop(name, getattr(self, name))

        if kwargs:
            raise TypeError("unexpected keyword arguments '%s'"
                % ", ".join(kwargs))

        return type(self)(**new_kwargs)


# }}}


# {{{ expansion terms wrangler

class ExpansionTermsWrangler:

    init_arg_names = ("order", "dim", "max_mi")

    def __init__(self, order, dim, max_mi=None):
        self.order = order
        self.dim = dim
        self.max_mi = max_mi

    def get_coefficient_identifiers(self):
        raise NotImplementedError

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives,
            rscale, sac=None):
        raise NotImplementedError

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale, sac=None):
        raise NotImplementedError

    @memoize_method
    def get_full_coefficient_identifiers(self):
        """
        Returns identifiers for every coefficient in the complete expansion.
        """
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        res = sorted(gnitstam(self.order, self.dim), key=sum)

        if self.max_mi is None:
            return res

        return [mi for mi in res if
            all(mi[i] <= self.max_mi[i] for i in range(self.dim))]

    def copy(self, **kwargs):
        new_kwargs = {
                name: getattr(self, name)
                for name in self.init_arg_names}

        for name in self.init_arg_names:
            new_kwargs[name] = kwargs.pop(name, getattr(self, name))

        if kwargs:
            raise TypeError("unexpected keyword arguments '%s'"
                % ", ".join(kwargs))

        return type(self)(**new_kwargs)

    @memoize_method
    def _split_coeffs_into_hyperplanes(self) -> List[Tuple[int, List[Tuple[int]]]]:
        r"""
        This splits the coefficients into :math:`O(p)` disjoint sets
        so that for each set, all the identifiers have the form,
        :math:`(m_1, m_2, ..., m_{j-1}, c, m_{j+1}, ... , m_{\text{dim}})`
        where :math:`c` is a constant. Geometrically, each set is an axis-aligned
        hyperplane.

        If this is an instance of :class:`LinearPDEBasedExpansionTermsWrangler`,
        then the number of sets will be :math:`O(1)`.

        In the returned object ``[(axis, [mi_1, mi2, ...]), ...]``,
        each element in the outer list represents a hyperplane. Each element
        is a 2-tuple where the first element in the tuple is the axis number *axis*
        to which the hyperplane is orthogonal. The second element in the tuple
        is a list of multi-indices in the hyperplane corresponding to the stored
        coefficients.

        E.g. for Laplace 3D order 4, the result could be::
        [
          (2, [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0), (1, 1, 0),
               (2, 1, 0), (0, 2, 0), (1, 2, 0), (0, 3, 0)]),
          (2, [(0, 0, 1), (1, 0, 1), (2, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]),
        ]
        """
        mis = self.get_full_coefficient_identifiers()
        mi_to_index = {mi: i for i, mi in enumerate(mis)}

        # Each hyperplane stored below is identified by a tuple of the axis
        # to which it is orthogonal to and the constant `c` described above
        hyperplanes = []
        if isinstance(self, LinearPDEBasedExpansionTermsWrangler):
            pde_dict, = self.knl.get_pde_as_diff_op().eqs

            if not all(ident.mi in mi_to_index for ident in pde_dict):
                # The order of the expansion is less than the order of the PDE.
                # Treat as if full expansion.
                pass
            else:
                # Calculate the multi-index that appears last in in the PDE in
                # degree lexicographic order.
                max_mi = self._get_max_index_in_pde(self)
                hyperplanes.extend(
                    (d, const)
                    for d in range(self.dim)
                    for const in range(max_mi[d]))

        if not hyperplanes:
            d = self.dim - 1
            hyperplanes.extend((d, const) for const in range(self.order + 1))

        res = []
        seen_mis = set()
        for d, const in hyperplanes:
            coeffs_in_hyperplane = []
            for mi in self.get_coefficient_identifiers():
                # Check if the multi-index is in this hyperplane and
                # if it is not in any of the hyperplanes we saw before
                # (This rejects coefficients that might be in multiple
                # hyperplanes, such as near the origin.)
                if mi[d] == const and mi not in seen_mis:
                    coeffs_in_hyperplane.append(mi)
                    seen_mis.add(mi)
            res.append((d, coeffs_in_hyperplane))

        return res


class FullExpansionTermsWrangler(ExpansionTermsWrangler):

    get_coefficient_identifiers = (
            ExpansionTermsWrangler.get_full_coefficient_identifiers)

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives,
            rscale, sac=None):
        return stored_kernel_derivatives

    get_stored_mpole_coefficients_from_full = (
            get_full_kernel_derivatives_from_stored)


# {{{ sparse matrix-vector multiplication

class CSEMatVecOperator:
    """
    A class to facilitate a fast matrix vector multiplication with
    common subexpression eliminated. In compressed Taylor
    series, the compression matrix's operation count can be
    reduced to `O(p**d)` from `O(p**{2d-1})` using CSE.
    Each row in the matrix is represented by a linear combination
    of values from input vector and a linear combination of values
    from the output vector.

    .. attribute:: from_input_coeffs_by_row

        An object of type ``List[List[Tuple[int, Any]]]``. Each element
        in the list represents a row of the matrix using a linear combination
        of values from the input vector. Each element has the form
        ``(index of input vector, coeff)``.

        Number of rows in the matrix represented is equal to the
        length of the `from_input_coeffs_by_row` list.

    .. attribute:: from_output_coeffs_by_row

        An object of type ``List[List[Tuple[int, Any]]]``. Each element
        in the list represents a row of the matrix using a linear combination
        of values from the output vector. Each element has the form
        ``(index of output vector, coeff)``.

    .. attribute:: shape

        Shape of the matrix as a tuple.
    """

    def __init__(self, from_input_coeffs_by_row, from_output_coeffs_by_row, shape):
        self.from_input_coeffs_by_row = from_input_coeffs_by_row
        self.from_output_coeffs_by_row = from_output_coeffs_by_row
        self.shape = shape
        assert len(self.from_input_coeffs_by_row) == shape[0]
        assert len(self.from_output_coeffs_by_row) == shape[0]

    def matvec(self, inp, wrap_intermediate=lambda x: x):
        """
        :arg inp: vector for the matrix vector multiplication

        :arg wrap_intermediate: a function to wrap intermediate expressions
             If not given, the number of operations might grow in the
             final expressions in the vector resulting in an expensive matvec.
        """
        assert len(inp) == self.shape[1]
        out = []
        for i in range(self.shape[0]):
            value = 0
            for input_index, coeff in self.from_input_coeffs_by_row[i]:
                value += inp[input_index] * coeff
            for output_index, coeff in self.from_output_coeffs_by_row[i]:
                value += out[output_index] * coeff
            out.append(wrap_intermediate(value))
        return out

    def transpose_matvec(self, inp, wrap_intermediate=lambda x: x):
        assert len(inp) == self.shape[0]
        res = [0]*self.shape[1]
        expr_all = list(inp)
        for i in reversed(range(self.shape[0])):
            for output_index, coeff in self.from_output_coeffs_by_row[i]:
                expr_all[output_index] += expr_all[i] * coeff
                expr_all[output_index] = wrap_intermediate(expr_all[output_index])
            for input_index, coeff in self.from_input_coeffs_by_row[i]:
                res[input_index] += expr_all[i] * coeff
        return res

# }}}


class LinearPDEBasedExpansionTermsWrangler(ExpansionTermsWrangler):
    """
    .. automethod:: __init__
    """

    init_arg_names = ("order", "dim", "knl", "max_mi")

    def __init__(self, order, dim, knl, max_mi=None):
        r"""
        :param order: order of the expansion
        :param dim: number of dimensions
        :param knl: kernel for the PDE
        """
        super().__init__(order, dim, max_mi)
        self.knl = knl

    def get_coefficient_identifiers(self):
        return self.stored_identifiers

    def get_full_kernel_derivatives_from_stored(self, stored_kernel_derivatives,
            rscale, sac=None):

        from sumpy.tools import add_to_sac
        projection_matrix = self.get_projection_matrix(rscale)
        return projection_matrix.matvec(stored_kernel_derivatives,
                lambda x: add_to_sac(sac, x))

    def get_stored_mpole_coefficients_from_full(self, full_mpole_coefficients,
            rscale, sac=None):

        from sumpy.tools import add_to_sac
        projection_matrix = self.get_projection_matrix(rscale)
        return projection_matrix.transpose_matvec(full_mpole_coefficients,
                lambda x: add_to_sac(sac, x))

    @property
    def stored_identifiers(self):
        stored_identifiers, _ = self.get_stored_ids_and_unscaled_projection_matrix()
        return stored_identifiers

    def _get_max_index_in_pde(self, pde_dict):
        """Calculate the multi-index that appears last in the PDE given the pde_dict
        A degree lexicographic order with the slowest varying index depending on
        the PDE is used. For two dimensions, this is either deglex or degrevlex.
        """
        pde_dict, = self.knl.get_pde_as_diff_op().eqs
        slowest_varying_index = self.dim - 1
        for ident in pde_dict.keys():
            if ident.mi.count(0) == self.dim - 1:
                non_zero_index = next(i for i in range(self.dim) if ident.mi[i] != 0)
                slowest_varying_index = min(slowest_varying_index, non_zero_index)

        mi_compare_axis = list(range(self.dim))
        mi_compare_axis[0], mi_compare_axis[slowest_varying_index] = \
                slowest_varying_index, 0

        def mi_key(ident):
            mi = ident.mi
            key = [sum(mi)]
            for i in range(self.dim):
                key.append(mi[mi_compare_axis[i]])
            return key

        return max((ident for ident in pde_dict.keys()), key=mi_key)

    @memoize_method
    def get_stored_ids_and_unscaled_projection_matrix(self):
        from pytools import ProcessLogger
        plog = ProcessLogger(logger, "compute PDE for Taylor coefficients")

        mis = self.get_full_coefficient_identifiers()
        coeff_ident_enumerate_dict = {tuple(mi): i for
                                            (i, mi) in enumerate(mis)}

        diff_op = self.knl.get_pde_as_diff_op()
        assert len(diff_op.eqs) == 1
        pde_dict = {k.mi: v for k, v in diff_op.eqs[0].items()}
        for ident in pde_dict.keys():
            if ident not in coeff_ident_enumerate_dict:
                # Order of the expansion is less than the order of the PDE.
                # In that case, the compression matrix is the identity matrix
                # and there's nothing to project
                from_input_coeffs_by_row = [[(i, 1)] for i in range(len(mis))]
                from_output_coeffs_by_row = [[] for _ in range(len(mis))]
                shape = (len(mis), len(mis))
                op = CSEMatVecOperator(from_input_coeffs_by_row,
                                       from_output_coeffs_by_row, shape)
                return mis, op

        max_mi = self._get_max_index_in_pde(self).mi
        max_mi_coeff = pde_dict[max_mi]
        max_mi_mult = -1/sym.sympify(max_mi_coeff)

        def is_stored(mi):
            """
            A multi_index mi is not stored if mi >= max_mi
            """
            return any(mi[d] < max_mi[d] for d in range(self.dim))

        stored_identifiers = []

        from_input_coeffs_by_row = []
        from_output_coeffs_by_row = []
        for i, mi in enumerate(mis):
            # If the multi-index is to be stored, keep the projection matrix
            # entry empty
            if is_stored(mi):
                idx = len(stored_identifiers)
                stored_identifiers.append(mi)
                from_input_coeffs_by_row.append([(idx, 1)])
                from_output_coeffs_by_row.append([])
                continue
            diff = [mi[d] - max_mi[d] for d in range(self.dim)]

            # eg: u_xx + u_yy + u_zz is represented as
            # [((2, 0, 0), 1), ((0, 2, 0), 1), ((0, 0, 2), 1)]
            assignment = []
            for other_mi, coeff in pde_dict.items():
                j = coeff_ident_enumerate_dict[add_mi(other_mi, diff)]
                if i == j:
                    # Skip the u_zz part here.
                    continue
                # PDE might not have max_mi_coeff = -1, divide by -max_mi_coeff
                # to get a relation of the form, u_zz = - u_xx - u_yy for Laplace 3D.
                assignment.append((j, coeff*max_mi_mult))
            from_input_coeffs_by_row.append([])
            from_output_coeffs_by_row.append(assignment)

        plog.done()

        logger.debug("number of Taylor coefficients was reduced from {orig} to {red}"
                     .format(orig=len(self.get_full_coefficient_identifiers()),
                             red=len(stored_identifiers)))

        shape = (len(mis), len(stored_identifiers))
        op = CSEMatVecOperator(from_input_coeffs_by_row,
                               from_output_coeffs_by_row, shape)
        return stored_identifiers, op

    @memoize_method
    def get_projection_matrix(self, rscale):
        r"""
        Return a :class:`CSEMatVecOperator` object which exposes a matrix vector
        multiplication operator for the projection matrix that expresses
        every derivative in terms of a set of "stored" derivatives.

        For example, for the PDE::

            u_xx + u_yy + u_zz = 0

        the coefficient matrix features the following entries::

                ... u_xx u_yy ... <= cols = only stored derivatives
                ==================
             ...| ...  ...  ... ...
                |
            u_zz| ...  -1   -1  ...

            ^ rows = one for every derivative

        the projection matrix `M` is the transpose of the coefficient matrix,
        so that
        .. math::

            c^{\text{local}}_{\text{full}} = M^T c^{\text{local}}_{\text{stored}}.\\
            c^{\text{mpole}}_{\text{stored}} = M c^{\text{mpole}}_{\text{full}}.
        """
        _, projection_matrix = \
            self.get_stored_ids_and_unscaled_projection_matrix()

        full_coeffs = self.get_full_coefficient_identifiers()

        projection_with_rscale = []
        for row, assignment in \
                enumerate(projection_matrix.from_output_coeffs_by_row):
            # For eg: (u_xxx / rscale**3) = (u_yy / rscale**2) * coeff1 +
            #                               (u_xx / rscale**2) * coeff2
            # is converted to u_xxx = u_yy * (rscale * coeff1) +
            #                         u_xx * (rscale * coeff2)
            row_rscale = sum(full_coeffs[row])
            from_output_coeffs_with_rscale = []
            for k, coeff in assignment:
                diff = row_rscale - sum(full_coeffs[k])
                mult = rscale**diff
                from_output_coeffs_with_rscale.append((k, coeff * mult))
            projection_with_rscale.append(from_output_coeffs_with_rscale)

        shape = projection_matrix.shape
        return CSEMatVecOperator(projection_matrix.from_input_coeffs_by_row,
                                 projection_with_rscale, shape)


# }}}


# {{{ volume taylor

class VolumeTaylorExpansionBase:

    @classmethod
    def get_or_make_expansion_terms_wrangler(cls, *key):
        """
        This stores the expansion terms wrangler at the class attribute level
        because recreating the expansion terms wrangler implicitly empties its
        caches.
        """
        try:
            wrangler = cls.expansion_terms_wrangler_cache[key]
        except KeyError:
            wrangler = cls.expansion_terms_wrangler_class(*key)
            cls.expansion_terms_wrangler_cache[key] = wrangler

        return wrangler

    @property
    def expansion_terms_wrangler(self):
        return self.get_or_make_expansion_terms_wrangler(
                *self.expansion_terms_wrangler_key)

    def get_coefficient_identifiers(self):
        """
        Returns the identifiers of the coefficients that actually get stored.
        """
        return self.expansion_terms_wrangler.get_coefficient_identifiers()

    def get_full_coefficient_identifiers(self):
        return self.expansion_terms_wrangler.get_full_coefficient_identifiers()

    @property
    @memoize_method
    def _storage_loc_dict(self):
        return {i: idx for idx, i in
                    enumerate(self.get_coefficient_identifiers())}

    def get_storage_index(self, i):
        return self._storage_loc_dict[i]


class VolumeTaylorExpansion(VolumeTaylorExpansionBase):

    expansion_terms_wrangler_class = FullExpansionTermsWrangler
    expansion_terms_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        self.expansion_terms_wrangler_key = (order, kernel.dim)


class LinearPDEConformingVolumeTaylorExpansion(VolumeTaylorExpansionBase):

    expansion_terms_wrangler_class = LinearPDEBasedExpansionTermsWrangler
    expansion_terms_wrangler_cache = {}

    # not user-facing, be strict about having to pass use_rscale
    def __init__(self, kernel, order, use_rscale):
        self.expansion_terms_wrangler_key = (order, kernel.dim, kernel)


class LaplaceConformingVolumeTaylorExpansion(
        LinearPDEConformingVolumeTaylorExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("LaplaceConformingVolumeTaylorExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class HelmholtzConformingVolumeTaylorExpansion(
        LinearPDEConformingVolumeTaylorExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("HelmholtzConformingVolumeTaylorExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class BiharmonicConformingVolumeTaylorExpansion(
        LinearPDEConformingVolumeTaylorExpansion):

    def __init__(self, *args, **kwargs):
        from warnings import warn
        warn("BiharmonicConformingVolumeTaylorExpansion is deprecated. "
             "Use LinearPDEConformingVolumeTaylorExpansion instead.",
                DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)

# }}}


# {{{ expansion factory

class ExpansionFactoryBase:
    """An interface
    .. automethod:: get_local_expansion_class
    .. automethod:: get_multipole_expansion_class
    """

    def get_local_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        raise NotImplementedError()

    def get_multipole_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        raise NotImplementedError()


class VolumeTaylorExpansionFactory(ExpansionFactoryBase):
    """An implementation of :class:`ExpansionFactoryBase` that uses Volume Taylor
    expansions for each kernel.
    """

    def get_local_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.expansion.local import VolumeTaylorLocalExpansion
        return VolumeTaylorLocalExpansion

    def get_multipole_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.expansion.multipole import VolumeTaylorMultipoleExpansion
        return VolumeTaylorMultipoleExpansion


class DefaultExpansionFactory(ExpansionFactoryBase):
    """An implementation of :class:`ExpansionFactoryBase` that gives the 'best known'
    expansion for each kernel.
    """

    def get_local_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.kernel import (HelmholtzKernel, YukawaKernel)

        from sumpy.expansion.local import (H2DLocalExpansion, Y2DLocalExpansion,
                LinearPDEConformingVolumeTaylorLocalExpansion,
                VolumeTaylorLocalExpansion)

        if (isinstance(base_kernel.get_base_kernel(), HelmholtzKernel)
                and base_kernel.dim == 2):
            return H2DLocalExpansion
        elif (isinstance(base_kernel.get_base_kernel(), YukawaKernel)
                and base_kernel.dim == 2):
            return Y2DLocalExpansion
        try:
            base_kernel.get_base_kernel().get_pde_as_diff_op()
            return LinearPDEConformingVolumeTaylorLocalExpansion
        except NotImplementedError:
            return VolumeTaylorLocalExpansion

    def get_multipole_expansion_class(self, base_kernel):
        """Returns a subclass of :class:`ExpansionBase` suitable for *base_kernel*.
        """
        from sumpy.kernel import (HelmholtzKernel, YukawaKernel)

        from sumpy.expansion.multipole import (H2DMultipoleExpansion,
                Y2DMultipoleExpansion,
                LinearPDEConformingVolumeTaylorMultipoleExpansion,
                VolumeTaylorMultipoleExpansion)

        if (isinstance(base_kernel.get_base_kernel(), HelmholtzKernel)
                and base_kernel.dim == 2):
            return H2DMultipoleExpansion
        elif (isinstance(base_kernel.get_base_kernel(), YukawaKernel)
                and base_kernel.dim == 2):
            return Y2DMultipoleExpansion
        try:
            base_kernel.get_base_kernel().get_pde_as_diff_op()
            return LinearPDEConformingVolumeTaylorMultipoleExpansion
        except NotImplementedError:
            return VolumeTaylorMultipoleExpansion

# }}}


# vim: fdm=marker
