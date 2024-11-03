# mypy: disallow-untyped-defs

from __future__ import annotations


__copyright__ = "Copyright (C) 2019 Isuru Fernando"

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import sympy as sp
import sympy.polys.agca.modules as sp_modules
from immutabledict import immutabledict

from pytools import memoize

import sumpy.symbolic as sym
from sumpy.tools import add_mi


logger = logging.getLogger(__name__)

__doc__ = """
Differential operator interface
-------------------------------

.. autoclass:: LinearPDESystemOperator
.. autoclass:: DerivativeIdentifier
.. autofunction:: make_identity_diff_op
.. autofunction:: as_scalar_pde
"""


@dataclass(frozen=True)
class DerivativeIdentifier:
    """
    .. autoattribute:: mi
    .. autoattribute: vec_idx
    """

    mi: tuple[int, ...]
    """
    Multi-index of the derivative being taken, a tuple with a number of entries
    corresponding to the dimension.
    """

    vec_idx: int
    """
    In a PDE system of :math:`n` variables, an integer between :math:`0` and :math:`n-1`
    indicating which variable is being differentiated.
    """


Number_ish = int | float | complex | np.number


@dataclass(frozen=True, eq=True)
class LinearPDESystemOperator:
    r"""
    Represents a constant-coefficient linear differential operator of a
    vector-valued function with `dim` spatial variables. It is represented by a
    tuple of immutable dictionaries. The dictionary maps a
    :class:`DerivativeIdentifier` to the coefficient. This object is immutable.
    Optionally supports a time variable as the last variable in the multi-index
    of the :class:`DerivativeIdentifier`.

    .. autoattribute:: dim
    .. autoattribute:: eqs

    .. autoattribute:: order
    .. autoattribute:: total_dims
    .. automethod:: to_sym
    """

    dim: int
    eqs: tuple[Mapping[DerivativeIdentifier, sp.Expr], ...]

    if __debug__:
        def __post_init__(self) -> None:
            hash(self)

    @property
    def order(self) -> int:
        deg = 0
        for eq in self.eqs:
            deg = max(deg, max(sum(ident.mi) for ident in eq))
        return deg

    def __mul__(self, param: Number_ish) -> LinearPDESystemOperator:
        eqs: list[Mapping[DerivativeIdentifier, sp.Expr]] = []
        for eq in self.eqs:
            deriv_ident_to_coeff = {}
            for k, v in eq.items():
                deriv_ident_to_coeff[k] = v * param
            eqs.append(immutabledict(deriv_ident_to_coeff))
        return LinearPDESystemOperator(self.dim, tuple(eqs))

    __rmul__ = __mul__

    def __add__(
                self, other_diff_op: LinearPDESystemOperator
            ) -> LinearPDESystemOperator:
        assert self.dim == other_diff_op.dim
        assert len(self.eqs) == len(other_diff_op.eqs)
        eqs: list[Mapping[DerivativeIdentifier, sp.Expr]] = []
        for eq, other_eq in zip(self.eqs, other_diff_op.eqs, strict=True):
            res = dict(eq)
            for k, v in other_eq.items():
                if k in res:
                    res[k] += v
                else:
                    res[k] = v
            eqs.append(immutabledict(res))
        return LinearPDESystemOperator(self.dim, tuple(eqs))

    __radd__ = __add__

    def __sub__(
                self, other_diff_op: LinearPDESystemOperator
            ) -> LinearPDESystemOperator:
        return self + (-1)*other_diff_op

    def __repr__(self) -> str:
        return f"LinearPDESystemOperator({self.dim}, {self.eqs!r})"

    def __getitem__(self, idx: int | slice) -> LinearPDESystemOperator:
        item = self.eqs.__getitem__(idx)
        eqs = item if isinstance(item, tuple) else (item,)
        return LinearPDESystemOperator(self.dim, eqs)

    @property
    def total_dims(self) -> int:
        """
        Returns the total number of dimensions including time
        """
        did = next(iter(self.eqs[0].keys()))
        return len(did.mi)

    def to_sym(self, fnames: Sequence[str] | None = None) -> list[sp.Expr]:
        from sumpy.symbolic import Function, make_sym_vector
        x = list(make_sym_vector("x", self.dim))
        x += list(make_sym_vector("t", self.total_dims - self.dim))

        if fnames is None:
            noutputs = 0
            for eq in self.eqs:
                for deriv_ident in eq:
                    noutputs = max(noutputs, deriv_ident.vec_idx)
            fnames = [f"f{i}" for i in range(noutputs+1)]

        funcs = [Function(fname)(*x) for fname in fnames]

        res = []
        for eq in self.eqs:
            sym_eq: sp.Expr = sp.sympify(0)
            for deriv_ident, coeff in eq.items():
                expr = funcs[deriv_ident.vec_idx]
                for i, val in enumerate(deriv_ident.mi):
                    for _ in range(val):
                        expr = expr.diff(x[i])
                sym_eq += expr * coeff
            res.append(sym_eq)
        return res


def convert_module_to_matrix(
            module: Sequence[sp_modules.FreeModuleElement],
            generators: Sequence[sp.Expr]
        ) -> sp.Matrix:
    import sympy
    # poly is a sympy DMP (dense multi-variate polynomial)
    # type and we convert it to a sympy expression because
    # sympy matrices with polynomial entries are not supported.
    # see https://github.com/sympy/sympy/issues/21497
    return sympy.Matrix([[sympy.Poly(poly.to_dict(), *generators,
            domain=sympy.EX).as_expr() for poly in ideal.data] for ideal in module])


@memoize
def _get_all_scalar_pdes(pde: LinearPDESystemOperator) -> list[LinearPDESystemOperator]:
    import sympy
    from sympy.polys.orderings import grevlex
    gens = [sympy.symbols(f"_x{i}") for i in range(pde.dim)]
    gens += [sympy.symbols(f"_t{i}") for i in range(pde.total_dims - pde.dim)]

    max_vec_idx = max(deriv_ident.vec_idx for eq in pde.eqs
                      for deriv_ident in eq)

    pde_system_mat = sympy.zeros(len(pde.eqs), max_vec_idx + 1)
    for row, eq in enumerate(pde.eqs):
        for deriv_ident, coeff in eq.items():
            deriv_as_poly = 1
            for i, val in enumerate(deriv_ident.mi):
                deriv_as_poly *= gens[i]**val
            pde_system_mat[row, deriv_ident.vec_idx] += coeff * deriv_as_poly

    ring = sympy.EX.old_poly_ring(*gens, order=grevlex)
    column_ideals = [ring.free_module(1).submodule(*pde_system_mat[:, i].tolist(),
                        order=grevlex)
            for i in range(pde_system_mat.shape[1])]
    column_syzygy_modules = [ideal.syzygy_module() for ideal in column_ideals]

    ncols = len(column_syzygy_modules)

    # For each column i, we need to get the intersection of all the syzygy modules
    # except for the ith module. For n number of modules, this is $n*(n-2)$ work.
    # To reduce that we first calculate the intersection from the left adding one
    # module every iteration and store them. We then calculate the intersection
    # from the right adding one module every iteration and store them. Then
    # for each column we calculate the intersection of the left modules and the
    # right modules. This requires only $3*(n-2)$ work.

    def intersect(
                a: sp_modules.SubModulePolyRing,
                b: sp_modules.SubModulePolyRing,
            ) -> sp_modules.SubModulePolyRing:
        return a.intersect(b)

    left_intersections = list(accumulate(column_syzygy_modules, func=intersect))
    right_intersections = list(reversed(list(accumulate(reversed(
        column_syzygy_modules), func=intersect))))

    # At the end, calculate the intersection of the left modules and right modules
    # and calculate a groebner basis for it.
    module_intersections = [right_intersections[1]] + [
        intersect(left_intersections[i - 1], right_intersections[i + 1])
        for i in range(1, ncols - 1)
    ] + [left_intersections[ncols - 2]]

    # For each column in the PDE system matrix, we multiply that column by
    # the syzygy module intersection to get a set of scalar PDEs for that
    # column.
    scalar_pdes_vec = [
        (convert_module_to_matrix(module_intersections[i]._groebner_vec(),
            gens) * pde_system_mat)[:, i]
        for i in range(ncols)
    ]

    results = []
    for col in range(ncols):
        scalar_pde_polys = [sympy.Poly(pde, *gens, domain=sympy.EX) for
            pde in scalar_pdes_vec[col]]
        scalar_pdes = [pde for pde in scalar_pde_polys if pde.degree() > 0]
        scalar_pde = min(scalar_pdes, key=lambda x: x.degree()).monic()
        pde_dict = {
            DerivativeIdentifier(mi, 0): sym.sympify(coeff.as_expr().simplify()) for
            (mi, coeff) in zip(scalar_pde.monoms(), scalar_pde.coeffs(), strict=True)
        }
        results.append(LinearPDESystemOperator(pde.dim, (immutabledict(pde_dict),)))

    return results


def as_scalar_pde(pde: LinearPDESystemOperator, comp_idx: int) \
        -> LinearPDESystemOperator:
    r"""
    Returns a scalar PDE that is satisfied by the *comp_idx* component
    of *pde*.

    To do this, we first convert a system of PDEs into a matrix where each
    row represents one PDE of the system of PDEs and each column represents
    a component. We convert a derivative to a polynomial expression and
    multiply that by the coefficient and enter that value into the matrix.

    eg:

    .. math::
        \frac{\partial^2 u}{\partial x^2} + \
            2 \frac{\partial^2 v}{\partial x y} = 0 \\
        3 \frac{\partial^2 u}{\partial y^2} + \
            \frac{\partial^2 v}{\partial x^2} = 0

    is converted into,

    .. math::
      \begin{bmatrix}
        x^2   & 2xy \\
        2y^2  & x^2
      \end{bmatrix}.

    Let :math:`r_i` be the columns of the above matrix.  In order find a scalar PDE
    for the :math:`i`-th component, we need to find some polynomials,
    :math:`a_1, a_2, \ldots, a_n` such that the vector :math:`\sum_i a_i r_i` has
    zeros except for the :math:`i`-th component. In other words, we need to
    find a vector of polynomials such that the inner product of it with each of
    the columns except for the :math:`i`-th column is zero. i.e. :math:`a_1,
    a_2, \ldots, a_n` is a syzygy of all columns except for the :math:`i`-th
    column.

    To calculate a module that annihilates all but the :math:`i`-th column, we first
    calculate the syzygy module of each column. A syzygy of a column vector is a
    row vector which has inner product zero with the column vector.
    A syzygy module is the module of all such syzygies and is generated by a
    finite set of syzygies. After calculating the syzygy module of each column,
    we intersect them except for the :math:`i`-th module resulting in a
    syzygy module that when multiplied by the matrix gives a matrix
    with zeros **EXCEPT** for the :math:`i`-th column. Therefore the vector
    :math:`a_1, a_2, \ldots, a_n` is in this module.

    When there are multiple such scalar PDEs, we want to get a combination of
    them that has the smallest positive degree. To do this, we calculate a groebner
    basis of the polynomials. We use the Groebner basis property that the largest
    monomial of each polynomial generated by set of polynomials is divisible
    by the largest monomial of some polynomial in the Groebner basis.
    When a graded monomial ordering is used this implies that the degree of any
    polynomial generated is greater than or equal to the degree of a polynomial
    in the Groebner basis. We choose that polynomial as our scalar PDE.

    :arg pde: An instance of :class:`LinearPDESystemOperator`
    :arg comp_idx: the index of the component of the PDE solution vector
        for which a scalar PDE is requested.
    """
    indices = set()
    for eq in pde.eqs:
        for deriv_ident in eq:
            indices.add(deriv_ident.vec_idx)

    # this is already a scalar pde
    if len(indices) == 1 and next(iter(indices)) == comp_idx:
        return pde

    return _get_all_scalar_pdes(pde)[comp_idx]


def laplacian(diff_op: LinearPDESystemOperator) -> LinearPDESystemOperator:
    dim = diff_op.dim
    empty: tuple[Mapping[DerivativeIdentifier, sp.Expr], ...] = \
        (immutabledict(),) * len(diff_op.eqs)
    res = LinearPDESystemOperator(dim, empty)
    for j in range(dim):
        mi = [0]*diff_op.total_dims
        mi[j] = 2
        res = res + diff(diff_op, tuple(mi))
    return res


def diff(
            diff_op: LinearPDESystemOperator, mi: tuple[int, ...]
        ) -> LinearPDESystemOperator:
    eqs: list[Mapping[DerivativeIdentifier, sp.Expr]] = []
    for eq in diff_op.eqs:
        res = {}
        for deriv_ident, v in eq.items():
            new_mi = add_mi(deriv_ident.mi, mi)
            res[DerivativeIdentifier(new_mi, deriv_ident.vec_idx)] = v
        eqs.append(immutabledict(res))
    return LinearPDESystemOperator(diff_op.dim, tuple(eqs))


def divergence(diff_op: LinearPDESystemOperator) -> LinearPDESystemOperator:
    assert len(diff_op.eqs) == diff_op.dim
    res = LinearPDESystemOperator(diff_op.dim, (immutabledict(),))
    for i in range(diff_op.dim):
        mi = [0]*diff_op.total_dims
        mi[i] = 1
        res += diff(diff_op[i], tuple(mi))
    return res


def gradient(diff_op: LinearPDESystemOperator) -> LinearPDESystemOperator:
    assert len(diff_op.eqs) == 1
    eqs = []
    dim = diff_op.dim
    for i in range(dim):
        mi = [0]*diff_op.total_dims
        mi[i] = 1
        eqs.append(diff(diff_op, tuple(mi)).eqs[0])
    return LinearPDESystemOperator(dim, tuple(eqs))


def curl(diff_op: LinearPDESystemOperator) -> LinearPDESystemOperator:
    assert len(diff_op.eqs) == 3
    assert diff_op.dim == 3
    eqs = []
    mis = []
    for i in range(3):
        mi = [0]*diff_op.total_dims
        mi[i] = 1
        mis.append(tuple(mi))

    for i in range(3):
        new_pde = diff(diff_op[(i+2) % 3], mis[(i+1) % 3]) - \
            diff(diff_op[(i+1) % 3], mis[(i+2) % 3])
        eqs.append(new_pde.eqs[0])

    return LinearPDESystemOperator(diff_op.dim, tuple(eqs))


def concat(*ops: LinearPDESystemOperator) -> LinearPDESystemOperator:
    assert len(ops) >= 1
    dim = ops[0].dim
    for op in ops:
        assert op.dim == dim
    eqs = list(ops[0].eqs)
    for op in ops[1:]:
        eqs.extend(list(op.eqs))
    return LinearPDESystemOperator(dim, tuple(eqs))


def make_identity_diff_op(
            ninput: int, noutput: int = 1, time_dependent: bool = False
        ) -> LinearPDESystemOperator:
    """
    Returns the identity as a linear PDE system operator.
    if *include_time* is true, then the last dimension of the
    multi-index is time.

    :arg ninput: number of spatial variables of the function
    :arg noutput: number of output values of function
    :arg time_dependent: include time as a dimension
    """
    if time_dependent:  # noqa: SIM108
        mi = tuple([0]*(ninput + 1))
    else:
        mi = tuple([0]*ninput)
    return LinearPDESystemOperator(ninput, tuple(immutabledict(
                    {DerivativeIdentifier(mi, i): sp.sympify(1)})
                    for i in range(noutput)))
