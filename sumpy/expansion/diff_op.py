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

from collections import namedtuple
from pyrsistent import pmap
from pytools import memoize
from sumpy.tools import add_mi
import logging

logger = logging.getLogger(__name__)

__doc__ = """
Differential operator interface
-------------------------------

.. autoclass:: LinearPDESystemOperator
.. autoclass:: DerivativeIdentifier
.. autofunction:: make_identity_diff_op
.. autofunction:: as_scalar_pde
"""

DerivativeIdentifier = namedtuple("DerivativeIdentifier", ["mi", "vec_idx"])


class LinearPDESystemOperator:
    r"""
    Represents a constant-coefficient linear differential operator of a
    vector-valued function with `dim` spatial variables. It is represented by a
    tuple of immutable dictionaries. The dictionary maps a
    :class:`DerivativeIdentifier` to the coefficient. This object is immutable.
    Optionally supports a time variable as the last variable in the multi-index
    of the :class:`DerivativeIdentifier`.
    """
    def __init__(self, dim, *eqs):
        """
        :arg dim: Number of spatial dimensions of the LinearPDESystemOperator
        :arg eqs: A list of dictionaries mapping a :class:`DerivativeIdentifier`
                  to a coefficient.
        """
        self.dim = dim
        self.eqs = tuple(eqs)

    def __eq__(self, other):
        return self.dim == other.dim and self.eqs == other.eqs

    def __hash__(self):
        return hash((self.dim, self.eqs))

    @property
    def order(self):
        deg = 0
        for eq in self.eqs:
            deg = max(deg, max(sum(ident.mi) for ident in eq.keys()))
        return deg

    def __mul__(self, param):
        eqs = []
        for eq in self.eqs:
            deriv_ident_to_coeff = {}
            for k, v in eq.items():
                deriv_ident_to_coeff[k] = v * param
            eqs.append(pmap(deriv_ident_to_coeff))
        return LinearPDESystemOperator(self.dim, *eqs)

    __rmul__ = __mul__

    def __add__(self, other_diff_op):
        assert self.dim == other_diff_op.dim
        assert len(self.eqs) == len(other_diff_op.eqs)
        eqs = []
        for eq, other_eq in zip(self.eqs, other_diff_op.eqs):
            res = dict(eq)
            for k, v in other_eq.items():
                if k in res:
                    res[k] += v
                else:
                    res[k] = v
            eqs.append(pmap(res))
        return LinearPDESystemOperator(self.dim, *eqs)

    __radd__ = __add__

    def __sub__(self, other_diff_op):
        return self + (-1)*other_diff_op

    def __repr__(self):
        return f"LinearPDESystemOperator({self.dim}, {repr(self.eqs)})"

    def __getitem__(self, idx):
        item = self.eqs.__getitem__(idx)
        if not isinstance(item, tuple):
            item = (item,)
        return LinearPDESystemOperator(self.dim, *item)

    @property
    def total_dims(self):
        """
        Returns the total number of dimensions including time
        """
        return len(self.eqs[0].keys()[0].mi)

    def to_sym(self, fnames=None):
        from sumpy.symbolic import make_sym_vector, Function
        x = list(make_sym_vector("x", self.dim))
        x += list(make_sym_vector("t", self.total_dims - self.dim))

        if fnames is None:
            noutputs = 0
            for eq in self.eqs:
                for deriv_ident in eq.keys():
                    noutputs = max(noutputs, deriv_ident.vec_idx)
            fnames = [f"f{i}" for i in range(noutputs+1)]

        funcs = [Function(fname)(*x) for fname in fnames]

        res = []
        for eq in self.eqs:
            sym_eq = 0
            for deriv_ident, coeff in eq.items():
                expr = funcs[deriv_ident.vec_idx]
                for i, val in enumerate(deriv_ident.mi):
                    for _ in range(val):
                        expr = expr.diff(x[i])
                sym_eq += expr * coeff
            res.append(sym_eq)
        return res


def convert_module_to_matrix(module, generators):
    import sympy
    # poly is a sympy DMP (dense multi-variate polynomial)
    # type and we convert it to a sympy expression
    return sympy.Matrix([[sympy.Poly(poly.to_dict(), *generators,
            domain=sympy.EX).as_expr() for poly in ideal.data] for ideal in module])


@memoize
def _get_all_scalar_pdes(pde: LinearPDESystemOperator) -> LinearPDESystemOperator:
    import sympy
    from sympy.polys.orderings import grevlex
    gens = [sympy.symbols(f"_x{i}") for i in range(pde.dim)]
    gens += [sympy.symbols(f"_t{i}") for i in range(pde.total_dims - pde.dim)]

    max_vec_idx = 0
    for eq in pde.eqs:
        for deriv_ident in eq.keys():
            max_vec_idx = max(max_vec_idx, deriv_ident.vec_idx)

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

    # None represents the free module of the cartesian product of $m$ number of
    # polynomials where $m$ is the number of rows of the PDE matrix.
    left_intersections = [None]*ncols
    right_intersections = [None]*ncols

    def intersect(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return a.intersect(b)

    for i in range(1, ncols):
        left_intersections[i] = \
            intersect(left_intersections[i-1], column_syzygy_modules[i-1])
    for i in reversed(range(0, ncols-1)):
        right_intersections[i] = \
            intersect(right_intersections[i+1], column_syzygy_modules[i+1])

    # At the end, calculate the intersection of the left modules and right modules
    # and calculate a groebner basis for it.
    module_intersections = [
        intersect(left_intersections[i], right_intersections[i])._groebner_vec()
        for i in range(ncols)
    ]
    # For each column in the PDE system matrix, we multiply that column by
    # the syzygy module intersection to get a set of scalar PDEs for that
    # column.
    scalar_pdes_vec = [
        (convert_module_to_matrix(module_intersections[i],
            gens) * pde_system_mat)[:, i]
        for i in range(ncols)
    ]
    results = []
    for col in range(ncols):
        scalar_pdes = scalar_pdes_vec[col]

        minimal = None
        for scalar_pde in scalar_pdes:
            p = sympy.Poly(sympy.simplify(scalar_pde), *gens, domain=sympy.EX)
            if not p.degree() > 0:
                continue
            if minimal is None or p.degree() < minimal.degree():
                minimal = p

        scalar_pde = minimal.monic()
        pde_dict = {}
        for mi, coeff in zip(scalar_pde.monoms(), scalar_pde.coeffs()):
            pde_dict[DerivativeIdentifier(mi, 0)] = coeff
        results.append(LinearPDESystemOperator(pde.dim, pmap(pde_dict)))

    return results


def as_scalar_pde(pde: LinearPDESystemOperator, vec_idx: int) \
        -> LinearPDESystemOperator:
    r"""
    Returns a scalar PDE that is satisfied by the *vec_idx* component
    of *pde*. 123.

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

    Let :math:`r_i` be the columns of the above matrix.  In order find a scalar PDE for
    the i-th component, we need to find some polynomials,
    :math:`a_1, a_2, \ldots, a_n` such that the vector :math:`\sum_i a_i r_i` has zeros
    except for the i-th component. In other words, we need to find a vector of
    polynomials such that the inner product of it with each of the columns except
    for the i-th column is zero.
    i.e. :math:`a_1, a_2, \ldots, a_n` is a syzygy of all columns except for the i-th
    column.

    To calculate this, we first calculate the syzygy module of each column.
    A syzygy of a column vector is a row vector which has inner product zero
    with the column vector. A syzygy module is the module of all such syzygies
    and is generated by a finite set of syzygies.
    After calculating the syzygy module of each column, we intersect them together
    resulting in one module of syzgies that is finitely generated and this module
    generates all the syzygies that when multiplied by the matrix gives a vector
    with zeros except for the i-th component. Therefore the vector
    :math:`a_1, a_2, \ldots, a_n` is in this module.

    For each syzygy in the generating set, we calculate the inner product with the
    first column to get a polynomial that represents a scalar PDE for the first
    component. When there are multiple such scalar PDEs, we want to get a
    combination of them that has the smallest positive degree.
    To do this, we calculate a groebner basis of the polynomials. We use the
    Groebner basis property that the largest monomial of each polynomial generated
    by set of polynomials is divisible by the largest monomial of some polynomial
    in the Groebner basis. When a graded monomial ordering is used this implies
    that the degree of any polynomial generated is greater than or equal to the
    degree of a polynomial in the Groebner basis. We choose that polynomial
    as our scalar PDE.

    :arg pde: An instance of :class:`LinearPDESystemOperator`
    :arg vec_idx: the index of the vector-valued function that we
                  want as a scalar PDE
    """
    indices = set()
    for eq in pde.eqs:
        for deriv_ident in eq.keys():
            indices.add(deriv_ident.vec_idx)

    # this is already a scalar pde
    if len(indices) == 1 and list(indices)[0] == vec_idx:
        return pde

    return _get_all_scalar_pdes(pde)[vec_idx]


def laplacian(diff_op):
    dim = diff_op.dim
    empty = [pmap()] * len(diff_op.eqs)
    res = LinearPDESystemOperator(dim, *empty)
    for j in range(dim):
        mi = [0]*diff_op.total_dims
        mi[j] = 2
        res = res + diff(diff_op, tuple(mi))
    return res


def diff(diff_op, mi):
    eqs = []
    for eq in diff_op.eqs:
        res = {}
        for deriv_ident, v in eq.items():
            new_mi = add_mi(deriv_ident.mi, mi)
            res[DerivativeIdentifier(new_mi, deriv_ident.vec_idx)] = v
        eqs.append(pmap(res))
    return LinearPDESystemOperator(diff_op.dim, *eqs)


def divergence(diff_op):
    assert len(diff_op.eqs) == diff_op.dim
    res = LinearPDESystemOperator(diff_op.dim, pmap())
    for i in range(diff_op.dim):
        mi = [0]*diff_op.total_dims
        mi[i] = 1
        res += diff(diff_op[i], tuple(mi))
    return res


def gradient(diff_op):
    assert len(diff_op.eqs) == 1
    eqs = []
    dim = diff_op.dim
    for i in range(dim):
        mi = [0]*diff_op.total_dims
        mi[i] = 1
        eqs.append(diff(diff_op, tuple(mi)).eqs[0])
    return LinearPDESystemOperator(dim, *eqs)


def curl(pde):
    assert len(pde.eqs) == 3
    assert pde.dim == 3
    eqs = []
    mis = []
    for i in range(3):
        mi = [0]*pde.total_dims
        mi[i] = 1
        mis.append(tuple(mi))

    for i in range(3):
        new_pde = diff(pde[(i+2) % 3], mis[(i+1) % 3]) - \
            diff(pde[(i+1) % 3], mis[(i+2) % 3])
        eqs.append(new_pde.eqs[0])

    return LinearPDESystemOperator(pde.dim, *eqs)


def concat(*ops):
    ops = list(ops)
    assert len(ops) >= 1
    dim = ops[0].dim
    for op in ops:
        assert op.dim == dim
    eqs = list(ops[0].eqs)
    for op in ops[1:]:
        eqs.extend(list(op.eqs))
    return LinearPDESystemOperator(dim, *eqs)


def make_identity_diff_op(ninput, noutput=1, time_dependent=False):
    """
    Returns the identity as a linear PDE system operator.
    if *include_time* is true, then the last dimension of the
    multi-index is time.

    :arg ninput: number of spatial variables of the function
    :arg noutput: number of output values of function
    :arg time_dependent: include time as a dimension
    """
    if time_dependent:
        mi = tuple([0]*(ninput + 1))
    else:
        mi = tuple([0]*ninput)
    eqs = [pmap({DerivativeIdentifier(mi, i): 1}) for i in range(noutput)]
    return LinearPDESystemOperator(ninput, *eqs)
