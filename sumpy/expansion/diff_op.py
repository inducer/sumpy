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
from sumpy.tools import add_mi, find_linear_relationship
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
                    for j in range(val):
                        expr = expr.diff(x[i])
                sym_eq += expr * coeff
            res.append(sym_eq)
        return res


@memoize
def as_scalar_pde(pde, vec_idx):
    r"""
    Returns a scalar PDE that is satisfied by the *vec_idx* component
    of *pde*.

    :arg pde: An instance of :class:`LinearPDESystemOperator`
    :arg vec_idx: the index of the vector-valued function that we
                  want as a scalar PDE
    """
    from sumpy.tools import nullspace

    indices = set()
    for eq in pde.eqs:
        for deriv_ident in eq.keys():
            indices.add(deriv_ident.vec_idx)

    # this is already a scalar pde
    if len(indices) == 1 and list(indices)[0] == vec_idx:
        return pde

    from pytools import ProcessLogger
    plog = ProcessLogger(logger, "computing single PDE for multiple PDEs")

    from pytools import (
            generate_nonnegative_integer_tuples_summing_to_at_most
            as gnitstam)

    dim = pde.total_dims

    # slowly increase the order of the derivatives that we take of the
    # system of PDEs. Once we reach the order of the scalar PDE, this
    # loop will break
    for order in range(2, 100):
        mis = sorted(gnitstam(order, dim), key=sum)

        pde_mat = []
        coeff_ident_enumerate_dict = dict((tuple(mi), i) for
                                            (i, mi) in enumerate(mis))
        offset = len(mis)

        # Create a matrix of equations that are derivatives of the
        # original system of PDEs
        for mi in mis:
            for pde_dict in pde.eqs:
                eq = [0]*(len(mis)*(max(indices)+1))
                for ident, coeff in pde_dict.items():
                    c = tuple(add_mi(ident.mi, mi))
                    if c not in coeff_ident_enumerate_dict:
                        break
                    idx = offset*ident.vec_idx + coeff_ident_enumerate_dict[c]
                    eq[idx] = coeff
                else:
                    pde_mat.append(eq)

        if len(pde_mat) == 0:
            continue

        # Get the nullspace of the matrix and get the rows related to this
        # vec_idx
        n = nullspace(pde_mat)[offset*vec_idx:offset*(vec_idx+1), :]
        indep_row = find_linear_relationship(n)
        if len(indep_row) > 0:
            pde_dict = {}
            mult = indep_row[max(indep_row.keys())]
            for k, v in indep_row.items():
                pde_dict[DerivativeIdentifier(mis[k], 0)] = v / mult
            plog.done()
            return LinearPDESystemOperator(pde.dim, pmap(pde_dict))

    plog.done()
    assert False


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
