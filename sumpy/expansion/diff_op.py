from __future__ import division, absolute_import

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

from pyrsistent import pmap
from sumpy.tools import add_mi
from collections import namedtuple

__doc__ = """
Differential operator interface
-------------------------------

.. autoclass:: LinearPDESystemOperator
.. autoclass:: DerivativeIdentifier
.. autofunction:: make_identity_diff_op
"""

DerivativeIdentifier = namedtuple("DerivativeIdentifier", ["mi", "vec_idx"])


class LinearPDESystemOperator(object):
    r"""
    Represents a constant-coefficient linear differential operator of a
    vector-valued function with `dim` variables. It is represented by a tuple of
    immutable dictionaries. The dictionary maps a :class:`DerivativeIdentifier`
    to the coefficient. This object is immutable.
    """
    def __init__(self, dim, *eqs):
        """
        :arg dim: dimension of the LinearPDESystemOperator
        :arg eqs: A list of dictionaries mapping a :class:`DerivativeIdentifier`
                  to a coefficient.
        """
        self.dim = dim
        self.eqs = tuple(eqs)

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

    def to_sym(self, fnames=None):
        from sumpy.symbolic import make_sym_vector, Function
        x = make_sym_vector("x", self.dim)

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


def laplacian(diff_op):
    dim = diff_op.dim
    empty = [pmap()] * len(diff_op.eqs)
    res = LinearPDESystemOperator(dim, *empty)
    for j in range(dim):
        mi = [0]*dim
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
    dim = diff_op.dim
    res = LinearPDESystemOperator(dim, pmap())
    for i in range(dim):
        mi = [0]*dim
        mi[i] = 1
        res += diff(diff_op[i], tuple(mi))
    return res


def gradient(diff_op):
    assert len(diff_op.eqs) == 1
    eqs = []
    dim = diff_op.dim
    for i in range(dim):
        mi = [0]*dim
        mi[i] = 1
        eqs.append(diff(diff_op, tuple(mi)).eqs[0])
    return LinearPDESystemOperator(dim, *eqs)


def concat(op1, op2):
    assert op1.dim == op2.dim
    eqs = list(op1.eqs)
    eqs.extend(list(op2.eqs))
    return LinearPDESystemOperator(op1.dim, *eqs)


def make_identity_diff_op(ninput, noutput=1):
    """
    Returns the identity as a linear PDE system operator.
    :arg ninput: number of input variables to the function
    :arg noutput: number of output values of function
    """
    mi = tuple([0]*ninput)
    eqs = [pmap({DerivativeIdentifier(mi, i): 1}) for i in range(noutput)]
    return LinearPDESystemOperator(ninput, *eqs)
