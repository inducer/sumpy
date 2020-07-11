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

__doc__ = """
Differential operator interface
-------------------------------

.. autoclass:: DifferentialOperator
"""


class DifferentialOperator(object):
    r"""
    Represents a scalar, constant-coefficient DifferentialOperator of
    dimension `dim`. It is represented by a frozen dictionary.
    The dictionary maps a multi-index given as a tuple to the coefficient.
    This object is immutable.
    """
    def __init__(self, dim, mi_to_coeff):
        """
        :arg dim: dimension of the DifferentialOperator
        :arg mi_to_coeff: A dictionary mapping a multi-index to a coefficient
        """
        self.dim = dim
        self.mi_to_coeff = mi_to_coeff

    def __mul__(self, param):
        mi_to_coeff = {}
        for k, v in self.mi_to_coeff.items():
            mi_to_coeff[k] = v * param
        return DifferentialOperator(self.dim, pmap(mi_to_coeff))

    __rmul__ = __mul__

    def __add__(self, other_pde):
        assert self.dim == other_pde.dim
        res = dict(self.mi_to_coeff)
        for k, v in other_pde.mi_to_coeff.items():
            if k in res:
                res[k] += v
            else:
                res[k] = v
        return DifferentialOperator(self.dim, pmap(res))

    __radd__ = __add__

    def __sub__(self, other_pde):
        return self + (-1)*other_pde

    def __repr__(self):
        return f"DifferentialOperator({self.dim}, {repr(self.mi_to_coeff)})"


def laplacian(pde):
    dim = pde.dim
    res = DifferentialOperator(dim, pmap())
    for j in range(dim):
        mi = [0]*dim
        mi[j] = 2
        res = res + diff(pde, tuple(mi))
    return res


def diff(pde, mi):
    res = {}
    for mi_to_coeff_mi, v in pde.mi_to_coeff.items():
        res[add_mi(mi_to_coeff_mi, mi)] = v
    return DifferentialOperator(pde.dim, pmap(res))


def make_identity_diff_op(dim):
    """
    Returns the identity as a differential operator.
    """
    mi = tuple([0]*dim)
    return DifferentialOperator(dim, pmap({mi: 1}))
