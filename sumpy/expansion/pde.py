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

from sumpy.tools import add_mi

__doc__ = """
PDE interface
-------------

.. autoclass:: PDE
"""


class PDE(object):
    r"""
    Represents a scalar, constant-coefficient PDE of dimension `dim`.
    It is represented by a dictionary. The dictionary maps a multi-index
    given as a tuple to the coefficient. This object is immutable.
    """
    def __init__(self, dim, eq):
        """
        :arg dim: dimension of the PDE
        :arg eq: A dictionary mapping a multi-index to a coefficient
        """
        self.dim = dim
        self.eq = eq

    def __mul__(self, param):
        eq = {}
        for k, v in self.eq.items():
            eq[k] = v * param
        return PDE(self.dim, eq)

    __rmul__ = __mul__

    def __add__(self, other_pde):
        assert self.dim == other_pde.dim
        res = self.eq.copy()
        for k, v in other_pde.eq.items():
            if k in res:
                res[k] += v
            else:
                res[k] = v
        return PDE(self.dim, res)

    __radd__ = __add__

    def __sub__(self, other_pde):
        return self + (-1)*other_pde

    def __repr__(self):
        return f"PDE({self.dim}, {repr(self.eq)})"


def laplacian(pde):
    dim = pde.dim
    res = PDE(dim, {})
    for j in range(dim):
        mi = [0]*dim
        mi[j] = 2
        res = res + diff(pde, tuple(mi))
    return res


def diff(pde, mi):
    res = {}
    for eq_mi, v in pde.eq.items():
        res[add_mi(eq_mi, mi)] = v
    return PDE(pde.dim, res)


def make_pde_sym(dim):
    """
    Returns a PDE u = 0
    """
    mi = tuple([0]*dim)
    return PDE(dim, {mi: 1})
