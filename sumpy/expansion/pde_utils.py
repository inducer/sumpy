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

from collections import defaultdict
from sumpy.tools import CoeffIdentifier, add_mi, nth_root_assume_positive
import sumpy.symbolic as sym

class PDE(object):
    r"""
    Represents a system of PDEs of dimension `dim`. It is represented by a
    list of dictionaries with each dictionary representing a single PDE.
    Each dictionary maps a :class:`CoeffIdentifier` object to a value,

        .. math::

            \sum_{(mi, ident), c \text{pde\_dict}}
            \frac{\partial^{\sum(mi)}}{\partial u^mi} c = 0,

        where :math:`u` is the solution vector of the PDE.
    """
    def __init__(self, dim, *eqs):
        """
        :arg dim: dimension of the PDE
        :arg eqs: list of dictionaries mapping a :class:`CoeffIdentifier` to a
                  value or PDE instance
        """
        self.dim = dim
        self.eqs = []
        for obj in eqs:
            if isinstance(obj, PDE):
                self.eqs.extend(obj.eqs)
            else:
                self.eqs.append(obj)

    def __mul__(self, param):
        eqs = []
        for eq in self.eqs:
            new_eq = dict()
            for k, v in eq.items():
                new_eq[k] = eq[k] * param
            eqs.append(new_eq)
        return PDE(self.dim, *eqs)

    __rmul__ = __mul__

    def __add__(self, other_pde):
        assert self.dim == other_pde.dim
        assert len(self.eqs) == len(other_pde.eqs)
        eqs = []
        for eq1, eq2 in zip(self.eqs, other_pde.eqs):
            eq = defaultdict(lambda: 0)
            for k, v in eq1.items():
                eq[k] += v
            for k, v in eq2.items():
                eq[k] += v
            eqs.append(dict(eq))
        return PDE(self.dim, *eqs)

    __radd__ = __add__

    def __sub__(self, other_pde):
        return self + (-1)*other_pde

    def __getitem__(self, key):
        eqs = self.eqs.__getitem__(key)
        if not isinstance(eqs, list):
            eqs = [eqs]
        return PDE(self.dim, *eqs)

    def __repr__(self):
        return repr(self.eqs)


def laplacian(pde):
    p = PDE(pde.dim)
    for j in range(len(pde.eqs)):
        p.eqs.append(div(grad(pde[j])).eqs[0])
    return p


def diff(pde, mi):
    eqs = []
    for eq in pde.eqs:
        new_eq = defaultdict(lambda: 0)
        for ident, v in eq.items():
            new_mi = add_mi(ident.mi, mi)
            new_ident = CoeffIdentifier(tuple(new_mi), ident.iexpr)
            new_eq[new_ident] += v
        eqs.append(dict(new_eq))
    return PDE(pde.dim, *eqs)


def grad(pde):
    assert len(pde.eqs) == 1
    eqs = []
    for d in range(pde.dim):
        mi = [0]*pde.dim
        mi[d] += 1
        eqs.append(diff(pde, mi).eqs[0])
    return PDE(pde.dim, *eqs)


def curl(pde):
    assert len(pde.eqs) == pde.dim
    if pde.dim == 2:
        f1, f2 = pde[0], pde[1]
        return diff(f2, (1, 0)) - diff(f1, (0, 1))

    assert pde.dim == 3
    eqs = []
    for d in range(3):
        f1, f2 = pde[(d+1) % 3], pde[(d+2) % 3]
        mi1 = [0, 0, 0]
        mi1[(d+1) % 3] = 1
        mi2 = [0, 0, 0]
        mi2[(d+2) % 3] = 1
        new_eqs = diff(f2, mi1) - diff(f1, mi2)
        eqs.extend(new_eqs.eqs)
    return PDE(pde.dim, *eqs)


def div(pde):
    result = defaultdict(lambda: 0)
    for d, eq in enumerate(pde.eqs):
        for ident, v in eq.items():
            mi = list(ident.mi)
            mi[d] += 1
            new_ident = CoeffIdentifier(tuple(mi), ident.iexpr)
            result[new_ident] += v
    return PDE(pde.dim, dict(result))


def process_pde(pde):
    """
    Process a PDE object to return a PDE and a multiplier such that
    the sum of multiplier ** order * derivative * coefficient gives the
    original PDE `pde`.
    """
    multiplier = None
    for eq in pde.eqs:
        for ident1, val1 in eq.items():
            for ident2, val2 in eq.items():
                s1 = sum(ident1.mi)
                s2 = sum(ident2.mi)
                if s1 == s2:
                    continue
                m = nth_root_assume_positive(val1/val2, s2 - s1)
                if multiplier is None and not isinstance(m, (int, sym.Integer)):
                    multiplier = m

    if multiplier is None:
        return pde, 1
    eqs = []
    for eq in pde.eqs:
        new_eq = dict()
        for i, (k, v) in enumerate(eq.items()):
            new_eq[k] = v * multiplier**sum(k.mi)
            if i == 0:
                val = new_eq[k]
            new_eq[k] /= sym.sympify(val)
            if isinstance(new_eq[k], sym.Integer):
                new_eq[k] = int(new_eq[k])
        eqs.append(new_eq)
    return PDE(pde.dim, *eqs), multiplier


def make_pde_syms(dim, nexprs):
    """
    Returns a list of expressions of size `nexprs` to create a PDE
    of dimension `dim`.
    """
    eqs = []
    for iexpr in range(nexprs):
        mi = [0]*dim
        eq = dict()
        eq[CoeffIdentifier(tuple(mi), iexpr)] = 1
        eqs.append(eq)
    return PDE(dim, *eqs)

