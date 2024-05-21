__copyright__ = """
Copyright (C) 2024 Andreas Kloeckner
Copyright (C) 2024 Hirish Chandrasekaran
"""

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
from itertools import accumulate
import sumpy.symbolic as sym
import logging
from typing import List
import sympy as sp
from sumpy.expansion.diff_op import LinearPDESystemOperator
from pytools.obj_array import make_obj_array


__doc__ = """
.. autoclass:: Recurrence

.. autofunction:: make_sympy_vec
"""

def make_sympy_vec(name, n):
    return make_obj_array([sp.Symbol(f"{name}{i}") for i in range(n)])

class Recurrence:
    def __init__(self, sumpy_pde):
        self.sumpy_pde = sumpy_pde
    
    '''
    get_pde_in_recurrence_form
    Input: 
        - pde a LinearPDESystemOperator such that assert(len(pde.eqs) == 1) is true.
    Output: 
        - ode_in_r, which is the pde but now as an ode in r, which f(r) satisfies. 
        - var represents the variables for the input
        - n_derivs, the max number of derivatives of f that are floating around.

    Description: We assume we are handed a system of 1 sumpy PDE (pde) and we output the 
    pde in a way that allows us to easily replace derivatives with respect to r. 
    '''
    def get_pde_in_recurrence_form():
        laplace = self.sumpy_pde
        dim = laplace.dim
        order = laplace.order
        assert(len(laplace.eqs) == 1)
        ops = len(laplace.eqs[0])
        derivs = []
        coeffs = []
        for i in laplace.eqs[0]:
            derivs.append(i.mi)
            coeffs.append(laplace.eqs[0][i])
        var = make_sympy_vec("x", dim)
        r = sp.sqrt(sum(var**2))

        eps = sp.symbols("epsilon")
        rval = r + eps
        
        f = sp.Function("f")
        f_derivs = [sp.diff(f(rval),eps,i) for i in range(order+1)]
        
        def compute_term(a, t):
            term = a
            for i in range(len(t)):
                term = term.diff(var[i], t[i])
            return term

        pde = 0
        for i in range(ops):
            pde += coeffs[i] * compute_term(f(rval), derivs[i])
            
        n_derivs = len(f_derivs)
        f_r_derivs = make_sympy_vec("f_r", n_derivs)

        for i in range(n_derivs):
            pde = pde.subs(f_derivs[i], f_r_derivs[i])
            
        return pde, var, n_derivs