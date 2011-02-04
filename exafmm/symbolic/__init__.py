from __future__ import division
import sympy as sp




def make_sym_vector(name, components):
    return sp.Matrix(
            [sp.Symbol("%s%d" % (name, i)) for i in range(components)])





def make_coulomb_kernel(dimensions=3):
    from exafmm.symbolic import make_sym_vector
    tgt = make_sym_vector("t", dimensions)
    src = make_sym_vector("s", dimensions)

    return 1/sp.sqrt(((tgt-src).T*(tgt-src))[0,0])





