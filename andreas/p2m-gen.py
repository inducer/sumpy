from __future__ import division

from pytools import memoize_method
import sympy as sp
import numpy as np




def test_make_p2m_sym():
    dimensions = 3
    from exafmm.symbolic import make_coulomb_kernel_in
    texp = TaylorExpansion(
            make_coulomb_kernel_in("b", dimensions),
            order=2, dimensions=dimensions)
    for mi, bi in zip(texp.multi_indices, texp.basis):
        print mi
        sp.pprint(bi)

    for mi, ci in zip(texp.multi_indices, texp.coefficients):
        print mi
        sp.pprint(ci)

    def gen_c_source_subst_map(dimensions):
        result = {}
        for i in range(dimensions):
            result["s%d" % i] = "src.s%d" % i
            result["t%d" % i] = "tgt.s%d" % i
            result["c%d" % i] = "ctr.s%d" % i

        return result

    subst_map = gen_c_source_subst_map(dimensions)

    from exafmm.symbolic.codegen import generate_cl_statements_from_assignments
    from exafmm.symbolic import vector_subs, make_sym_vector

    # {{{ generate P2M

    old_var = make_sym_vector("a", dimensions)
    new_var = (make_sym_vector("c", dimensions)
            - make_sym_vector("s", dimensions))

    print "-------------------------------"
    print "P2M"
    print "-------------------------------"
    vars_and_exprs = generate_cl_statements_from_assignments(
            [("mpole%d"% i, 
                vector_subs(coeff_i, old_var, new_var))
                for i, coeff_i in enumerate(texp.coefficients)], 
            subst_map=subst_map)

    for var, expr in vars_and_exprs:
        print "%s = %s" % (var, expr)

    # }}}

    # {{{ generate M2P

    print "-------------------------------"
    print "M2P"
    print "-------------------------------"

    old_var = make_sym_vector("b", dimensions)
    new_var = (make_sym_vector("t", dimensions)
            - make_sym_vector("c", dimensions))

    from exafmm.symbolic import vector_subs
    from exafmm.symbolic.codegen import generate_cl_statements_from_assignments
    vars_and_exprs = generate_cl_statements_from_assignments(
            [("contrib%d" % i, 
                vector_subs(basis_i, old_var, new_var))
                for i, basis_i in enumerate(texp.basis)], 
            subst_map=subst_map)

    for var, expr in vars_and_exprs:
        print "%s = %s" % (var, expr)

    # }}}




def test_make_m2p():
    dimensions = 3
    from exafmm.symbolic import make_coulomb_kernel_in
    from exafmm.expansion import TaylorExpansion
    texp = TaylorExpansion(
            make_coulomb_kernel_in("b", dimensions),
            order=2, dimensions=dimensions)

    from exafmm.m2p import make_m2p_source
    print make_m2p_source(np.float32, texp,
        [lambda expr: expr,
          lambda expr: sp.diff(expr, sp.Symbol("t0"))
            ])



if __name__ == "__main__":
    test_make_m2p()




# vim: foldmethod=marker
