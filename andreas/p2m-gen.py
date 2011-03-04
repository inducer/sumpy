from __future__ import division

from pytools import memoize_method
import sympy as sp




# {{{ multi_index helpers

def mi_factorial(mi):
    from pytools import factorial
    result = 1
    for mi_i in mi:
        result *= factorial(mi_i)
    return result

def mi_power(vector, mi):
    result = 1
    for mi_i, vec_i in zip(mi, vector):
        result *= vec_i**mi_i
    return result



# }}}




M2P_KERNEL = """

__kernel void m2p(
% for i in range(dimensions)
  ${geometry_type} *c${i}_g,
% endfor
% for i in range(dimensions)
  ${geometry_type} *t${i}_g,
% endfor
  ${offset_t} *mpole_offset_starts_g,
  ${offset_t} *mpole_offset_g,
  ${coeff_type} *mpole_coeff_g)
{
  // Each work group is responsible for accumulating one
  // target cell.

  // 
  iblok = (N-1)/THREADS;
  int index = offset + iblok * THREADS + threadIdx.x;
  __syncthreads();
  for( int i=0; i<13; i++ )
    multipShrd[threadIdx.x*13+i] = multipGlob[index*13+i];
  __syncthreads();
  for( int i=0; i<N - (iblok * THREADS); i++ ) {
    multipole(i,target,multipShrd);
  }
  targetGlob[blockIdx.x * THREADS + threadIdx.x] = target;
}
"""





class TaylorExpansion:
    def __init__(self, kernel, order, dimensions):
        """
        :arg: kernel in terms of 'b' variable
        """

        self.order = order
        self.kernel = kernel
        self.dimensions = dimensions

        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        self.multi_indices = sorted(gnitstam(self.order, self.dimensions), key=sum)

        # given in terms of b variable
        self.basis = [
                self.diff_kernel(mi)
                for mi in self.multi_indices]

        # given in terms of a variable
        from exafmm.symbolic import make_sym_vector

        a = make_sym_vector("a", 3)
        self.coefficients = [
                mi_power(a, mi)/mi_factorial(mi)
                for mi in self.multi_indices]

    @memoize_method
    def diff_kernel(self, multi_index):
        if sum(multi_index) == 0:
            return self.kernel

        first_nonzero_axis = min(
                i for i in range(self.dimensions)
                if multi_index[i] > 0)

        lowered_mi = list(multi_index)
        lowered_mi[first_nonzero_axis] -= 1
        lowered_mi = tuple(lowered_mi)

        lower_diff_kernel = self.diff_kernel(lowered_mi)

        return sp.diff(lower_diff_kernel,
                sp.Symbol("b%d" % first_nonzero_axis))











def test_make_p2m():
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

    # {{{ generate M2P

    old_var = make_sym_vector("a", dimensions)
    new_var = (make_sym_vector("c", dimensions)
            - make_sym_vector("s", dimensions))

    print "-------------------------------"
    print "M2P"
    print "-------------------------------"
    vars_and_exprs = generate_cl_statements_from_assignments(
            [("mpole%d"% i, 
                vector_subs(coeff_i, old_var, new_var))
                for i, coeff_i in enumerate(texp.coefficients)], 
            subst_map=subst_map)

    for var, expr in vars_and_exprs:
        print "%s = %s" % (var, expr)

    # }}}

    # {{{ generate P2M

    print "-------------------------------"
    print "P2M"
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



if __name__ == "__main__":
    test_make_p2m()




# vim: foldmethod=marker
