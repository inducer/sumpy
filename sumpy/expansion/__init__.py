from __future__ import division

from pytools import memoize_method
import sympy as sp




class Expansion(object):
    def padded_coefficient_count_with_center(self, dtype):
        # cell centers are saved with expansion coefficients
        coeff_count = len(self.coefficients) + self.dimensions

        # FIXME: coefficients might be complex, cell centers
        # certainly not.

        coeff_size = dtype.itemsize
        align_to = 64 # bytes

        granularity, remainder = divmod(align_to, coeff_size)
        assert remainder == 0

        return granularity * (
                (coeff_count + granularity - 1) // granularity)




class TaylorMultipoleExpansion(Expansion):
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
        self.mi_to_index = dict(
                (mi, i) for i, mi in enumerate(self.multi_indices))

        # given in terms of b variable
        from sumpy.symbolic import diff_multi_index
        self.basis = [
                diff_multi_index(kernel, mi, "b")
                for mi in self.multi_indices]

        # given in terms of a variable
        from sumpy.symbolic import make_sym_vector

        a = make_sym_vector("a", dimensions)
        from sumpy.tools import mi_power, mi_factorial
        self.coefficients = [
                mi_power(a, mi)/mi_factorial(mi)
                for mi in self.multi_indices]

    def m2m_exprs(self, get_coefficient_expr):
        """Expressions for coefficients of shifted expansion, in terms of s
        (the shift from the old center to the new center), as well as the
        coefficients returned by *get_coefficient_expr* for each multi-index.
        """

        from sumpy.symbolic import make_sym_vector
        a = make_sym_vector("a", self.dimensions)
        s = make_sym_vector("s", self.dimensions)
        # new center = c+s

        from sumpy.symbolic import IdentityMapper, find_power_of
        class ToCoefficientMapper(IdentityMapper):
            def map_Mul(subself, expr):
                a_powers = tuple(int(find_power_of(ai, expr)) for ai in a)

                return (
                        expr/mi_power(a, a_powers)
                        * get_coefficient_expr(self.mi_to_index[a_powers]))

            map_Symbol = map_Mul

        tcm = ToCoefficientMapper()

        from sumpy.tools import mi_power
        return [tcm(mi_power(a+s, mi).expand()) for mi in self.multi_indices]

    def m2l_exprs(self, loc_exp, get_coefficient_expr):
        """Expressions for coefficients of the local expansion *loc_exp* of the
        multipole expansion *self*, whose coefficients are obtained by
        *get_coefficient_expr* for each coefficient index. The expressions are
        given in terms of *s* (the shift from the multipole center to the local
        center).
        """

        b2s_map = dict(
                (sp.Symbol("b%d" % i), sp.Symbol("s%d" % i))
                for i in range(self.dimensions))

        expansion = sum(
            get_coefficient_expr(i)
            * basis_func.subs(b2s_map)
            for i, basis_func in enumerate(self.basis))

        from sumpy.symbolic import diff_multi_index
        from sumpy.tools import mi_factorial
        return [diff_multi_index(expansion, loc_mi, "s") / mi_factorial(loc_mi)
                for loc_mi in loc_exp.multi_indices]






class TaylorLocalExpansion(Expansion):
    def __init__(self, order, dimensions):
        self.order = order
        self.dimensions = dimensions

        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        self.multi_indices = sorted(gnitstam(self.order, self.dimensions), key=sum)
        self.mi_to_index = dict(
                (mi, i) for i, mi in enumerate(self.multi_indices))

        from sumpy.symbolic import make_sym_vector

        from sumpy.tools import mi_power, mi_factorial

        # given in terms of b variable
        b = make_sym_vector("b", dimensions)
        self.basis = [
                mi_power(b, mi)/mi_factorial(mi)
                for mi in self.multi_indices]
