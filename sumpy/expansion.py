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




class TaylorExpansion(Expansion):
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
        from sumpy.symbolic import make_sym_vector

        a = make_sym_vector("a", 3)
        from sumpy.tools import mi_power, mi_factorial
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

