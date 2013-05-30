from __future__ import division

import sympy as sp

from sumpy.expansion import ExpansionBase, VolumeTaylorExpansionBase




class MultipoleExpansionBase(ExpansionBase):
    pass

# {{{ volume taylor

class VolumeTaylorMultipoleExpansion(MultipoleExpansionBase, VolumeTaylorExpansionBase):
    def coefficients_from_source(self, avec, bvec):
        from sumpy.tools import mi_derivative
        ppkernel = self.kernel.postprocess_at_source(
                self.kernel.get_expression(avec), avec)
        return [mi_derivative(ppkernel, avec, mi)
                for mi in self.get_coefficient_indices()]

    def evaluate(self, coeffs, bvec):
        from sumpy.tools import mi_power, mi_factorial
        return sum(
                coeff
                * self.kernel.postprocess_at_target(mi_power(bvec, mi), bvec)
                / mi_factorial(mi)
                for coeff, mi in zip(coeffs, self.get_coefficient_indices()))

# }}}

# vim: fdm=marker
