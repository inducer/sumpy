from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

import sympy as sp  # noqa

from sumpy.expansion import ExpansionBase, VolumeTaylorExpansionBase


class MultipoleExpansionBase(ExpansionBase):
    pass


# {{{ volume taylor

class VolumeTaylorMultipoleExpansion(
        MultipoleExpansionBase, VolumeTaylorExpansionBase):
    def coefficients_from_source(self, avec, bvec):
        from sumpy.kernel import DirectionalSourceDerivative
        kernel = self.kernel

        from sumpy.tools import mi_power, mi_factorial

        if isinstance(kernel, DirectionalSourceDerivative):
            if kernel.get_base_kernel() is not kernel.inner_kernel:
                raise NotImplementedError("more than one source derivative "
                        "not supported at present")

            from sumpy.symbolic import make_sym_vector
            dir_vec = make_sym_vector(kernel.dir_vec_name, kernel.dim)

            coeff_identifiers = self.get_coefficient_identifiers()
            result = [0] * len(coeff_identifiers)

            for idim in xrange(kernel.dim):
                for i, mi in enumerate(coeff_identifiers):
                    if mi[idim] == 0:
                        continue

                    derivative_mi = tuple(mi_i - 1 if iaxis == idim else mi_i
                            for iaxis, mi_i in enumerate(mi))

                    result[i] += (
                        - 1/mi_factorial(derivative_mi)
                        * mi_power(avec, derivative_mi)
                        * dir_vec[idim])

            return result
        else:
            return [
                    mi_power(avec, mi) / mi_factorial(mi)
                    for mi in self.get_coefficient_identifiers()]

    def evaluate(self, coeffs, bvec):
        ppkernel = self.kernel.postprocess_at_target(
                self.kernel.get_expression(bvec), bvec)

        from sumpy.tools import mi_derivative
        return sum(
                coeff
                * mi_derivative(ppkernel, bvec, mi)
                for coeff, mi in zip(coeffs, self.get_coefficient_identifiers()))

# }}}

# vim: fdm=marker
