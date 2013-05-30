from __future__ import division

import sympy as sp

from sumpy.expansion import ExpansionBase, VolumeTaylorExpansionBase





class LocalExpansionBase(ExpansionBase):
    pass

# {{{ line taylor

class LineTaylorLocalExpansion(LocalExpansionBase):
    def get_storage_index(self, k):
        return k

    def get_coefficient_indices(self):
        return range(self.order+1)

    def coefficients_from_source(self, avec, bvec):
        if bvec is None:
            raise RuntimeError("cannot use line-Taylor expansions in a setting "
                    "where the center-target vector is not known at coefficient "
                    "formation")
        avec_line = avec + sp.Symbol("tau")*bvec

        line_kernel = self.kernel.get_expression(avec_line)

        return [
                self.kernel.postprocess_at_target(
                    self.kernel.postprocess_at_source(
                        line_kernel.diff("tau", i),
                        avec),
                    bvec)
                .subs("tau", 0)
                for i in self.get_coefficient_indices()]

    def evaluate(self, coeffs, bvec):
        from pytools import factorial
        return sum(
                coeffs[self.get_storage_index(i)] / factorial(i)
                for i in self.get_coefficient_indices())

# }}}

# {{{ volume taylor

class VolumeTaylorLocalExpansion(LocalExpansionBase, VolumeTaylorExpansionBase):
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

# {{{ 2D J-expansion

class H2DLocalExpansion(LocalExpansionBase):
    def __init__(self, kernel, order):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dimensions == 2)

        LocalExpansionBase.__init__(self, kernel, order)

    def get_storage_index(self, k):
        return self.order+k

    def get_coefficient_indices(self):
        return range(-self.order, self.order+1)

    def coefficients_from_source(self, avec, bvec):
        hankel_1 = sp.Function("hankel_1")
        center_source_angle = sp.atan2(-avec[1], -avec[0])

        from sumpy.symbolic import sympy_real_norm_2
        u = sympy_real_norm_2(avec)

        e_i_csangle = sp.exp(sp.I*center_source_angle)
        return [
                self.kernel.postprocess_at_source(
                    hankel_1(i, sp.Symbol("k")*u)*e_i_csangle**i,
                    avec)
                    for i in self.get_coefficient_indices()]

    def evaluate(self, coeffs, bvec):
        bessel_j = sp.Function("bessel_j")

        from sumpy.symbolic import sympy_real_norm_2
        v = sympy_real_norm_2(bvec)

        center_target_angle = sp.atan2(bvec[1], bvec[0])

        e_i_ctangle = sp.exp(-sp.I*center_target_angle)
        return sum(
                    coeffs[self.get_storage_index(i)]
                    * self.kernel.postprocess_at_target(
                        bessel_j(i, sp.Symbol("k")*v)
                        * e_i_ctangle**i, bvec)
                for i in self.get_coefficient_indices())

# }}}

# vim: fdm=marker
