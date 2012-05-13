from __future__ import division

import sympy as sp





class LocalExpansionBase(object):
    def __init__(self, kernel, order):
        self.kernel = kernel
        self.order = order

    # {{{ propagate kernel interface

    @property
    def dimensions(self):
        return self.kernel.dimensions

    @property
    def is_complex(self):
        return self.kernel.is_complex

    def prepare_loopy_kernel(self, loopy_knl):
        return self.kernel.prepare_loopy_kernel(loopy_knl)

    def get_scaling(self):
        return self.kernel.get_scaling()

    def get_args(self):
        return self.kernel.get_args()

    def get_preambles(self):
        return self.kernel.get_preambles()

    # }}}

    def coefficients_from_source(self, expr, avec, bvec):
        """
        :arg bvec: vector from center to target. Not usually necessary,
            except for line-Taylor expansion.
        """
        raise NotImplementedError

    def evaluate(self, coeffs, bvec):
        raise NotImplementedError




class LineTaylorLocalExpansion(LocalExpansionBase):
    @property
    def coefficient_count(self):
        return self.order+1

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






class H2DLocalExpansion(LocalExpansionBase):
    def __init__(self, kernel, order):
        from sumpy.kernel import HelmholtzKernel
        assert (isinstance(kernel.get_base_kernel(), HelmholtzKernel)
                and kernel.dimensions == 2)

        LocalExpansionBase.__init__(self, kernel, order)

    @property
    def coefficient_count(self):
        return 2*self.order+1

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
                self.kernel.postprocess_at_target(
                    coeffs[self.get_storage_index(i)]
                    * bessel_j(i, sp.Symbol("k")*v)
                    * e_i_ctangle**i,
                    bvec)
                for i in self.get_coefficient_indices())
