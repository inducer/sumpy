from __future__ import division

from pytools import memoize_method




# {{{ base class

class ExpansionBase(object):
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

# }}}

# {{{ volume taylor

class VolumeTaylorExpansionBase(object):
    @memoize_method
    def _storage_loc_dict(self):
        return dict((idx, i) for i, idx in enumerate(self.get_coefficient_indices()))

    def get_storage_index(self, k):
        return self._storage_loc_dict()[k]

    @memoize_method
    def get_coefficient_indices(self):
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        return sorted(gnitstam(self.order, self.kernel.dimensions), key=sum)

        return range(self.order+1)

# }}}




# vim: fdm=marker
