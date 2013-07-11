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

from pytools import memoize_method


# {{{ base class

class ExpansionBase(object):
    def __init__(self, kernel, order):
        from sumpy.kernel import TargetDerivativeRemover
        kernel = TargetDerivativeRemover()(kernel)

        self.kernel = kernel
        self.order = order

    # {{{ propagate kernel interface

    @property
    def dim(self):
        return self.kernel.dim

    @property
    def is_complex_valued(self):
        return self.kernel.is_complex_valued

    def prepare_loopy_kernel(self, loopy_knl):
        return self.kernel.prepare_loopy_kernel(loopy_knl)

    def transform_to_code(self, expr):
        return self.kernel.transform_to_code(expr)

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
    def get_coefficient_identifiers(self):
        from pytools import (
                generate_nonnegative_integer_tuples_summing_to_at_most
                as gnitstam)

        return sorted(gnitstam(self.order, self.kernel.dim), key=sum)

# }}}


# vim: fdm=marker
