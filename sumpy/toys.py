from __future__ import division, absolute_import

__copyright__ = "Copyright (C) 2017 Andreas Kloeckner"

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

import six
from six.moves import range

import numpy as np
import loopy as lp

import logging
logger = logging.getLogger(__name__)


class ToyContext(object):
    def __init__(self, cl_context, kernel):
        self.code_cache = {}


class PotentialSource(object):
    def __init__(self, toy_ctx, center):
        self.toy_ctx = toy_ctx
        self.center

    def eval(self, toy_ctx, targets):
        raise NotImplementedError()

    def plot(self, extent):
        pass


class PointSources(PotentialSource):
    """
    .. attribute:: points

        ``[ndim, npoints]``
    """

    def __init__(self, toy_ctx, points, weights, center=None):
        super(PointSources, self).__init__(toy_ctx, center)


class Multipole(PotentialSource):
    def __init__(self, toy_ctx, center, coeffs):
        super(Multipole, self).__init__(toy_ctx)


class Local(PotentialSource):
    def __init__(self, toy_ctx, center, coeffs):
        super(Local, self).__init__(toy_ctx)
