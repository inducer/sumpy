from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2017 Matt Wala"

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

import numpy as np
import sys

import pyopencl as cl

import logging
logger = logging.getLogger(__name__)


try:
    import faulthandler
except ImportError:
    pass
else:
    faulthandler.enable()


def test_direct(ctx_getter):
    # This evaluates a single layer potential on a circle.
    logging.basicConfig(level=logging.INFO)

    ctx = ctx_getter()
    queue = cl.CommandQueue(ctx)

    from sumpy.kernel import LaplaceKernel
    lknl = LaplaceKernel(2)

    order = 12

    from sumpy.qbx import LayerPotential
    from sumpy.expansion.local import LineTaylorLocalExpansion

    lpot = LayerPotential(ctx, [LineTaylorLocalExpansion(lknl, order)])

    mode_nr = 25

    from pytools.convergence import EOCRecorder

    eocrec = EOCRecorder()

    for n in [200, 300, 400]:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        unit_circle = np.exp(1j * t)
        unit_circle = np.array([unit_circle.real, unit_circle.imag])

        sigma = np.cos(mode_nr * t)
        eigval = 1/(2*mode_nr)

        result_ref = eigval * sigma

        h = 2 * np.pi / n

        targets = unit_circle
        sources = unit_circle

        radius = 7 * h
        centers = unit_circle * (1 - radius)

        strengths = (sigma * h,)
        evt, (result_qbx,) = lpot(queue, targets, sources, centers, strengths)

        eocrec.add_data_point(h, np.max(np.abs(result_ref - result_qbx)))

    print(eocrec)

    slack = 1.5
    assert eocrec.order_estimate() > order - slack


# You can test individual routines by typing
# $ python test_kernels.py 'test_p2p(cl.create_some_context)'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from py.test.cmdline import main
        main([__file__])

# vim: fdm=marker
