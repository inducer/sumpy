from __future__ import annotations


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

from typing import TYPE_CHECKING, Any, final

import numpy as np


if TYPE_CHECKING:
    import optype.numpy as onp


def fftdiff(x: onp.Array1D[np.floating[Any]], *,
            period: float = 1.0) -> onp.Array1D[np.floating[Any]]:
    n = len(x)
    return np.fft.ifft(
        2j * np.pi * np.fft.fftfreq(n, d=period/n) * np.fft.fft(x)
    ).real


@final
class CurveGrid:
    pos: onp.Array2D[np.floating[Any]]
    mean_curvature: onp.Array1D[np.floating[Any]]
    normal: onp.Array2D[np.floating[Any]]

    def __init__(self,
                 x: onp.Array1D[np.floating[Any]],
                 y: onp.Array1D[np.floating[Any]]) -> None:
        self.pos = np.vstack([x, y]).copy()
        xp = self.xp = fftdiff(x, period=1)
        yp = self.yp = fftdiff(y, period=1)
        xpp = self.xpp = fftdiff(xp, period=1)
        ypp = self.ypp = fftdiff(yp, period=1)
        self.mean_curvature = (xp*ypp-yp*xpp)/((xp**2+yp**2)**(3/2))

        speed = self.speed = np.sqrt(xp**2+yp**2)
        self.normal = (np.vstack([yp, -xp])/speed).copy()

    def __len__(self) -> int:
        return len(self.pos)

    def plot(self) -> None:
        import matplotlib.pyplot as pt
        pt.plot(self.pos[:, 0], self.pos[:, 1])
