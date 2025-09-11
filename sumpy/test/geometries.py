from __future__ import annotations


__copyright__ = "Copyright (C) 2025 University of Illinois Board of Trustees"

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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.linalg as la


if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Geometry:
    nodes: NDArray[np.floating]
    normals: NDArray[np.floating]
    weights: NDArray[np.floating]
    area_elements: NDArray[np.floating]


def make_ellipsoid(a: float = 1, b: float = 0.5, npoints: int = 100):
    def map_to_curve(t: NDArray[np.floating]):
        t = t*(2*np.pi)

        x = a*np.cos(t)
        y = b*np.sin(t)

        w = (np.zeros_like(t)+1)/len(t)

        return x, y, w

    from sumpy.test.curve import CurveGrid

    t = np.linspace(0, 1, npoints, endpoint=False)
    x, y, weights = map_to_curve(t)
    curve = CurveGrid(x, y)

    return Geometry(
        nodes=curve.pos,
        normals=curve.normal,
        weights=weights,
        area_elements=curve.speed,
    )


def make_torus(
            r_major: float = 1,
            r_minor: float = 0.5,
            n_major: int = 200,
            n_minor: int = 200
        ):
    u = np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)
    v = np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)
    u, v = np.meshgrid(u, v, copy=False)
    nodes = np.stack([
        np.cos(u) * (r_major + r_minor * np.cos(v)),
        np.sin(u) * (r_major + r_minor * np.cos(v)),
        r_minor * np.sin(v)
        ])

    def diff2d(ary: NDArray[np.floating]):
        import scipy.fftpack as fftpack
        return np.array([
                            fftpack.diff(ary[idx])
                            for idx in np.ndindex(ary.shape[:-1])
                        ])
    dnodes_du = np.array(
                    [diff2d(nodes[i].T).T for i in range(3)])
    dnodes_dv = np.array(
                    [diff2d(nodes[i]) for i in range(3)])

    normal = -np.cross(dnodes_du, dnodes_dv, axis=0)
    area_el = la.norm(normal, axis=0)
    normal /= area_el

    weights = np.zeros_like(area_el) + (2*np.pi)**2/n_major/n_minor

    if 0:
        import matplotlib.pyplot as plt
        plt.figure().add_subplot(projection="3d")
        plt.gca().quiver(
                nodes[0], nodes[1], nodes[2],
                normal[0], normal[1], normal[2],
                length=0.1
            )
        plt.show()
        1/0  # noqa: B018

    geo = Geometry(
        nodes=nodes.reshape(3, -1).copy(),
        normals=normal.reshape(3, -1).copy(),
        area_elements=area_el.reshape(-1).copy(),
        weights=weights.reshape(-1).copy(),
    )

    surface_area_ref = 4*np.pi**2*r_major*r_minor
    surface_area = geo.area_elements @ geo.weights
    surface_area_err = abs(surface_area - surface_area_ref)/surface_area_ref
    assert surface_area_err < 1e-14

    return geo
