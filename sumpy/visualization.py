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


__doc__ = """
.. autofunction:: make_field_plotter_from_bbox
.. autoclass:: FieldPlotter
"""

import numpy as np


def separate_by_real_and_imag(data, real_only):
    from pytools.obj_array import obj_array_imag_copy, obj_array_real_copy

    for name, field in data:
        try:
            # Look inside object arrays to get the entry dtype.
            entry_dtype = field[0].dtype
        except AttributeError:
            entry_dtype = field.dtype

        assert entry_dtype.kind != "O"

        if real_only or entry_dtype.kind != "c":
            yield (name, obj_array_real_copy(field))
        else:
            yield (f"{name}_r", obj_array_real_copy(field))
            yield (f"{name}_i", obj_array_imag_copy(field))


def make_field_plotter_from_bbox(bbox, h, extend_factor=0):
    """
    :arg bbox: a tuple (low, high) of points represented as 1D numpy arrays
        indicating the low and high ends of the extent of a bounding box.
    :arg h: Either a number or a sequence of numbers indicating the desired
        (approximate) grid spacing in all or each of the dimensions. If a
        sequence, the length must match the number of dimensions.
    :arg extend_factor: A floating point number indicating by what percentage
        the plot area should be grown compared to *bbox*.
    """
    low, high = bbox

    extent = (high-low) * (1 + extend_factor)
    center = 0.5*(high+low)

    dimensions = len(center)
    from numbers import Number
    if isinstance(h, Number):
        h = (h,)*dimensions
    else:
        if len(h) != dimensions:
            raise ValueError("length of 'h' must match number of dimensions")

    from math import ceil

    npoints = tuple(
            int(ceil(extent[i] / h[i]))
            for i in range(dimensions))

    return FieldPlotter(center, extent, npoints)


class FieldPlotter:
    """
    .. automethod:: set_matplotlib_limits
    .. automethod:: show_scalar_in_matplotlib
    .. automethod:: show_scalar_in_mayavi
    .. automethod:: write_vtk_file
    """
    def __init__(self, center, extent=1, npoints=1000):
        center = np.asarray(center)
        self.dimensions, = dim, = center.shape
        self.a = a = center-extent*0.5
        self.b = b = center+extent*0.5

        from numbers import Number
        if isinstance(npoints, Number):
            npoints = dim*(npoints,)
        else:
            if len(npoints) != dim:
                raise ValueError("length of npoints must match dimension")

        for i in range(dim):
            if npoints[i] == 1:
                a[i] = center[i]

        mgrid_index = tuple(
                slice(a[i], b[i], 1j*npoints[i])
                for i in range(dim))

        mgrid = np.mgrid[mgrid_index]

        # (axis, point x idx, point y idx, ...)
        self.nd_points = mgrid

        self.points = self.nd_points.reshape(dim, -1).copy()

        from pytools import product
        self.npoints = product(npoints)

    def _get_nontrivial_dims(self):
        return np.array(self.nd_points.shape[1:]) != 1

    def _get_squeezed_bounds(self):
        nontriv_dims = self._get_nontrivial_dims()

        return self.a[nontriv_dims], self.b[nontriv_dims]

    def show_scalar_in_matplotlib(self, fld, max_val=None,
            func_name="imshow", **kwargs):
        squeezed_points = self.points.squeeze()

        if len(squeezed_points.shape) != 2:
            raise RuntimeError(
                    "matplotlib plotting requires 2D geometry")

        if len(fld.shape) == 1:
            fld = fld.reshape(self.nd_points.shape[1:])

        squeezed_fld = fld.squeeze()

        if max_val is not None:
            squeezed_fld[squeezed_fld > max_val] = max_val
            squeezed_fld[squeezed_fld < -max_val] = -max_val  # pylint: disable=E1130

        squeezed_fld = squeezed_fld[..., ::-1]

        a, b = self._get_squeezed_bounds()

        kwargs["extent"] = (
                # (left, right, bottom, top)
                a[0], b[0],
                a[1], b[1])

        import matplotlib.pyplot as pt
        return getattr(pt, func_name)(squeezed_fld.T, **kwargs)

    def set_matplotlib_limits(self):
        import matplotlib.pyplot as pt

        a, b = self._get_squeezed_bounds()
        pt.xlim((a[0], b[0]))
        pt.ylim((a[1], b[1]))

    def show_vector_in_mayavi(self, fld, do_show=True, **kwargs):
        c = self.points

        from mayavi import mlab  # pylint: disable=import-error

        mlab.quiver3d(c[0], c[1], c[2], fld[0], fld[1], fld[2],
                **kwargs)

        if do_show:
            mlab.show()

    def write_vtk_file(self, file_name, data, real_only=False, overwrite=False):
        from pyvisfile.vtk import write_structured_grid
        write_structured_grid(file_name, self.nd_points,
                point_data=list(separate_by_real_and_imag(data, real_only)),
                overwrite=overwrite)

    def show_scalar_in_mayavi(self, fld, max_val=None, **kwargs):
        if max_val is not None:
            fld[fld > max_val] = max_val
            fld[fld < -max_val] = -max_val  # pylint: disable=E1130

        if len(fld.shape) == 1:
            fld = fld.reshape(self.nd_points.shape[1:])

        nd_points = self.nd_points.squeeze()[self._get_nontrivial_dims()]
        squeezed_fld = fld.squeeze()

        from mayavi import mlab  # pylint: disable=import-error
        mlab.surf(nd_points[0], nd_points[1], squeezed_fld, **kwargs)

# vim: foldmethod=marker
