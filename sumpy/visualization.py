from __future__ import division, absolute_import

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


import numpy as np
from six.moves import range


def separate_by_real_and_imag(data, real_only):
    for name, field in data:
        from pytools.obj_array import log_shape
        ls = log_shape(field)

        if ls != () and ls[0] > 1:
            assert len(ls) == 1
            from pytools.obj_array import (
                    oarray_real_copy, oarray_imag_copy,
                    with_object_array_or_scalar)

            if field[0].dtype.kind == "c":
                if real_only:
                    yield (name,
                            with_object_array_or_scalar(oarray_real_copy, field))
                else:
                    yield (name+"_r",
                            with_object_array_or_scalar(oarray_real_copy, field))
                    yield (name+"_i",
                            with_object_array_or_scalar(oarray_imag_copy, field))
            else:
                yield (name, field)
        else:
            # ls == ()
            if field.dtype.kind == "c":
                yield (name+"_r", field.real.copy())
                yield (name+"_i", field.imag.copy())
            else:
                yield (name, field)


class FieldPlotter:
    def __init__(self, center, extent=1, npoints=1000):
        center = np.asarray(center)
        self.dimensions, = dim, = center.shape
        self.a = a = center-extent*0.5
        self.b = b = center+extent*0.5

        if not isinstance(npoints, tuple):
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

        return self.a[non1_dims], self.b[non1_dims]

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
            squeezed_fld[squeezed_fld < -max_val] = -max_val

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

        from mayavi import mlab
        mlab.quiver3d(c[0], c[1], c[2], fld[0], fld[1], fld[2],
                **kwargs)

        if do_show:
            mlab.show()

    def write_vtk_file(self, file_name, data, real_only=False):
        from pyvisfile.vtk import write_structured_grid
        write_structured_grid(file_name, self.nd_points,
                point_data=list(separate_by_real_and_imag(data, real_only)))

    def show_scalar_in_mayavi(self, fld, max_val=None, **kwargs):
        if max_val is not None:
            fld[fld > max_val] = max_val
            fld[fld < -max_val] = -max_val

        if len(fld.shape) == 1:
            fld = fld.reshape(self.nd_points.shape[1:])

        nd_points = self.nd_points.squeeze()[self._get_nontrivial_dims()]
        squeezed_fld = fld.squeeze()

        from mayavi import mlab
        mlab.surf(nd_points[0], nd_points[1], squeezed_fld, **kwargs)

# vim: foldmethod=marker
