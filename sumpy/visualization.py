from __future__ import division
import numpy as np




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
                    yield (name, with_object_array_or_scalar(oarray_real_copy, field))
                else:
                    yield (name+"_r", with_object_array_or_scalar(oarray_real_copy, field))
                    yield (name+"_i", with_object_array_or_scalar(oarray_imag_copy, field))
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
    def __init__(self, center, extent=1, points=1000):
        center = np.asarray(center)
        self.dimensions, = dim, = center.shape
        self.a = a = center-extent*0.5
        self.b = b = center+extent*0.5

        if not isinstance(points, tuple):
            points = dim*(points,)

        for i in range(dim):
            if points[i] == 1:
                a[i] = center[i]

        mgrid_index = tuple(
                slice(a[i], b[i], 1j*points[i])
                for i in range(dim))

        mgrid = np.mgrid[mgrid_index]
        self.nd_points_axis_first = mgrid.transpose(
                *((0,)+tuple(range(1, dim+1))[::-1]))

        # (point x idx, point y idx, ..., axis)
        self.nd_points = mgrid.transpose(
                *(tuple(range(1, dim+1))+(0,)))

        self.points = self.nd_points.reshape(-1, dim).copy()

        from pytools import product
        self.npoints = product(points)

    def get_target(self):
        from hellskitchen.discretization.target import PointsTarget
        return PointsTarget(self.points)

    def show_scalar_in_matplotlib(self, fld, maxval=None,
            func_name="imshow", **kwargs):
        if len(self.a) != 2:
            raise RuntimeError(
                    "matplotlib plotting requires 2D geometry")

        if len(fld.shape) == 1:
            fld = fld.reshape(self.nd_points.shape[:-1])

        fld = fld[:, ::-1]

        if maxval is not None:
            fld[fld>maxval] = maxval
            fld[fld<-maxval] = -maxval

        kwargs["extent"] = (
                # (left, right, bottom, top)
                self.a[0], self.b[0],
                self.a[1], self.b[1])

        import matplotlib.pyplot as pt
        return getattr(pt, func_name)(fld.T, **kwargs)

    def set_matplotlib_limits(self):
        import matplotlib.pyplot as pt
        pt.xlim((self.a[0], self.b[0]))
        pt.ylim((self.a[1], self.b[1]))

    def show_vector_in_mayavi(self, fld, do_show=True, **kwargs):
        c = self.points

        from mayavi import mlab
        mlab.quiver3d(c[:,0], c[:,1], c[:,2], fld[0], fld[1], fld[2],
                **kwargs)

        if do_show:
            mlab.show()

    def write_vtk_file(self, file_name, data, real_only=False):
        from pyvisfile.vtk import write_structured_grid
        write_structured_grid(file_name, self.nd_points_axis_first,
                point_data=list(separate_by_real_and_imag(data, real_only)))

    def show_scalar_in_mayavi(self, fld, maxval=None, **kwargs):
        if maxval is not None:
            fld[fld>maxval] = maxval
            fld[fld<-maxval] = -maxval

        if len(fld.shape) == 1:
            fld = fld.reshape(self.nd_points.shape[:-1])

        fld = fld[:, ::-1]

        from mayavi import mlab
        mlab.surf(self.nd_points[:,:,0], self.nd_points[:,:,1], fld, **kwargs)



# vim: foldmethod=marker
