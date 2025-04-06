import numpy as np
import numpy.linalg as la

import pyopencl as cl


def show_scalar_in_matplotlib(self, fld, max_val=None, func_name="imshow", **kwargs):
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

import matplotlib.pyplot as plt
from sumpy.visualization import FieldPlotter
center = np.asarray([0, 0], dtype=np.float64)
fp = FieldPlotter(center, npoints=1000, extent=6)

plt.clf()
vol_pot = np.outer(-0.3**np.arange(1, 100), 1.3**np.arange(1, 100))
plotval = np.log10(1e-20+np.abs(vol_pot))
im = fp.show_scalar_in_matplotlib(plotval.real)
from matplotlib.colors import Normalize
im.set_norm(Normalize(vmin=-2, vmax=1))

cb = plt.colorbar(shrink=0.9)
cb.set_label(r"$\log_{10}(\mathdefault{Error})$")
fp.set_matplotlib_limits()

plt.show()