import numpy as np

import pyopencl as cl

from sumpy import toys
from sumpy.visualization import FieldPlotter
from sumpy.kernel import (      # noqa: F401
        YukawaKernel,
        HeatKernel,
        HelmholtzKernel,
        LaplaceKernel)

try:
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except ImportError:
    USE_MATPLOTLIB = False


def main():
    from sumpy.array_context import PyOpenCLArrayContext
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue, force_device_scalars=True)

    tctx = toys.ToyContext(
            actx.context,
            # LaplaceKernel(2),
            #YukawaKernel(2), extra_kernel_kwargs={"lam": 5},
            # HelmholtzKernel(2), extra_kernel_kwargs={"k": 0.3},
            HeatKernel(1), extra_kernel_kwargs={"alpha": 1},
            )

    src_size = 1
    pt_src = toys.PointSources(
            tctx,
            np.array([
                src_size*(np.random.rand(50) - 0.5),
                np.zeros(50)]),
            np.random.randn(50))

    fp = FieldPlotter([0, 0.5], extent=np.array([8, 1]))

    if 0 and USE_MATPLOTLIB:
        toys.logplot(fp, pt_src, cmap="jet", aspect=8)
        plt.colorbar()
        plt.show()


    p = 5
    mexp = toys.multipole_expand(pt_src, [0, 0], p)
    diff = mexp - pt_src

    x, t = fp.points

    r = np.sqrt(x**2+t**2)

    conv_factor = (src_size/r)**(p+1)

    def logplot_fp(fp: FieldPlotter, values, **kwargs) -> None:
        fp.show_scalar_in_matplotlib(
                np.log10(np.abs(values + 1e-15)), **kwargs)
    if USE_MATPLOTLIB:
        logplot_fp(fp, diff.eval(fp.points)/conv_factor, cmap="jet", vmin=-5, vmax=0, aspect=8)
        plt.colorbar()
        plt.show()
    1/0
    mexp2 = toys.multipole_expand(mexp, [0, 0.25])  # noqa: F841
    lexp = toys.local_expand(mexp, [3, 0])
    lexp2 = toys.local_expand(lexp, [3, 1], 3)

    # diff = mexp - pt_src
    # diff = mexp2 - pt_src
    diff = lexp2 - pt_src

    print(toys.l_inf(diff, 1.2, center=lexp2.center))


if __name__ == "__main__":
    main()
