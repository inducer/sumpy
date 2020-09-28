import pyopencl as cl
import sumpy.toys as t
import numpy as np
from sumpy.visualization import FieldPlotter
import matplotlib.pyplot as plt


def main():
    from sumpy.kernel import (  # noqa: F401
            YukawaKernel, HelmholtzKernel, LaplaceKernel)
    tctx = t.ToyContext(
            cl.create_some_context(),
            #LaplaceKernel(2),
            YukawaKernel(2), extra_kernel_kwargs={"lam": 5},
            #HelmholtzKernel(2), extra_kernel_kwargs={"k": 0.3},
            )

    pt_src = t.PointSources(
            tctx,
            np.random.rand(2, 50) - 0.5,
            np.ones(50))

    fp = FieldPlotter([3, 0], extent=8)

    if 0:
        t.logplot(fp, pt_src, cmap="jet")
        plt.colorbar()
        plt.show()

    mexp = t.multipole_expand(pt_src, [0, 0], 5)
    mexp2 = t.multipole_expand(mexp, [0, 0.25])  # noqa: F841
    lexp = t.local_expand(mexp, [3, 0])
    lexp2 = t.local_expand(lexp, [3, 1], 3)

    #diff = mexp - pt_src
    #diff = mexp2 - pt_src
    diff = lexp2 - pt_src

    print(t.l_inf(diff, 1.2, center=lexp2.center))
    if 1:
        t.logplot(fp, diff, cmap="jet", vmin=-3, vmax=0)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()
