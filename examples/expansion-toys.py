import numpy as np

import pyopencl as cl

import sumpy.toys as t
from sumpy.kernel import HelmholtzKernel, LaplaceKernel, YukawaKernel  # noqa: F401
from sumpy.visualization import FieldPlotter


try:
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except ImportError:
    USE_MATPLOTLIB = False


def main():
    from sumpy.array_context import PyOpenCLArrayContext
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    actx = PyOpenCLArrayContext(queue)

    tctx = t.ToyContext(
            # LaplaceKernel(2),
            YukawaKernel(2), extra_kernel_kwargs={"lam": 5},
            # HelmholtzKernel(2), extra_kernel_kwargs={"k": 0.3},
            )

    rng = np.random.default_rng()
    pt_src = t.PointSources(
            tctx,
            rng.uniform(size=(2, 50)) - 0.5,
            np.ones(50))

    fp = FieldPlotter([3, 0], extent=8)

    if USE_MATPLOTLIB:
        t.logplot(actx, fp, pt_src, cmap="jet")
        plt.colorbar()
        plt.show()

    mexp = t.multipole_expand(actx, pt_src, [0, 0], order=5)
    mexp2 = t.multipole_expand(actx, mexp, [0, 0.25])  # noqa: F841
    lexp = t.local_expand(actx, mexp, [3, 0])
    lexp2 = t.local_expand(actx, lexp, [3, 1], order=3)

    # diff = mexp - pt_src
    # diff = mexp2 - pt_src
    diff = lexp2 - pt_src

    print(t.l_inf(actx, diff, 1.2, center=lexp2.center))
    if USE_MATPLOTLIB:
        t.logplot(actx, fp, diff, cmap="jet", vmin=-3, vmax=0)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()
