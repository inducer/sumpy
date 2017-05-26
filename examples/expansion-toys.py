import pyopencl as cl
import sumpy.toys as t
import numpy as np
from sumpy.visualization import FieldPlotter
import matplotlib.pyplot as plt


def main():
    from sumpy.kernel import LaplaceKernel
    tctx = t.ToyContext(cl.create_some_context(), LaplaceKernel(2))

    pt_src = t.PointSources(
            tctx,
            np.random.rand(2, 50) - 0.5,
            np.ones(50))

    fp = FieldPlotter([3, 0], extent=8)

    if 0:
        t.logplot(fp, pt_src, cmap="jet")
        plt.colorbar()
        plt.show()

    mexp = t.multipole_expand(pt_src, [0, 0], 9)
    mexp2 = t.multipole_expand(mexp, [0, 0.25])
    lexp = t.local_expand(mexp, [3, 0])
    lexp2 = t.local_expand(lexp, [3, 1])

    diff = mexp - pt_src
    diff = mexp2 - pt_src
    diff = lexp - pt_src

    if 1:
        t.logplot(fp, diff, cmap="jet", vmin=-3)
        plt.colorbar()
        plt.show()





if __name__ == "__main__":
    main()
