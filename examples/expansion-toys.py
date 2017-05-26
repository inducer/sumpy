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

    lexp = t.multipole_expand(pt_src, [0, 0], 5)

    diff = lexp - pt_src

    if 1:
        t.logplot(fp, diff, cmap="jet", vmin=-3)
        plt.colorbar()
        plt.show()





if __name__ == "__main__":
    main()
