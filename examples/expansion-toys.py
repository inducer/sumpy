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

    fp = FieldPlotter([0, 0], extent=4)

    if 0:
        t.logplot(fp, pt_src, cmap="jet")
        plt.colorbar()
        plt.show()

    t.local_expand(pt_src, [3, 0], 5)


if __name__ == "__main__":
    main()
