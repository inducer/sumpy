import pyopencl as cl
import sumpy.toys as t
import numpy as np
from sumpy.visualization import FieldPlotter
import matplotlib.pyplot as plt


# FIXME: Get rid of this once everything is working

def main():
    dim = 2
    order = 7

    from sumpy.kernel import (  # noqa: F401
            YukawaKernel, HelmholtzKernel, LaplaceKernel,
            BiharmonicKernel, StokesletKernel, StressletKernel,
            AxisTargetDerivative)
    tctx = t.ToyContext(
            cl.create_some_context(),
            LaplaceKernel(dim),
            #AxisTargetDerivative(0, LaplaceKernel(dim)),
            #YukawaKernel(dim), extra_source_kwargs={"lam": 5},
            #HelmholtzKernel(dim), extra_source_kwargs={"k": 0.3},
            #BiharmonicKernel(dim),
            #StokesletKernel(dim, 1, 1), extra_source_kwargs={"mu": 0.3},
            #StressletKernel(dim, 1, 1, 0), extra_source_kwargs={"mu": 0.3},
            )

    np.random.seed(12)
    scale = 2**(-14)
    #scale = 1
    pts = np.random.rand(dim, 50) - 0.5
    pt_src = t.PointSources(
            tctx,
            scale * pts,
            np.ones(50))

    mctr = scale*np.array([0., 0, 0])[:dim]
    mexp = t.multipole_expand(pt_src, mctr, order=order, rscale=scale)

    lctr = scale*np.array([2.5, 0, 0])[:dim]
    #lexp = t.local_expand(pt_src, lctr, order=order, rscale=scale)

    exp = mexp
    print(exp.coeffs)
    #print(lexp.coeffs)

    diff = exp - pt_src

    diag = np.sqrt(dim)
    print(t.l_inf(diff, scale*0.5*diag, center=lctr)
            / t.l_inf(pt_src, scale*0.5*diag, center=lctr))
    if 0:
        fp = FieldPlotter(lexp.center, extent=scale*0.5)

        t.logplot(fp, diff, cmap="jet", vmin=-16, vmax=-7)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    main()
