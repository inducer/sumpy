import pyopencl as cl
import sumpy.toys as t
import numpy as np
from sumpy.visualization import FieldPlotter
import matplotlib.pyplot as plt


# FIXME: Get rid of this once everything is working

def main():
    dim = 2
    order = 10

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
    mctr1 = scale*np.array([0.2, 0, 0])[:dim]
    mexp1 = t.multipole_expand(pt_src, mctr1, order=order, rscale=scale)
    mexp = t.multipole_expand(mexp1, mctr, order=order, rscale=2*scale)

    lctr1 = scale*np.array([2.8, 0, 0])[:dim]
    lctr = scale*np.array([2.5, 0, 0])[:dim]
    lexp1 = t.local_expand(mexp, lctr1, order=order, rscale=scale)
    lexp = t.local_expand(lexp1, lctr, order=order, rscale=2*scale)

    #print(mexp.coeffs)
    print(lexp.coeffs)

    diff = lexp - pt_src

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
