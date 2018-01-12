import pyopencl as cl
import sumpy.toys as t
import numpy as np
import loopy as lp
import numpy.linalg as la
from sumpy.visualization import FieldPlotter
import matplotlib.pyplot as plt
from sumpy.point_calculus import CalculusPatch
from sumpy.symbolic import pymbolic_real_norm_2
from pymbolic.primitives import make_sym_vector
from pymbolic import var

from sumpy.kernel import (  # noqa: F401
        YukawaKernel, HelmholtzKernel, LaplaceKernel, StokesKernel, ExpressionKernel, KernelArgument)
from sumpy.point_calculus import CalculusPatch

import logging
logging.basicConfig(level=logging.INFO)


def main():
    dim = 3
    mu = 5
    center = [3, 0, 0][:dim]
    order = 5
    points = np.random.rand(dim, 50) - 0.5
    weights = [np.ones(50)]*dim 
    cp = CalculusPatch(center, h=0.01, order=order)
    res = np.empty((cp.points.shape[1], dim, dim))
    f = np.array([1, 2, 3][:dim])

    tctx = t.ToyContext(
                    cl.create_some_context(),
                    StokesKernel(dim, "mu"), extra_kernel_kwargs={"mu": mu},
                    )
    pt_src = t.PointSources(tctx, points, weights)
    #mexp = t.multipole_expand(pt_src, [0, 0, 3], 5)
    lexp = t.local_expand(pt_src, [0.1, 0.1, 3.1], 1)
    p = lexp.eval(cp.points)

    pdes = [(mu * sum(cp.diff(i, p[j], 2) for i in range(dim)) - cp.diff(j, p[dim], 1)) for j in range(dim)]
    errs = [la.norm(pde) for pde in pdes]

    pde1 = sum(cp.diff(i, p[i]) for i in range(dim))
    err = la.norm(pde1)
    print(err)
    print(errs)

if __name__ == "__main__":
    main()
