import numpy as np
import pyopencl as cl
import loopy as lp
from sumpy.kernel import LaplaceKernel
from sumpy.expansion.local import (
        LaplaceConformingVolumeTaylorLocalExpansion,
        )
from sumpy.expansion.multipole import (
        LaplaceConformingVolumeTaylorMultipoleExpansion,
        )
from sumpy.e2e import E2EFromCSR


def find_flops():
    ctx = cl.create_some_context()

    knl = LaplaceKernel(2)

    orders = list(range(1, 21, 1))
    flop_counts = []
    for order in orders:
        print(order)
        m_expn = LaplaceConformingVolumeTaylorMultipoleExpansion(knl, order)
        l_expn = LaplaceConformingVolumeTaylorLocalExpansion(knl, order)
        m2l = E2EFromCSR(ctx, m_expn, l_expn)

        loopy_knl = m2l.get_kernel()
        loopy_knl = lp.add_and_infer_dtypes(
                loopy_knl,
                {
                    "target_boxes,src_box_lists,src_box_starts": np.int32,
                    "centers,src_expansions": np.float64,
                    })

        flops = lp.get_op_map(loopy_knl).filter_by(dtype=[np.float64]).sum()
        flop_counts.append(
                flops.eval_with_dict(
                    dict(isrc_start=0, isrc_stop=1, ntgt_boxes=1)))

    print(orders)
    print(flop_counts)


def plot_flops():
    if 0:
        case = "3D Laplace M2L"
        orders = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        flops = [62, 300, 914, 2221, 4567, 8405, 14172, 22538, 34113]

    elif 1:
        case = "2D Laplace M2L"
        orders = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20]
        flops = [36, 99, 193, 319, 476, 665, 889, 1143, 1429, 1747, 2097, 2479, 2893,
                3339, 3817, 4327, 4869, 5443, 6049, 6687]

    import matplotlib.pyplot as plt
    plt.rc("font", size=16)
    plt.title(case)
    plt.ylabel("Flop count")
    plt.xlabel("Expansion order")
    plt.loglog(orders, flops, "o-")
    plt.grid()
    plt.tight_layout()
    plt.savefig("laplace-m2l-complexity-2d.pdf")



if __name__ == "__main__":
    #find_flops()
    plot_flops()
