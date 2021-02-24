import pyopencl as cl
import numpy as np
import numpy.linalg as la

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from mayavi import mlab
except ModuleNotFoundError:
    mlab = None


def process_kernel(knl, what_operator):
    target_knl = knl
    source_knl = knl
    if what_operator == "S":
        pass
    elif what_operator == "S0":
        from sumpy.kernel import TargetDerivative
        target_knl = TargetDerivative(0, knl)
    elif what_operator == "S1":
        from sumpy.kernel import TargetDerivative
        target_knl = TargetDerivative(1, knl)
    elif what_operator == "D":
        from sumpy.kernel import DirectionalSourceDerivative
        source_knl = DirectionalSourceDerivative(knl)
    # DirectionalTargetDerivative (temporarily?) removed
    # elif what_operator == "S'":
    #     from sumpy.kernel import DirectionalTargetDerivative
    #     knl = DirectionalTargetDerivative(knl)
    else:
        raise RuntimeError("unrecognized operator '%s'" % what_operator)

    return source_knl, target_knl


def draw_pot_figure(aspect_ratio,
        nsrc=100, novsmp=None, helmholtz_k=0,
        what_operator="S",
        what_operator_lpot=None,
        order=4,
        ovsmp_center_exp=0.66,
        force_center_side=None):

    import logging
    logging.basicConfig(level=logging.INFO)

    if novsmp is None:
        novsmp = 4*nsrc

    if what_operator_lpot is None:
        what_operator_lpot = what_operator

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # {{{ make plot targets

    center = np.asarray([0, 0], dtype=np.float64)
    from sumpy.visualization import FieldPlotter
    fp = FieldPlotter(center, npoints=1000, extent=6)

    # }}}

    # {{{ make p2p kernel calculator

    from sumpy.p2p import P2P
    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    from sumpy.expansion.local import H2DLocalExpansion, LineTaylorLocalExpansion
    if helmholtz_k:
        if isinstance(helmholtz_k, complex):
            knl = HelmholtzKernel(2, allow_evanescent=True)
            expn_class = H2DLocalExpansion
            knl_kwargs = {"k": helmholtz_k}
        else:
            knl = HelmholtzKernel(2)
            expn_class = H2DLocalExpansion
            knl_kwargs = {"k": helmholtz_k}

    else:
        knl = LaplaceKernel(2)
        expn_class = LineTaylorLocalExpansion
        knl_kwargs = {}

    vol_source_knl, vol_target_knl = process_kernel(knl, what_operator)
    p2p = P2P(ctx, source_kernels=(vol_source_knl,),
            target_kernels=(vol_target_knl,),
            exclude_self=False,
            value_dtypes=np.complex128)

    lpot_source_knl, lpot_target_knl = process_kernel(knl, what_operator_lpot)

    from sumpy.qbx import LayerPotential
    lpot = LayerPotential(ctx, expansion=expn_class(knl, order=order),
            source_kernels=(lpot_source_knl,), target_kernels=(lpot_target_knl,),
            value_dtypes=np.complex128)

    # }}}

    # {{{ set up geometry

    # r,a,b match the corresponding letters from G. J. Rodin and O. Steinbach,
    # Boundary Element Preconditioners for problems defined on slender domains.
    # https://dx.doi.org/10.1137/S1064827500372067

    a = 1
    b = 1/aspect_ratio

    def map_to_curve(t):
        t = t*(2*np.pi)

        x = a*np.cos(t)
        y = b*np.sin(t)

        w = (np.zeros_like(t)+1)/len(t)

        return x, y, w

    from curve import CurveGrid

    native_t = np.linspace(0, 1, nsrc, endpoint=False)
    native_x, native_y, native_weights = map_to_curve(native_t)
    native_curve = CurveGrid(native_x, native_y)

    ovsmp_t = np.linspace(0, 1, novsmp, endpoint=False)
    ovsmp_x, ovsmp_y, ovsmp_weights = map_to_curve(ovsmp_t)
    ovsmp_curve = CurveGrid(ovsmp_x, ovsmp_y)

    curve_len = np.sum(ovsmp_weights * ovsmp_curve.speed)
    hovsmp = curve_len/novsmp
    center_dist = 5*hovsmp

    if force_center_side is not None:
        center_side = force_center_side*np.ones(len(native_curve))
    else:
        center_side = -np.sign(native_curve.mean_curvature)

    centers = (native_curve.pos
            + center_side[:, np.newaxis]
            * center_dist*native_curve.normal)

    #native_curve.plot()
    #plt.show()

    volpot_kwargs = knl_kwargs.copy()
    lpot_kwargs = knl_kwargs.copy()

    if what_operator == "D":
        volpot_kwargs["src_derivative_dir"] = native_curve.normal

    if what_operator_lpot == "D":
        lpot_kwargs["src_derivative_dir"] = ovsmp_curve.normal

    if what_operator_lpot == "S'":
        lpot_kwargs["tgt_derivative_dir"] = native_curve.normal

    # }}}

    if 0:
        # {{{ build matrix

        from fourier import make_fourier_interp_matrix
        fim = make_fourier_interp_matrix(novsmp, nsrc)
        from sumpy.tools import build_matrix
        from scipy.sparse.linalg import LinearOperator

        def apply_lpot(x):
            xovsmp = np.dot(fim, x)
            evt, (y,) = lpot(queue, native_curve.pos, ovsmp_curve.pos,
                    centers,
                    [xovsmp * ovsmp_curve.speed * ovsmp_weights],
                    expansion_radii=np.ones(centers.shape[1]),
                    **lpot_kwargs)

            return y

        op = LinearOperator((nsrc, nsrc), apply_lpot)
        mat = build_matrix(op, dtype=np.complex128)
        w, v = la.eig(mat)
        plt.plot(w.real, "o-")
        #import sys; sys.exit(0)
        return

        # }}}

    # {{{ compute potentials

    mode_nr = 0
    density = np.cos(mode_nr*2*np.pi*native_t).astype(np.complex128)
    ovsmp_density = np.cos(mode_nr*2*np.pi*ovsmp_t).astype(np.complex128)
    evt, (vol_pot,) = p2p(queue, fp.points, native_curve.pos,
            [native_curve.speed*native_weights*density], **volpot_kwargs)

    evt, (curve_pot,) = lpot(queue, native_curve.pos, ovsmp_curve.pos,
            centers,
            [ovsmp_density * ovsmp_curve.speed * ovsmp_weights],
            expansion_radii=np.ones(centers.shape[1]),
            **lpot_kwargs)

    # }}}

    if 0:
        # {{{ plot on-surface potential in 2D

        plt.plot(curve_pot, label="pot")
        plt.plot(density, label="dens")
        plt.legend()
        plt.show()

        # }}}

    fp.write_vtk_file("potential.vts", [
        ("potential", vol_pot.real)
        ])

    if 0:
        # {{{ 2D false-color plot

        plt.clf()
        plotval = np.log10(1e-20+np.abs(vol_pot))
        im = fp.show_scalar_in_matplotlib(plotval.real)
        from matplotlib.colors import Normalize
        im.set_norm(Normalize(vmin=-2, vmax=1))

        src = native_curve.pos
        plt.plot(src[:, 0], src[:, 1], "o-k")
        # close the curve
        plt.plot(src[-1::-len(src)+1, 0], src[-1::-len(src)+1, 1], "o-k")

        #plt.gca().set_aspect("equal", "datalim")
        cb = plt.colorbar(shrink=0.9)
        cb.set_label(r"$\log_{10}(\mathdefault{Error})$")
        #from matplotlib.ticker import NullFormatter
        #plt.gca().xaxis.set_major_formatter(NullFormatter())
        #plt.gca().yaxis.set_major_formatter(NullFormatter())
        fp.set_matplotlib_limits()

        # }}}
    else:
        # {{{ 3D plots

        plotval_vol = vol_pot.real
        plotval_c = curve_pot.real

        scale = 1

        if 0:
            # crop singularities--doesn't work very well
            neighbors = [
                    np.roll(plotval_vol, 3, 0),
                    np.roll(plotval_vol, -3, 0),
                    np.roll(plotval_vol, 6, 0),
                    np.roll(plotval_vol, -6, 0),
                    ]
            avg = np.average(np.abs(plotval_vol))
            outlier_flag = sum(
                    np.abs(plotval_vol-nb) for nb in neighbors) > avg
            plotval_vol[outlier_flag] = sum(
                    nb[outlier_flag] for nb in neighbors)/len(neighbors)

        if mlab is not None:
            fp.show_scalar_in_mayavi(scale*plotval_vol, max_val=1)
            mlab.colorbar()
            if 1:
                mlab.points3d(
                        native_curve.pos[0],
                        native_curve.pos[1],
                        scale*plotval_c, scale_factor=0.02)
            mlab.show()

        # }}}


if __name__ == "__main__":
    draw_pot_figure(aspect_ratio=1, nsrc=100, novsmp=100, helmholtz_k=(35+4j)*0.3,
            what_operator="D", what_operator_lpot="D", force_center_side=1)

#    plt.savefig("eigvals-ext-nsrc100-novsmp100.pdf")
    #plt.clf()
    #draw_pot_figure(aspect_ratio=1, nsrc=100, novsmp=100, helmholtz_k=0,
    #        what_operator="D", what_operator_lpot="D", force_center_side=-1)
    #plt.savefig("eigvals-int-nsrc100-novsmp100.pdf")
    #plt.clf()
    #draw_pot_figure(aspect_ratio=1, nsrc=100, novsmp=200, helmholtz_k=0,
    #        what_operator="D", what_operator_lpot="D", force_center_side=-1)
    #plt.savefig("eigvals-int-nsrc100-novsmp200.pdf")

# vim: fdm=marker
