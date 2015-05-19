from __future__ import division
import pyopencl as cl
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pt


def process_kernel(knl, what_operator):
    if what_operator == "S":
        pass
    elif what_operator == "S0":
        from sumpy.kernel import TargetDerivative
        knl = TargetDerivative(0, knl)
    elif what_operator == "S1":
        from sumpy.kernel import TargetDerivative
        knl = TargetDerivative(1, knl)
    elif what_operator == "D":
        from sumpy.kernel import DirectionalSourceDerivative
        knl = DirectionalSourceDerivative(knl)
    # DirectionalTargetDerivative (temporarily?) removed
    # elif what_operator == "S'":
    #     from sumpy.kernel import DirectionalTargetDerivative
    #     knl = DirectionalTargetDerivative(knl)
    else:
        raise RuntimeError("unrecognized operator '%s'" % what_operator)

    return knl


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

    vol_knl = process_kernel(knl, what_operator)
    p2p = P2P(ctx, [vol_knl], exclude_self=False,
            value_dtypes=np.complex128)

    lpot_knl = process_kernel(knl, what_operator_lpot)

    from sumpy.qbx import LayerPotential
    lpot = LayerPotential(ctx, [
        expn_class(lpot_knl, order=order)],
            value_dtypes=np.complex128)

    # }}}

    # {{{ set up geometry

    # r,a,b match the corresponding letters from G. J. Rodin and O. Steinbach,
    # Boundary Element Preconditioners for problems defined on slender domains.
    # http://dx.doi.org/10.1137/S1064827500372067

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
    #pt.show()

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
                    **lpot_kwargs)

            return y

        op = LinearOperator((nsrc, nsrc), apply_lpot)
        mat = build_matrix(op, dtype=np.complex128)
        w, v = la.eig(mat)
        pt.plot(w.real, "o-")
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
            **lpot_kwargs)

    # }}}

    if 0:
        # {{{ plot on-surface potential in 2D

        pt.plot(curve_pot, label="pot")
        pt.plot(density, label="dens")
        pt.legend()
        pt.show()

        # }}}

    if 0:
        # {{{ 2D false-color plot

        pt.clf()
        plotval = np.log10(1e-20+np.abs(vol_pot))
        im = fp.show_scalar_in_matplotlib(plotval.real)
        from matplotlib.colors import Normalize
        im.set_norm(Normalize(vmin=-2, vmax=1))

        src = native_curve.pos
        pt.plot(src[:, 0], src[:, 1], "o-k")
        # close the curve
        pt.plot(src[-1::-len(src)+1, 0], src[-1::-len(src)+1, 1], "o-k")

        #pt.gca().set_aspect("equal", "datalim")
        cb = pt.colorbar(shrink=0.9)
        cb.set_label(r"$\log_{10}(\mathdefault{Error})$")
        #from matplotlib.ticker import NullFormatter
        #pt.gca().xaxis.set_major_formatter(NullFormatter())
        #pt.gca().yaxis.set_major_formatter(NullFormatter())
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

        fp.show_scalar_in_mayavi(scale*plotval_vol, max_val=1)
        from mayavi import mlab
        mlab.colorbar()
        if 1:
            mlab.points3d(
                    native_curve.pos[0],
                    native_curve.pos[1],
                    scale*plotval_c, scale_factor=0.02)
        mlab.show()

        # }}}


if __name__ == "__main__":
    draw_pot_figure(aspect_ratio=1, nsrc=100, novsmp=100, helmholtz_k=(15+4j)*0.3,
            what_operator="D", what_operator_lpot="D", force_center_side=1)
    pt.savefig("eigvals-ext-nsrc100-novsmp100.pdf")
    #pt.clf()
    #draw_pot_figure(aspect_ratio=1, nsrc=100, novsmp=100, helmholtz_k=0,
    #        what_operator="D", what_operator_lpot="D", force_center_side=-1)
    #pt.savefig("eigvals-int-nsrc100-novsmp100.pdf")
    #pt.clf()
    #draw_pot_figure(aspect_ratio=1, nsrc=100, novsmp=200, helmholtz_k=0,
    #        what_operator="D", what_operator_lpot="D", force_center_side=-1)
    #pt.savefig("eigvals-int-nsrc100-novsmp200.pdf")

# vim: fdm=marker
