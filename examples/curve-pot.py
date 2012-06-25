from __future__ import division
import pyopencl as cl
import numpy as np
import matplotlib.pyplot as pt
from pytools import Record




def draw_pot_figure(aspect_ratio,
        nsrc=100, novsmp=None, helmholtz_k=0, what_operator="S", order=5,
        ovsmp_center_exp=0.66):

    if novsmp is None:
        novsmp = 4*nsrc

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)


    # {{{ make plot targets

    center = np.asarray([0,0], dtype=np.float64)
    from sumpy.visualization import FieldPlotter
    fp = FieldPlotter(center, points=1000, extent=3)

    tgt = fp.points.copy()

    # }}}

    # {{{ make p2p kernel calculator

    from sumpy.p2p import P2P
    from sumpy.kernel import LaplaceKernel, HelmholtzKernel
    from sumpy.local_expansion import H2DLocalExpansion, LineTaylorLocalExpansion
    if helmholtz_k:
        knl = HelmholtzKernel(2)
        expn_class = H2DLocalExpansion
        knl_kwargs = {"k": helmholtz_k}
    else:
        knl = LaplaceKernel(2)
        expn_class = LineTaylorLocalExpansion
        knl_kwargs = {}

    if what_operator == "S":
        pass
    elif what_operator == "D":
        from sumpy.kernel import SourceDerivative
        knl = SourceDerivative(knl)
    else:
        raise RuntimeError("unrecognized operator '%s'" % what_operator)

    p2p = P2P(ctx, [knl], exclude_self=False,
            value_dtypes=np.complex128)

    from sumpy.layerpot import LayerPotential, JumpTermApplier
    lpot = LayerPotential(ctx, [
        expn_class(knl, order=order)],
            value_dtypes=np.complex128)

    jt = JumpTermApplier(ctx, [knl],
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
    hnative = curve_len/nsrc
    center_dist = 5*hnative * (hovsmp/hnative)**ovsmp_center_exp

    center_side = -np.sign(native_curve.mean_curvature)
    centers = (native_curve.pos
            + center_side[:, np.newaxis]
            *  center_dist*native_curve.normal)

    #native_curve.plot()
    #pt.show()

    kwargs = knl_kwargs.copy()
    if what_operator == "D":
        kwargs["src_derivative_dir"] = native_curve.normal
    ovsmp_kwargs = knl_kwargs.copy()
    if what_operator == "D":
        ovsmp_kwargs["src_derivative_dir"] = ovsmp_curve.normal

    # }}}

    len(native_curve)
    density = np.cos(3*2*np.pi*native_t).astype(np.complex128)
    ovsmp_density = np.cos(3*2*np.pi*ovsmp_t).astype(np.complex128)
    evt, (vol_pot,) = p2p(queue, tgt, native_curve.pos, [density], **kwargs)

    evt, (curve_pot,) = lpot(queue, native_curve.pos, ovsmp_curve.pos, centers,
            [ovsmp_density], ovsmp_curve.speed, ovsmp_weights,
            **ovsmp_kwargs)
    class JumpTermArgs(Record):
        pass
    evt, (curve_pot,) = jt(queue, [curve_pot], JumpTermArgs(
        src_derivative_dir=native_curve.normal,
        density_0=density,
        side=center_side,
        normal=native_curve.normal,
        ))
    pt.plot(curve_pot, label="pot")
    pt.plot(density, label="dens")
    pt.legend()
    pt.show()

    if 0:
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
    else:
        if 0:
            plotval_vol = np.log10(1e-20+np.abs(vol_pot))
        else:
            plotval_vol = vol_pot.real
            plotval_c = curve_pot.real

        scale = 3e-2

        fp.show_scalar_in_mayavi(scale*plotval_vol, maxval=1)
        from mayavi import mlab
        mlab.colorbar()
        mlab.points3d(
                native_curve.pos[:,0],
                native_curve.pos[:,1],
                scale*plotval_c)
        mlab.show()




if __name__ == "__main__":
    draw_pot_figure(aspect_ratio=1, nsrc=300, helmholtz_k=0, what_operator="D")
    pt.show()
