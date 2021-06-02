__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
from pymbolic.mapper import IdentityMapper as IdentityMapperBase
import pymbolic.primitives as prim

import logging
logger = logging.getLogger(__name__)


# {{{ symbolic backend

def _find_symbolic_backend():
    global USE_SYMENGINE

    try:
        import symengine  # noqa
        symengine_found = True
    except ImportError as import_error:
        symengine_found = False
        symengine_error = import_error

    ALLOWED_BACKENDS = ("sympy", "symengine")  # noqa
    BACKEND_ENV_VAR = "SUMPY_FORCE_SYMBOLIC_BACKEND"  # noqa

    import os
    backend = os.environ.get(BACKEND_ENV_VAR)
    if backend is not None:
        if backend not in ALLOWED_BACKENDS:
            raise RuntimeError(
                "%s value is unrecognized: '%s' "
                "(allowed values are %s)" % (
                    BACKEND_ENV_VAR,
                    backend,
                    ", ".join("'%s'" % val for val in ALLOWED_BACKENDS)))

        if backend == "symengine" and not symengine_found:
            raise RuntimeError("could not find SymEngine: %s" % symengine_error)

        USE_SYMENGINE = (backend == "symengine")
    else:
        USE_SYMENGINE = symengine_found


_find_symbolic_backend()

# }}}

# Symbolic API common to SymEngine and sympy.
# Before adding a function here, make sure it's present in both modules.
SYMBOLIC_API = """
Add Basic Mul Pow exp sqrt log symbols sympify cos sin atan2 Function Symbol
Derivative Integer Matrix Subs I pi functions Number""".split()

if USE_SYMENGINE:
    import symengine as sym
    from pymbolic.interop.symengine import (
        PymbolicToSymEngineMapper as PymbolicToSympyMapper,
        SymEngineToPymbolicMapper as SympyToPymbolicMapper)
else:
    import sympy as sym
    from pymbolic.interop.sympy import (
        PymbolicToSympyMapper, SympyToPymbolicMapper)

for _apifunc in SYMBOLIC_API:
    globals()[_apifunc] = getattr(sym, _apifunc)


def _coeff_isneg(a):
    if a.is_Mul:
        a = a.args[0]
    return a.is_Number and a.is_negative


have_unevaluated_expr = False
if not USE_SYMENGINE:
    try:
        from sympy import UnevaluatedExpr
        have_unevaluated_expr = True
    except ImportError:
        pass

if not have_unevaluated_expr:
    def UnevaluatedExpr(x):  # noqa
        return x


if USE_SYMENGINE:
    def unevaluated_pow(a, b):
        return sym.Pow(a, b)
else:
    def unevaluated_pow(a, b):
        return sym.Pow(a, b, evaluate=False)


# {{{ debugging of sympy CSE via Maxima

class _DerivativeKiller(IdentityMapperBase):
    def map_derivative(self, expr):
        from pymbolic import var
        return var("d_"+"_".join(expr.variables))(expr.child)

    def map_substitution(self, expr):
        return self.rec(expr.child)


def _get_assignments_in_maxima(assignments, prefix=""):
    my_variable_names = set(assignments.keys())
    written_assignments = set()

    prefix_subst_dict = {
            vn: prefix+vn for vn in my_variable_names}

    from pymbolic.maxima import MaximaStringifyMapper
    mstr = MaximaStringifyMapper()
    s2p = SympyToPymbolicMapper()
    dkill = _DerivativeKiller()

    result = []

    def write_assignment(name):
        symbols = [atm for atm in assignments[name].atoms()
                if isinstance(atm, sym.Symbol)
                and atm.name in my_variable_names]

        for symb in symbols:
            if symb.name not in written_assignments:
                write_assignment(symb.name)

        result.append("{}{} : {};".format(
            prefix, name, mstr(dkill(s2p(
                assignments[name].subs(prefix_subst_dict))))))
        written_assignments.add(name)

    for name in assignments.keys():
        if name not in written_assignments:
            write_assignment(name)

    return "\n".join(result)


def checked_cse(exprs, symbols=None):
    kwargs = {}
    if symbols is not None:
        kwargs["symbols"] = symbols

    new_assignments, new_exprs = sym.cse(exprs, **kwargs)

    max_old = _get_assignments_in_maxima({
            "old_expr%d" % i: expr
            for i, expr in enumerate(exprs)})
    new_ass_dict = {
            "new_expr%d" % i: expr
            for i, expr in enumerate(new_exprs)}
    for name, val in new_assignments:
        new_ass_dict[name.name] = val
    max_new = _get_assignments_in_maxima(new_ass_dict)

    with open("check.mac", "w") as outf:
        outf.write("ratprint:false;\n")
        outf.write("%s\n\n" % max_old)
        outf.write("%s\n" % max_new)
        for i in range(len(exprs)):
            outf.write('print("diff in expr %d:\n");\n' % i)
            outf.write("print(ratsimp(old_expr%d - new_expr%d));\n" % (i, i))

    from subprocess import check_call
    check_call(["maxima", "--very-quiet", "-r", 'load("check.mac");'])

    return new_assignments, new_exprs

# }}}


def sym_real_norm_2(x):
    return sym.sqrt((x.T*x)[0, 0])


def pymbolic_real_norm_2(x):
    from pymbolic import var
    return var("sqrt")(np.dot(x, x))


def make_sym_vector(name, components):
    return sym.Matrix(
            [sym.Symbol("%s%d" % (name, i)) for i in range(components)])


def vector_xreplace(expr, from_vec, to_vec):
    substs = {}
    assert (from_vec.rows, from_vec.cols) == (to_vec.rows, to_vec.cols)
    for irow in range(from_vec.rows):
        for icol in range(from_vec.cols):
            substs[from_vec[irow, icol]] = to_vec[irow, icol]

    return expr.xreplace(substs)


def find_power_of(base, prod):
    remdr = sym.Wild("remdr")
    power = sym.Wild("power")
    result = prod.match(remdr*base**power)
    if result is None:
        return 0
    return result[power]


class PymbolicToSympyMapperWithSymbols(PymbolicToSympyMapper):
    def map_variable(self, expr):
        if expr.name == "I":
            return sym.I
        elif expr.name == "pi":
            return sym.pi
        else:
            return PymbolicToSympyMapper.map_variable(self, expr)

    def map_subscript(self, expr):
        if isinstance(expr.aggregate, prim.Variable) and isinstance(expr.index, int):
            return sym.Symbol("%s%d" % (expr.aggregate.name, expr.index))
        else:
            self.raise_conversion_error(expr)

    def map_call(self, expr):
        if expr.function.name == "hankel_1":
            args = [self.rec(param) for param in expr.parameters]
            args.append(0)
            return Hankel1(*args)
        elif expr.function.name == "bessel_j":
            args = [self.rec(param) for param in expr.parameters]
            args.append(0)
            return BesselJ(*args)
        else:
            return PymbolicToSympyMapper.map_call(self, expr)


import sympy


class _BesselOrHankel(sympy.Function):
    """A symbolic function for BesselJ or Hankel1 functions
    that keeps track of the derivatives taken of the function.
    Arguments are ``(order, z, nderivs)``.
    """
    nargs = (3,)

    def fdiff(self, argindex=1):
        if argindex in (1, 3):
            # we are not differentiating w.r.t order or nderivs
            raise ValueError()
        order, z, nderivs = self.args
        return self.func(order, z, nderivs+1)


class BesselJ(_BesselOrHankel):
    pass


class Hankel1(_BesselOrHankel):
    pass


_SympyBesselJ = BesselJ
_SympyHankel1 = Hankel1

if USE_SYMENGINE:
    def BesselJ(*args):   # noqa: N802
        return sym.sympify(_SympyBesselJ(*args))

    def Hankel1(*args):   # noqa: N802
        return sym.sympify(_SympyHankel1(*args))

# vim: fdm=marker
