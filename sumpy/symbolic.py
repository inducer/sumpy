from __future__ import division

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




import sympy as sp




# {{{ trivial assignment elimination

def make_one_step_subst(assignments):
    unwanted_vars = set(sp.Symbol(name) for name, value in assignments)

    result = []
    for name, value in assignments:
        while value.atoms() & unwanted_vars:
            value = value.subs(assignments)

        result.append((name, value))

    return result

def is_assignment_nontrivial(name, value):
    if value.is_Number:
        return False
    elif isinstance(value, sp.Symbol):
        return False
    elif (isinstance(value, sp.Mul)
            and len(value.args) == 2
            and sum(1 for arg in value.args if arg.is_Number) == 1
            and sum(1 for arg in value.args if isinstance(arg, sp.Symbol)) == 1
            ):
        # const*var: not good enough
        return False

    return True

def kill_trivial_assignments(assignments, retain_names=set()):
    approved_assignments = []
    rejected_assignments = []

    for name, value in assignments:
        if name in retain_names or is_assignment_nontrivial(name, value):
            approved_assignments.append((name, value))
        else:
            rejected_assignments.append((name, value))

    # un-substitute rejected assignments
    unsubst_rej = make_one_step_subst(rejected_assignments)

    return [(name, expr.subs(unsubst_rej))
            for name, expr in approved_assignments]

# }}}

# {{{ debugging of sympy CSE via Maxima

from pymbolic.mapper import IdentityMapper as PymbolicIdentityMapper

class _DerivativeKiller(PymbolicIdentityMapper):
    def map_derivative(self, expr):
        from pymbolic import var
        return var("d_"+"_".join(expr.variables))(expr.child)

    def map_substitution(self, expr):
        return self.rec(expr.child)

def _get_assignments_in_maxima(assignments, prefix=""):
    my_variable_names = set(assignments.iterkeys())
    written_assignments = set()

    prefix_subst_dict = dict(
            (vn, prefix+vn) for vn in my_variable_names)

    from pymbolic.maxima import MaximaStringifyMapper
    mstr = MaximaStringifyMapper()
    from pymbolic.sympy_interface import SympyToPymbolicMapper
    s2p = SympyToPymbolicMapper()
    dkill = _DerivativeKiller()

    result = []

    def write_assignment(name):
        symbols = [atm for atm in assignments[name].atoms()
                if isinstance(atm, sp.Symbol)
                and atm.name in my_variable_names]

        for sym in symbols:
            if sym.name not in written_assignments:
                write_assignment(sym.name)

        result.append("%s%s : %s;" % (
            prefix, name, mstr(dkill(s2p(
                assignments[name].subs(prefix_subst_dict))))))
        written_assignments.add(name)

    for name in assignments.iterkeys():
        if name not in written_assignments:
            write_assignment(name)

    return "\n".join(result)

def checked_cse(exprs, symbols=None):
    kwargs = {}
    if symbols is not None:
        kwargs["symbols"] = symbols

    new_assignments, new_exprs = sp.cse(exprs, **kwargs)

    max_old = _get_assignments_in_maxima(dict(
            ("old_expr%d" % i, expr)
            for i, expr in enumerate(exprs)))
    new_ass_dict = dict(
            ("new_expr%d" % i, expr)
            for i, expr in enumerate(new_exprs))
    for name, val in new_assignments:
        new_ass_dict[name.name] = val
    max_new = _get_assignments_in_maxima(new_ass_dict)

    with open("check.mac", "w") as outf:
        outf.write("ratprint:false;\n")
        outf.write("%s\n\n" % max_old)
        outf.write("%s\n" % max_new)
        for i in xrange(len(exprs)):
            outf.write("print(\"diff in expr %d:\n\");\n" % i)
            outf.write("print(ratsimp(old_expr%d - new_expr%d));\n" % (i, i))

    from subprocess import check_call
    check_call(["maxima", "--very-quiet", "-r", "load(\"check.mac\");"])

    return new_assignments, new_exprs

# }}}




def sympy_real_norm_2(x):
    return sp.sqrt((x.T*x)[0,0])

def make_sym_vector(name, components):
    return sp.Matrix(
            [sp.Symbol("%s%d" % (name, i)) for i in range(components)])




def vector_subs(expr, old, new):
    result = expr
    for old_i, new_i in zip(old, new):
        result = result.subs(old_i, new_i)

    return result



def find_power_of(base, prod):
    remdr = sp.Wild("remdr")
    power = sp.Wild("power")
    result = prod.match(remdr*base**power)
    if result is None:
        return 0
    return result[power]

# vim: fdm=marker
