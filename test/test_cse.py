__copyright__ = """
Copyright (C) 2017 Matt Wala
Copyright (C) 2006-2016 SymPy Development Team
"""

# {{{ license and original license

__license__ = """
Modifications from original are under the following license:

Copyright (C) 2017 Matt Wala

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

===========================================================================

Based on sympy/simplify/tests/test_cse.py from SymPy 1.0, license as follows:

Copyright (c) 2006-2016 SymPy Development Team

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of SymPy nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

# }}}

import pytest
import sys

import sumpy.symbolic as sym
from sumpy.cse import cse, preprocess_for_cse, postprocess_for_cse

if not sym.USE_SYMENGINE:
    from sympy.simplify.cse_opts import sub_pre, sub_post
    from sympy.functions.special.hyper import meijerg
    from sympy.simplify import cse_opts

import logging
logger = logging.getLogger(__name__)

w, x, y, z = sym.symbols("w,x,y,z")
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = sym.symbols("x:13")

sympyonly = (
    pytest.mark.skipif(sym.USE_SYMENGINE, reason="uses a sympy-only feature"))


def opt1(expr):
    return expr + y


def opt2(expr):
    return expr*z


# {{{ test_preprocess_for_cse

def test_preprocess_for_cse():
    assert preprocess_for_cse(x, [(opt1, None)]) == x + y
    assert preprocess_for_cse(x, [(None, opt1)]) == x
    assert preprocess_for_cse(x, [(None, None)]) == x
    assert preprocess_for_cse(x, [(opt1, opt2)]) == x + y
    assert preprocess_for_cse(
        x, [(opt1, None), (opt2, None)]) == (x + y)*z

# }}}


# {{{ test_postprocess_for_cse

def test_postprocess_for_cse():
    assert postprocess_for_cse(x, [(opt1, None)]) == x
    assert postprocess_for_cse(x, [(None, opt1)]) == x + y
    assert postprocess_for_cse(x, [(None, None)]) == x
    assert postprocess_for_cse(x, [(opt1, opt2)]) == x*z
    # Note the reverse order of application.
    assert postprocess_for_cse(
        x, [(None, opt1), (None, opt2)]) == x*z + y

# }}}


# {{{ test_cse_single

def test_cse_single():
    # Simple substitution.
    e = sym.Add(sym.Pow(x + y, 2), sym.sqrt(x + y))
    substs, reduced = cse([e])
    assert substs == [(x0, x + y)]
    assert reduced == [sym.sqrt(x0) + x0**2]

# }}}


# {{{

@sympyonly
def test_cse_not_possible():
    # No substitution possible.
    e = sym.Add(x, y)
    substs, reduced = cse([e])
    assert substs == []
    assert reduced == [x + y]
    # issue 6329
    eq = (meijerg((1, 2), (y, 4), (5,), [], x)  # pylint: disable=possibly-used-before-assignment
          + meijerg((1, 3), (y, 4), (5,), [], x))  # pylint: disable=possibly-used-before-assignment
    assert cse(eq) == ([], [eq])

# }}}


# {{{ test_nested_substitution

def test_nested_substitution():
    # Substitution within a substitution.
    e = sym.Add(sym.Pow(w*x + y, 2), sym.sqrt(w*x + y))
    substs, reduced = cse([e])
    assert substs == [(x0, w*x + y)]
    assert reduced == [sym.sqrt(x0) + x0**2]

# }}}


# {{{ test_subtraction_opt

@sympyonly
def test_subtraction_opt():
    # Make sure subtraction is optimized.
    e = (x - y)*(z - y) + sym.exp((x - y)*(z - y))
    substs, reduced = cse(
        [e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])  # pylint: disable=possibly-used-before-assignment
    assert substs == [(x0, (x - y)*(y - z))]
    assert reduced == [-x0 + sym.exp(-x0)]
    e = -(x - y)*(z - y) + sym.exp(-(x - y)*(z - y))
    substs, reduced = cse(
        [e], optimizations=[(cse_opts.sub_pre, cse_opts.sub_post)])
    assert substs == [(x0, (x - y)*(y - z))]
    assert reduced == [x0 + sym.exp(x0)]
    # issue 4077
    n = -1 + 1/x
    e = n/x/(-n)**2 - 1/n/x
    assert cse(e, optimizations=[
               (cse_opts.sub_pre, cse_opts.sub_post)]  # pylint: disable=possibly-used-before-assignment
               ) == ([], [0])

# }}}


# {{{ test_multiple_expressions

def test_multiple_expressions():
    e1 = (x + y)*z
    e2 = (x + y)*w
    substs, reduced = cse([e1, e2])
    assert substs == [(x0, x + y)]
    assert reduced == [x0*z, x0*w]
    l_ = [w*x*y + z, w*y]
    substs, reduced = cse(l_)
    rsubsts, _ = cse(reversed(l_))
    assert substs == rsubsts
    assert reduced == [z + x*x0, x0]
    l_ = [w*x*y, w*x*y + z, w*y]
    substs, reduced = cse(l_)
    rsubsts, _ = cse(reversed(l_))
    assert substs == rsubsts
    assert reduced == [x1, x1 + z, x0]
    f = sym.Function("f")
    l_ = [f(x - z, y - z), x - z, y - z]
    substs, reduced = cse(l_)
    rsubsts, _ = cse(reversed(l_))
    assert substs == [(x0, -z), (x1, x + x0), (x2, x0 + y)]
    assert rsubsts == [(x0, -z), (x1, x0 + y), (x2, x + x0)]
    assert reduced == [f(x1, x2), x1, x2]
    l_ = [w*y + w + x + y + z, w*x*y]
    assert cse(l_) == ([(x0, w*y)], [w + x + x0 + y + z, x*x0])
    assert cse([x + y, x + y + z]) == ([(x0, x + y)], [x0, z + x0])
    assert cse([x + y, x + z]) == ([], [x + y, x + z])
    assert cse([x*y, z + x*y, x*y*z + 3]) == \
        ([(x0, x*y)], [x0, z + x0, 3 + x0*z])

# }}}


# {{{ test_issue_4203

def test_issue_4203():
    assert cse(sym.sin(x**x)/x**x) == ([(x0, x**x)], [sym.sin(x0)/x0])

# }}}


# {{{ test_dont_cse_subs

def test_dont_cse_subs():
    f = sym.Function("f")
    g = sym.Function("g")

    name_val, (expr,) = cse(f(x+y).diff(x) + g(x+y).diff(x))

    assert name_val == []

# }}}


# {{{ test_dont_cse_derivative

def test_dont_cse_derivative():
    f = sym.Function("f")

    deriv = sym.Derivative(f(x+y), x)

    name_val, (expr,) = cse(x + y + deriv)

    assert name_val == []
    assert expr == x + y + deriv

# }}}


# {{{ test_pow_invpow

def test_pow_invpow():
    assert cse(1/x**2 + x**2) == \
        ([(x0, x**2)], [x0 + 1/x0])
    assert cse(x**2 + (1 + 1/x**2)/x**2) == \
        ([(x0, x**2), (x1, 1/x0)], [x0 + x1*(x1 + 1)])
    assert cse(1/x**2 + (1 + 1/x**2)*x**2) == \
        ([(x0, x**2), (x1, 1/x0)], [x0*(x1 + 1) + x1])
    assert cse(sym.cos(1/x**2) + sym.sin(1/x**2)) == \
        ([(x0, x**(-2))], [sym.sin(x0) + sym.cos(x0)])
    assert cse(sym.cos(x**2) + sym.sin(x**2)) == \
        ([(x0, x**2)], [sym.sin(x0) + sym.cos(x0)])
    assert cse(y/(2 + x**2) + z/x**2/y) == \
        ([(x0, x**2)], [y/(x0 + 2) + z/(x0*y)])
    assert cse(sym.exp(x**2) + x**2*sym.cos(1/x**2)) == \
        ([(x0, x**2)], [x0*sym.cos(1/x0) + sym.exp(x0)])
    assert cse((1 + 1/x**2)/x**2) == \
        ([(x0, x**(-2))], [x0*(x0 + 1)])
    assert cse(x**(2*y) + x**(-2*y)) == \
        ([(x0, x**(2*y))], [x0 + 1/x0])

# }}}


# {{{ test_issue_4499

@sympyonly
def test_issue_4499():
    # previously, this gave 16 constants
    from sympy.abc import a, b
    from sympy import Tuple, S

    B = sym.Function("B")   # noqa: N806
    G = sym.Function("G")   # noqa: N806
    t = Tuple(*(
        a,
        a + S(1)/2,
        2*a,
        b,
        2*a - b + 1,
        (sym.sqrt(z)/2)**(-2*a + 1)
        * B(2*a-b, sym.sqrt(z))
        * B(b - 1, sym.sqrt(z))*G(b)*G(2*a - b + 1),
        sym.sqrt(z)*(sym.sqrt(z)/2)**(-2*a + 1)
        * B(b, sym.sqrt(z))
        * B(2*a - b, sym.sqrt(z))*G(b)*G(2*a - b + 1),
        sym.sqrt(z)*(sym.sqrt(z)/2)**(-2*a + 1)
        * B(b - 1, sym.sqrt(z))
        * B(2*a - b + 1, sym.sqrt(z))*G(b)*G(2*a - b + 1),
        (sym.sqrt(z)/2)**(-2*a + 1)
        * B(b, sym.sqrt(z))
        * B(2*a - b + 1, sym.sqrt(z))*G(b)*G(2*a - b + 1),
        1,
        0,
        S(1)/2,
        z/2,
        -b + 1,
        -2*a + b,
        -2*a))
    c = cse(t)
    assert len(c[0]) == 11

# }}}


# {{{ test_issue_6169

@sympyonly
def test_issue_6169():
    from sympy import CRootOf
    r = CRootOf(x**6 - 4*x**5 - 2, 1)
    assert cse(r) == ([], [r])
    # and a check that the right thing is done with the new
    # mechanism
    assert sub_post(sub_pre((-x - y)*z - x - y)) == -z*(x + y) - x - y  # pylint: disable=possibly-used-before-assignment

# }}}


# {{{ test_cse_Indexed

@sympyonly
def test_cse_indexed():
    from sympy import IndexedBase, Idx
    len_y = 5
    y = IndexedBase("y", shape=(len_y,))
    x = IndexedBase("x", shape=(len_y,))
    i = Idx("i", len_y-1)

    expr1 = (y[i+1]-y[i])/(x[i+1]-x[i])
    expr2 = 1/(x[i+1]-x[i])
    replacements, reduced_exprs = cse([expr1, expr2])
    assert len(replacements) > 0

# }}}


# {{{ test_Piecewise

@sympyonly
def test_piecewise():
    from sympy import Piecewise, Eq
    f = Piecewise((-z + x*y, Eq(y, 0)), (-z - x*y, True))
    ans = cse(f)
    actual_ans = ([(x0, -z), (x1, x*y)],
                  [Piecewise((x0+x1, Eq(y, 0)), (x0 - x1, True))])
    assert ans == actual_ans

# }}}


# {{{ test_name_conflict

def test_name_conflict():
    z1 = x0 + y
    z2 = x2 + x3
    l_ = [sym.cos(z1) + z1, sym.cos(z2) + z2, x0 + x2]
    substs, reduced = cse(l_)
    assert [e.subs(dict(substs)) for e in reduced] == l_

# }}}


# {{{ test_name_conflict_cust_symbols

def test_name_conflict_cust_symbols():
    z1 = x0 + y
    z2 = x2 + x3
    l_ = [sym.cos(z1) + z1, sym.cos(z2) + z2, x0 + x2]
    substs, reduced = cse(l_, sym.symbols("x:10"))
    assert [e.subs(dict(substs)) for e in reduced] == l_

# }}}


# {{{ test_symbols_exhausted_error

def test_symbols_exhausted_error():
    l_ = sym.cos(x+y)+x+y+sym.cos(w+y)+sym.sin(w+y)
    s = [x, y, z]
    with pytest.raises(ValueError):
        logger.info("cse:\n%s", cse(l_, symbols=s))

# }}}


# {{{ test_issue_7840

@sympyonly
def test_issue_7840():
    # daveknippers' example
    C393 = sym.sympify(     # noqa: N806
        "Piecewise((C391 - 1.65, C390 < 0.5), (Piecewise((C391 - 1.65, \
        C391 > 2.35), (C392, True)), True))"
    )
    C391 = sym.sympify(     # noqa: N806
        "Piecewise((2.05*C390**(-1.03), C390 < 0.5), (2.5*C390**(-0.625), True))"
    )
    C393 = C393.subs("C391", C391)   # noqa: N806
    # simple substitution
    sub = {}
    sub["C390"] = 0.703451854
    sub["C392"] = 1.01417794
    ss_answer = C393.subs(sub)
    # cse
    substitutions, new_eqn = cse(C393)
    for pair in substitutions:
        sub[pair[0].name] = pair[1].subs(sub)
    cse_answer = new_eqn[0].subs(sub)
    # both methods should be the same
    assert ss_answer == cse_answer

    # GitRay's example
    expr = sym.sympify(
        "Piecewise((Symbol('ON'), Equality(Symbol('mode'), Symbol('ON'))), \
        (Piecewise((Piecewise((Symbol('OFF'), StrictLessThan(Symbol('x'), \
        Symbol('threshold'))), (Symbol('ON'), S.true)), Equality(Symbol('mode'), \
        Symbol('AUTO'))), (Symbol('OFF'), S.true)), S.true))"
    )
    substitutions, new_eqn = cse(expr)
    # this Piecewise should be exactly the same
    assert new_eqn[0] == expr
    # there should not be any replacements
    assert len(substitutions) < 1

# }}}


# {{{ test_recursive_matching

def test_recursive_matching():
    assert cse([x+y, 2+x+y, x+y+z, 3+x+y+z]) == \
        ([(x0, x + y), (x1, x0 + z)], [x0, x0 + 2, x1, x1 + 3])
    assert cse(reversed([x+y, 2+x+y, x+y+z, 3+x+y+z])) == \
        ([(x0, x + y), (x1, x0 + z)], [x1 + 3, x1, x0 + 2, x0])
    # sympy 1.0 gives ([(x0, x*y*z)], [5*x0, w*(x*y), 3*x0])
    assert cse([x*y*z*5, x*y*w, x*y*z*3]) == \
        ([(x0, x*y), (x1, x0*z)], [5*x1, w*x0, 3*x1])
    # sympy 1.0 gives ([(x4, x*y*z)], [5*x4, w*x3*x4, 3*x*x0*x1*x2*y])
    assert cse([x*y*z*5, x*y*z*w*x3, x*y*3*x0*x1*x2]) == \
        ([(x4, x*y), (x5, x4*z)], [5*x5, w*x3*x5, 3*x0*x1*x2*x4])
    assert cse([2*x*x, x*x*y, x*x*y*w, x*x*y*w*x0, x*x*y*w*x2]) == \
        ([(x1, x**2), (x3, x1*y), (x4, w*x3)], [2*x1, x3, x4, x0*x4, x2*x4])

# }}}


# You can test individual routines by typing
# $ python test_cse.py 'test_recursive_matching()'

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        pytest.main([__file__])

# vim: fdm=marker
