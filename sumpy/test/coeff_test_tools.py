from __future__ import annotations


__copyright__ = """
Copyright (C) 2026 Shawn/Chaoqi Lin
"""

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
import sympy as sp


def to_scalar(val):
    """Convert symbolic or array value to scalar."""
    if hasattr(val, "evalf"):
        val = val.evalf()
    if hasattr(val, "item"):
        val = val.item()
    return complex(val)


class NumericMatVecOperator:
    """Wrapper for symbolic matrix-vector operator with numeric
    substitution."""

    def __init__(self, symbolic_op, repl_dict):
        self.symbolic_op = symbolic_op
        self.repl_dict = repl_dict
        self.shape = symbolic_op.shape

    def matvec(self, vec):
        result = self.symbolic_op.matvec(vec)
        out = []
        for expr in result:
            if hasattr(expr, "xreplace"):
                out.append(complex(expr.xreplace(self.repl_dict).evalf()))
            else:
                out.append(complex(expr))
        return np.array(out)


def get_repl_dict(kernel, extra_kwargs):
    """Numeric substitution for symbolic kernel parameters."""
    repl_dict = {}
    if "lam" in extra_kwargs:
        repl_dict[sp.Symbol("lam")] = extra_kwargs["lam"]
    if "k" in extra_kwargs:
        repl_dict[sp.Symbol("k")] = extra_kwargs["k"]
    return repl_dict
