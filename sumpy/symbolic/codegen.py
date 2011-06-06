from __future__ import division
import sympy as sp
from sympy.printing.codeprinter import CodePrinter as BaseCodePrinter
from exafmm.symbolic import IdentityMapper




class CLCodePrinter(BaseCodePrinter):
    def __init__(self, vectors=set(), subst_map={}):
        BaseCodePrinter.__init__(self)
        self.vectors = vectors
        self.subst_map = subst_map

    def _print_Pow(self, expr):
        from sympy.core import S
        from sympy.printing.precedence import precedence
        PREC = precedence(expr)
        if expr.exp is S.NegativeOne:
            return '1.0/%s'%(self.parenthesize(expr.base, PREC))
        elif expr.exp == 0.5:
            return 'sqrt(%s)' % self._print(expr.base)
        elif expr.exp == -0.5:
            return 'rsqrt(%s)' % self._print(expr.base)
        elif expr.exp == 2 and isinstance(expr.base, sp.Symbol):
            return '%s*%s' % (expr.base.name, expr.base.name)
        else:
            return 'pow(%s, %s)'%(self._print(expr.base),
                                 self._print(expr.exp))

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d.0/%d.0' % (p, q)

    def _print_Symbol(self, expr):
        try:
            return self.subst_map[expr.name]
        except KeyError:
            return BaseCodePrinter._print_Symbol(self, expr)





class PowRewriter(IdentityMapper):
    def __init__(self, symbol_gen, expr_to_var={}):
        self.assignments = []
        self.symbol_gen = iter(symbol_gen)
        self.expr_to_var = expr_to_var

    def get_var_for(self, expr):
        if isinstance(expr, sp.Symbol):
            return expr

        try:
            return self.expr_to_var[expr]
        except KeyError:
            sym = self.symbol_gen.next()
            self.assignments.append((sym.name, expr))
            self.expr_to_var[expr] = sym
            return sym

    def __call__(self, var_name, expr):
        self.assignments.append((var_name, self.rec(expr)))

    def map_Pow(self, expr):
        if (not isinstance(expr.base, sp.Symbol) 
                and isinstance(expr.exp, int) and expr.exp > 1):
            new_base = self.get_var_for(expr.base)
            return new_base**expr.exp
        if isinstance(expr.exp, sp.Rational) and abs(expr.exp.p) > 1:
            if expr.exp.p < 0:
                new_p = abs(expr.exp.p)
                new_q = -expr.exp.q
            else:
                new_p = expr.exp.p
                new_q = expr.exp.q

            if new_q == 1:
                new_base = self.get_var_for(expr.base)
            else:
                new_base = self.get_var_for(expr.base**(1/new_q))

            if new_p == 1:
                return new_base
            else:
                return new_base**new_p

        return IdentityMapper.map_Pow(self, expr)




def generate_cl_statements_from_assignments(assignments, subst_map={}):
    """
    :param assignments: a list of tuples *(var_name, expr)*
    """

    from sympy.utilities.iterables import numbered_symbols
    sym_gen = numbered_symbols("cse")

    # {{{ perform CSE

    from exafmm.symbolic import eliminate_common_subexpressions

    cses, exprs = eliminate_common_subexpressions(
            [expr for var_name, expr in assignments],
            sym_gen)

    assignments = cses + [(name, new_expr)
        for (name, old_expr), new_expr in
        zip(assignments, exprs)]

    # }}}

    # {{{ rewrite powers

    sq_rewriter = PowRewriter(sym_gen, expr_to_var=dict(
        (expr, sp.Symbol(var_name)) for var_name, expr in assignments))

    for var_name, expr in assignments:
        sq_rewriter(var_name, expr)

    assignments = sq_rewriter.assignments

    # }}}

    # {{{ print code

    ccp = CLCodePrinter(subst_map=subst_map)
    return [(var_name, ccp.doprint(expr))
            for var_name, expr in assignments]

    # }}}




def gen_c_source_subst_map(dimensions):
    result = {}
    for i in range(dimensions):
        result["s%d" % i] = "src.s%d" % i
        result["t%d" % i] = "tgt.s%d" % i
        result["c%d" % i] = "ctr.s%d" % i

    return result

