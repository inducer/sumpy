from __future__ import division
import sympy as sp
from sympy.printing.codeprinter import CodePrinter as BaseCodePrinter




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







class SympyMapper(object):
    def __call__(self, expr, *args, **kwargs):
        return self.rec(expr, *args, **kwargs)

    def rec(self, expr, *args, **kwargs):
        mro = list(type(expr).__mro__)

        while mro:
            method_name = "map_"+mro.pop(0).__name__

            try:
                method = getattr(self, method_name)
            except AttributeError:
                pass
            else:
                return method(expr, *args, **kwargs)

        raise NotImplementedError(
                "%s does not know how to map type '%s'"
                % (type(self).__name__,
                    type(expr).__name__))





class IdentityMapper(SympyMapper):
    def map_Add(self, expr):
        return type(expr)(*tuple(self.rec(arg) for arg in expr.args))

    map_Mul = map_Add
    map_Pow = map_Add
    map_Function = map_Add

    def map_Rational(self, expr):
        return type(expr)(self.rec(expr.p), self.rec(expr.q))

    def map_Integer(self, expr):
        return expr
    map_int = map_Integer

    map_Symbol = map_Integer
    map_Real = map_Integer




class SquareRewriter(IdentityMapper):
    def __init__(self, symbol_gen, expr_to_var={}):
        self.assignments = []
        self.symbol_gen = iter(symbol_gen)
        self.expr_to_var = expr_to_var

    def get_var_for(self, expr):
        try:
            return self.expr_to_var[expr]
        except KeyError:
            sym = self.symbol_gen.next()
            self.assignments.append((sym, expr))
            self.expr_to_var[expr] = sym
            return sym

    def __call__(self, var_name, expr):
        self.assignments.append((var_name, self.rec(expr)))

    def map_Pow(self, expr):
        if expr.exp == 2:
            new_base = self.get_var_for(expr.base)
            return new_base**2
        else:
            return IdentityMapper.map_Pow(self, expr)





def generate_cl_statements_from_assignments(assignments, subst_map={}):
    """
    :param assignments: a list of tuples *(var_name, expr)*
    """

    # {{{ perform CSE

    from sympy.utilities.iterables import  numbered_symbols
    sym_gen = numbered_symbols("cse")

    new_assignments = []
    for var_name, expr in assignments:
        from sympy.simplify.cse_main import cse
        replacements, reduced = cse([expr], sym_gen)

        new_assignments.extend(
                (sym.name, expr) for sym, expr in replacements)

        new_assignments.append((var_name, reduced[0]))

    assignments = new_assignments
    # }}}

    # {{{ rewrite squares

    sq_rewriter = SquareRewriter(sym_gen, expr_to_var=dict(
        (expr, sp.Symbol(var_name)) for var_name, expr in assignments))

    for var_name, expr in assignments:
        sq_rewriter(var_name, expr)

    # }}}

    # {{{ print code

    ccp = CLCodePrinter(subst_map=subst_map)
    return [(var_name, ccp.doprint(expr))
            for var_name, expr in sq_rewriter.assignments]

    # }}}
