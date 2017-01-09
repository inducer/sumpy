""" Tools for doing common subexpression elimination.
"""
from __future__ import print_function, division

__copyright__ = """
Copyright (C) 2017 Matt Wala
Copyright (C) 2006-2016 SymPy Development Team
"""

# {{{ license and original license

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

===========================================================================

Based on sympy/simplify/cse_main.py from SymPy 1.0, original license as follows:

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

from sympy.core import Basic, Mul, Add, Pow, sympify, Symbol, Tuple
#from symengine.sympy_compat import (Basic, Mul, Add, Pow, sympify, Symbol)
from sympy.core.singleton import S
from sympy.core.function import _coeff_isneg
from sympy.core.exprtools import factor_terms
from sympy.core.compatibility import iterable, range
from sympy.utilities.iterables import filter_symbols, \
    numbered_symbols, sift, topological_sort, ordered

#from . import cse_opts

# (preprocessor, postprocessor) pairs which are commonly useful. They should
# each take a sympy expression and return a possibly transformed expression.
# When used in the function ``cse()``, the target expressions will be transformed
# by each of the preprocessor functions in order. After the common
# subexpressions are eliminated, each resulting expression will have the
# postprocessor functions transform them in *reverse* order in order to undo the
# transformation if necessary. This allows the algorithm to operate on
# a representation of the expressions that allows for more optimization
# opportunities.
# ``None`` can be used to specify no transformation for either the preprocessor or
# postprocessor.


def preprocess_for_cse(expr, optimizations):
    """ Preprocess an expression to optimize for common subexpression
    elimination.

    Parameters
    ----------
    expr : sympy expression
        The target expression to optimize.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs.

    Returns
    -------
    expr : sympy expression
        The transformed expression.
    """
    for pre, post in optimizations:
        if pre is not None:
            expr = pre(expr)
    return expr


def postprocess_for_cse(expr, optimizations):
    """ Postprocess an expression after common subexpression elimination to
    return the expression to canonical sympy form.

    Parameters
    ----------
    expr : sympy expression
        The target expression to transform.
    optimizations : list of (callable, callable) pairs, optional
        The (preprocessor, postprocessor) pairs.  The postprocessors will be
        applied in reversed order to undo the effects of the preprocessors
        correctly.

    Returns
    -------
    expr : sympy expression
        The transformed expression.
    """
    for pre, post in reversed(optimizations):
        if post is not None:
            expr = post(expr)
    return expr


class FuncArgTracker(object):

    def __init__(self, funcs):
        # To minimize the number of symbolic comparisons, all function arguments
        # get assigned a value number.
        self.value_numbers = {}
        self.value_number_to_value = []

        # Both of these maps use integer indices for arguments / functions.
        self.arg_to_funcset = []
        self.func_to_argset = []

        for func_i, func in enumerate(funcs):
            func_argset = set()

            for func_arg in func.args:
                arg_number = self.get_or_add_value_number(func_arg)
                func_argset.add(arg_number)
                self.arg_to_funcset[arg_number].add(func_i)

            self.func_to_argset.append(func_argset)

    def get_args_in_value_order(self, argset):
        return [self.value_number_to_value[argn] for argn in sorted(argset)]

    def get_or_add_value_number(self, value):
        nvalues = len(self.value_numbers)
        value_number = self.value_numbers.setdefault(value, nvalues)
        if value_number == nvalues:
            self.value_number_to_value.append(value)
            self.arg_to_funcset.append(set())
        return value_number

    def stop_arg_tracking(self, func_i):
        for arg in self.func_to_argset[func_i]:
            self.arg_to_funcset[arg].remove(func_i)

    def gen_common_arg_candidates(self, argset, min_func_i, threshold=2):
        from collections import defaultdict
        count_map = defaultdict(lambda: 0)

        for arg in argset:
            for func_i in self.arg_to_funcset[arg]:
                if func_i >= min_func_i:
                    count_map[func_i] += 1

        count_map_keys_in_order = sorted(
            key for key, val in count_map.items()
            if val >= threshold)

        for item in count_map_keys_in_order:
            yield item

    def gen_subset_candidates(self, argset, min_func_i):
        iarg = iter(argset)

        indices = set(
            fi for fi in self.arg_to_funcset[next(iarg)]
            if fi >= min_func_i)

        for arg in iarg:
            indices &= self.arg_to_funcset[arg]

        for item in indices:
            yield item

    def update_func_argset(self, func_i, new_argset):
        new_args = set(new_argset)
        old_args = self.func_to_argset[func_i]

        for deleted_arg in old_args - new_args:
            self.arg_to_funcset[deleted_arg].remove(func_i)
        for added_arg in new_args - old_args:
            self.arg_to_funcset[added_arg].add(func_i)

        self.func_to_argset[func_i].clear()
        self.func_to_argset[func_i].update(new_args)


def match_common_args(Func, funcs, kind, opt_subs):
    funcs = list(funcs)
    arg_tracker = FuncArgTracker(funcs)

    changed = set()

    for i in range(len(funcs)):
        for j in arg_tracker.gen_common_arg_candidates(
                    arg_tracker.func_to_argset[i], i + 1, threshold=2):

            com_args = arg_tracker.func_to_argset[i].intersection(
                    arg_tracker.func_to_argset[j])

            if len(com_args) == 1:
                # This may happen if a set of common arguments was already
                # combined in a previous iteration.
                continue

            com_func = Func(*arg_tracker.get_args_in_value_order(com_args))
            com_func_number = arg_tracker.get_or_add_value_number(com_func)

            # for all sets, replace the common symbols by the function
            # over them, to allow recursive matches

            diff_i = arg_tracker.func_to_argset[i].difference(com_args)
            arg_tracker.update_func_argset(i, diff_i | set([com_func_number]))
            changed.add(i)

            diff_j = arg_tracker.func_to_argset[j].difference(com_args)
            arg_tracker.update_func_argset(j, diff_j | set([com_func_number]))
            changed.add(j)

            for k in arg_tracker.gen_subset_candidates(com_args, j + 1):
                diff_k = arg_tracker.func_to_argset[k].difference(com_args)
                arg_tracker.update_func_argset(k, diff_k | set([com_func_number]))
                changed.add(k)

        if i in changed:
            opt_subs[funcs[i]] = Func(
                *arg_tracker.get_args_in_value_order(arg_tracker.func_to_argset[i]),
                evaluate=False)

        arg_tracker.stop_arg_tracking(i)


def opt_cse(exprs):
    """Find optimization opportunities in Adds, Muls, Pows and negative
    coefficient Muls

    Parameters
    ----------
    exprs : list of sympy expressions
        The expressions to optimize.

    Returns
    -------
    opt_subs : dictionary of expression substitutions
        The expression substitutions which can be useful to optimize CSE.
    """
    opt_subs = dict()

    adds = set()
    muls = set()

    seen_subexp = set()

    def _find_opts(expr):

        if not isinstance(expr, Basic):
            return

        if expr.is_Atom: # or expr.is_Order:
            return

        if iterable(expr):
            list(map(_find_opts, expr))
            return

        if expr in seen_subexp:
            return expr
        seen_subexp.add(expr)

        list(map(_find_opts, expr.args))

        """
        def _coeff_isneg(expr):
            return expr.is_Number and expr < 0
        """

        if _coeff_isneg(expr):
            neg_expr = -expr
            if not neg_expr.is_Atom:
                opt_subs[expr] = Mul(S.NegativeOne, neg_expr, evaluate=False)
                seen_subexp.add(neg_expr)
                expr = neg_expr

        if isinstance(expr, Mul):
            muls.add(expr)

        elif isinstance(expr, Add):
            adds.add(expr)

        elif isinstance(expr, Pow):
            try:
                # symengine
                base, exp = expr.args
            except ValueError:
                # sympy
                base = expr.base
                exp = expr.exp
            if _coeff_isneg(exp):
                opt_subs[expr] = Pow(Pow(base, -exp), -1, evaluate=False)

    for e in exprs:
        if isinstance(e, Basic):
            _find_opts(e)

    ## Process Adds and Muls

    match_common_args(Add, adds, "add", opt_subs)
    match_common_args(Mul, muls, "mul", opt_subs)
    return opt_subs


def tree_cse(exprs, symbols, opt_subs=None, order='none'):
    """Perform raw CSE on expression tree, taking opt_subs into account.

    Parameters
    ==========

    exprs : list of sympy expressions
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out.
    opt_subs : dictionary of expression substitutions
        The expressions to be substituted before any CSE action is performed.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.
    """
    if opt_subs is None:
        opt_subs = dict()

    ## Find repeated sub-expressions

    to_eliminate = set()

    seen_subexp = set()

    def _find_repeated(expr):
        if not isinstance(expr, Basic):
            return

        if expr.is_Atom: #or expr.is_Order:
            return

        if iterable(expr):
            args = expr

        else:
            if expr in seen_subexp:
                to_eliminate.add(expr)
                return

            seen_subexp.add(expr)

            if expr in opt_subs:
                expr = opt_subs[expr]

            args = expr.args

        list(map(_find_repeated, args))

    for e in exprs:
        if isinstance(e, Basic):
            _find_repeated(e)

    ## Rebuild tree

    replacements = []

    subs = dict()

    def _rebuild(expr):
        if not isinstance(expr, Basic):
            return expr

        if not expr.args:
            return expr

        if iterable(expr):
            new_args = [_rebuild(arg) for arg in expr]
            return expr.func(*new_args)

        if expr in subs:
            return subs[expr]

        orig_expr = expr
        if expr in opt_subs:
            expr = opt_subs[expr]

        # If enabled, parse Muls and Adds arguments by order to ensure
        # replacement order independent from hashes
        if order != 'none':
            if isinstance(expr, (Mul, MatMul)):
                #c, nc = expr.args_cnc()
                #if c == [1]:
                #    args = nc
                #else:
                #    args = list(ordered(c)) + nc
                args = expr.args
            elif isinstance(expr, (Add, MatAdd)):
                args = list(ordered(expr.args))
            else:
                args = expr.args
        else:
            args = expr.args

        new_args = list(map(_rebuild, args))
        if new_args != args:
            new_expr = expr.func(*new_args)
        else:
            new_expr = expr

        if orig_expr in to_eliminate:
            try:
                sym = next(symbols)
            except StopIteration:
                raise ValueError("Symbols iterator ran out of symbols.")

            subs[orig_expr] = sym
            replacements.append((sym, new_expr))
            return sym

        else:
            return new_expr

    reduced_exprs = []
    for e in exprs:
        if isinstance(e, Basic):
            reduced_e = _rebuild(e)
        else:
            reduced_e = e
        reduced_exprs.append(reduced_e)

    return replacements, reduced_exprs


def cse(exprs, symbols=None, optimizations=None, postprocess=None,
        order='none'):
    """ Perform common subexpression elimination on an expression.

    Parameters
    ==========

    exprs : list of sympy expressions, or a single sympy expression
        The expressions to reduce.
    symbols : infinite iterator yielding unique Symbols
        The symbols used to label the common subexpressions which are pulled
        out. The ``numbered_symbols`` generator is useful. The default is a
        stream of symbols of the form "x0", "x1", etc. This must be an
        infinite iterator.
    optimizations : list of (callable, callable) pairs
        The (preprocessor, postprocessor) pairs of external optimization
        functions. Optionally 'basic' can be passed for a set of predefined
        basic optimizations. Such 'basic' optimizations were used by default
        in old implementation, however they can be really slow on larger
        expressions. Now, no pre or post optimizations are made by default.
    postprocess : a function which accepts the two return values of cse and
        returns the desired form of output from cse, e.g. if you want the
        replacements reversed the function might be the following lambda:
        lambda r, e: return reversed(r), e
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. If set to
        'canonical', arguments will be canonically ordered. If set to 'none',
        ordering will be faster but dependent on expressions hashes, thus
        machine dependent and variable. For large expressions where speed is a
        concern, use the setting order='none'.

    Returns
    =======

    replacements : list of (Symbol, expression) pairs
        All of the common subexpressions that were replaced. Subexpressions
        earlier in this list might show up in subexpressions later in this
        list.
    reduced_exprs : list of sympy expressions
        The reduced expressions with all of the replacements above.

    Examples
    ========

    >>> from sympy import cse, SparseMatrix
    >>> from sympy.abc import x, y, z, w
    >>> cse(((w + x + y + z)*(w + y + z))/(w + x)**3)
    ([(x0, y + z), (x1, w + x)], [(w + x0)*(x0 + x1)/x1**3])

    Note that currently, y + z will not get substituted if -y - z is used.

     >>> cse(((w + x + y + z)*(w - y - z))/(w + x)**3)
     ([(x0, w + x)], [(w - y - z)*(x0 + y + z)/x0**3])
    """
    if isinstance(exprs, Basic):
        exprs = [exprs]

    import time

    start = time.time()

    import logging

    logger = logging.getLogger(__name__)
    logger.info("cse: start")

    exprs = list(exprs)

    if optimizations is None:
        optimizations = []

    # Preprocess the expressions to give us better optimization opportunities.
    reduced_exprs = [preprocess_for_cse(e, optimizations) for e in exprs]

    if symbols is None:
        symbols = numbered_symbols()
    else:
        # In case we get passed an iterable with an __iter__ method instead of
        # an actual iterator.
        symbols = iter(symbols)

    # Remove symbols from the generator that conflict with names in the expressions.
    excluded_symbols = set().union(*[expr.atoms(Symbol) for expr in reduced_exprs])
    symbols = (symbol for symbol in symbols if symbol not in excluded_symbols)

    # Find other optimization opportunities.
    opt_subs = opt_cse(reduced_exprs)

    logger.info("cse: done after {}".format(time.time() - start))

    # Main CSE algorithm.
    replacements, reduced_exprs = tree_cse(reduced_exprs, symbols, opt_subs, order)

    # Postprocess the expressions to return the expressions to canonical form.
    for i, (sym, subtree) in enumerate(replacements):
        subtree = postprocess_for_cse(subtree, optimizations)
        replacements[i] = (sym, subtree)
    reduced_exprs = [postprocess_for_cse(e, optimizations) for e in reduced_exprs]

    if postprocess is None:
        return replacements, reduced_exprs

    return postprocess(replacements, reduced_exprs)
