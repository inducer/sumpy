from __future__ import division
from __future__ import absolute_import
import six
from six.moves import zip

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

import logging
logger = logging.getLogger(__name__)

__doc__ = """

Manipulating batches of assignments
-----------------------------------

.. autoclass:: SymbolicAssignmentCollection

"""


def _generate_unique_possibilities(prefix):
    yield prefix

    try_num = 0
    while True:
        yield "%s_%d" % (prefix, try_num)
        try_num += 1


class _SymbolGenerator(object):
    def __init__(self, taken_symbols):
        self.taken_symbols = taken_symbols
        self.generated_names = set()

    def __call__(self, base="expr"):
        for id_str in _generate_unique_possibilities(base):
            if id_str not in self.taken_symbols \
                    and id_str not in self.generated_names:
                self.generated_names.add(id_str)
                return sp.Symbol(id_str)

    def __iter__(self):
        return self

    def next(self):
        return self()

    __next__ = next


# {{{ CSE caching

def _map_cse_result(mapper, cse_result):
    replacements, reduced_exprs = cse_result

    new_replacements = [
            (sym, mapper(repl))
            for sym, repl in replacements]
    new_reduced_exprs = [
            mapper(expr)
            for expr in reduced_exprs]

    return new_replacements, new_reduced_exprs


def cached_cse(exprs, symbols):
    assert isinstance(symbols, _SymbolGenerator)

    from pytools.diskdict import get_disk_dict
    cache_dict = get_disk_dict("sumpy-cse-cache", version=1)

    # sympy expressions don't pickle properly :(
    # (as of Jun 7, 2013)
    # https://code.google.com/p/sympy/issues/detail?id=1198

    from pymbolic.interop.sympy import (
            SympyToPymbolicMapper,
            PymbolicToSympyMapper)

    s2p = SympyToPymbolicMapper()
    p2s = PymbolicToSympyMapper()

    key_exprs = tuple(s2p(expr) for expr in exprs)

    key = (key_exprs, frozenset(symbols.taken_symbols),
            frozenset(symbols.generated_names))

    try:
        result = cache_dict[key]
    except KeyError:
        from sumpy.cse import cse
        result = cse(exprs, symbols)
        cache_dict[key] = _map_cse_result(s2p, result)
        return result
    else:
        return _map_cse_result(p2s, result)

# }}}


# {{{ collection of assignments

class SymbolicAssignmentCollection(object):
    """Represents a collection of assignments::

        a = 5*x
        b = a**2-k

    In the above, *x* and *k* are external variables, and *a* and *b*
    are variables managed by this object.

    This is a stateful object, but the only state changes allowed
    are additions to *assignments*, and corresponding updates of
    its lookup tables.

    Note that user code is *only* allowed to hold on to *names* generated
    by this class, but not expressions using names defined in this collection.
    """

    def __init__(self, assignments=None):
        """
        :arg assignments: mapping from *var_name* to expression
        """

        if assignments is None:
            assignments = {}

        self.assignments = assignments

        self.symbol_generator = _SymbolGenerator(self.assignments)
        self.all_dependencies_cache = {}

        self.user_symbols = set()

    def __str__(self):
        return "\n".join(
            "%s <- %s" % (name, expr)
            for name, expr in six.iteritems(self.assignments))

    def get_all_dependencies(self, var_name):
        """Including recursive dependencies."""
        try:
            return self.all_dependencies_cache[var_name]
        except KeyError:
            pass

        if var_name not in self.assignments:
            return set()

        result = set()
        for dep in self.assignments[var_name].atoms():
            if not isinstance(dep, sp.Symbol):
                continue

            dep_name = dep.name
            if dep_name in self.assignments:
                result.update(self.get_all_dependencies(dep_name))
            else:
                result.add(dep)

        self.all_dependencies_cache[var_name] = result
        return result

    def add_assignment(self, name, expr, root_name=None, wrt_set=None):
        assert isinstance(name, str)
        assert name not in self.assignments

        if wrt_set is None:
            wrt_set = frozenset()
        if root_name is None:
            root_name = name

        self.assignments[name] = sp.sympify(expr)

    def assign_unique(self, name_base, expr):
        """Assign *expr* to a new variable whose name is based on *name_base*.
        Return the new variable name.
        """
        for new_name in _generate_unique_possibilities(name_base):
            if new_name not in self.assignments:
                break

        self.add_assignment(new_name, expr)
        self.user_symbols.add(new_name)
        return new_name

    def run_global_cse(self, extra_exprs=[]):
        logger.info("common subexpression elimination: start")

        assign_names = list(self.assignments)
        assign_exprs = [self.assignments[name] for name in assign_names]

        # Options here:
        # - cached_cse: Uses on-disk cache to speed up CSE.
        # - checked_cse: if you mistrust the result of the cse.
        #   Uses maxima to verify.
        # - sp.cse: The underlying sympy thing.
        #from sumpy.symbolic import checked_cse

        new_assignments, new_exprs = cached_cse(assign_exprs + extra_exprs,
                symbols=self.symbol_generator)

        new_assign_exprs = new_exprs[:len(assign_exprs)]
        new_extra_exprs = new_exprs[len(assign_exprs):]

        for name, new_expr in zip(assign_names, new_assign_exprs):
            self.assignments[name] = new_expr

        for name, value in new_assignments:
            assert isinstance(name, sp.Symbol)
            self.add_assignment(name.name, value)

        logger.info("common subexpression elimination: done")
        return new_extra_exprs

    def kill_trivial_assignments(self, exprs):
        logger.info("kill trivial assignments: start")

        approved_assignments = []
        rejected_assignments = []

        from sumpy.symbolic import is_assignment_nontrivial
        for name, value in six.iteritems(self.assignments):
            if name in self.user_symbols or is_assignment_nontrivial(name, value):
                approved_assignments.append((name, value))
            else:
                rejected_assignments.append((name, value))

        # un-substitute rejected assignments
        from sumpy.symbolic import make_one_step_subst
        unsubst_rej = make_one_step_subst(rejected_assignments)

        new_assignments = dict(
                (name, expr.subs(unsubst_rej))
                for name, expr in approved_assignments)

        exprs = [expr.subs(unsubst_rej) for expr in exprs]
        self.assignments = new_assignments

        logger.info("kill trivial assignments: done")

        return exprs

# }}}

# vim: fdm=marker
