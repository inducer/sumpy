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


class _SymbolGenerator(object):

    def __init__(self, taken_symbols):
        self.taken_symbols = taken_symbols
        from collections import defaultdict
        self.base_to_count = defaultdict(lambda: 0)

    def __call__(self, base="expr"):
        count = self.base_to_count[base]

        def make_id_str(base, count):
            return "{base}{suffix}".format(
                    base=base,
                    suffix="" if count == 0 else "_" + str(count - 1))

        id_str = make_id_str(base, count)
        while id_str in self.taken_symbols:
            count += 1
            id_str = make_id_str(base, count)

        self.base_to_count[base] = count + 1

        return sp.Symbol(id_str)

    def __iter__(self):
        return self

    def next(self):
        return self()

    __next__ = next


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
        new_name = self.symbol_generator(name_base).name

        self.add_assignment(new_name, expr)
        self.user_symbols.add(new_name)
        return new_name

    def run_global_cse(self, extra_exprs=[]):
        logger.info("common subexpression elimination: start")

        assign_names = sorted(self.assignments)
        assign_exprs = [self.assignments[name] for name in assign_names]

        # Options here:
        # - checked_cse: if you mistrust the result of the cse.
        #   Uses maxima to verify.
        # - sp.cse: The sympy thing.
        # - sumpy.cse.cse: Based on sympy, designed to go faster.
        #from sumpy.symbolic import checked_cse

        from sumpy.cse import cse
        new_assignments, new_exprs = cse(assign_exprs + extra_exprs,
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
