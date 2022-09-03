__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

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

from boxtree.array_context import PyOpenCLArrayContext as PyOpenCLArrayContextBase
from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        register_pytest_array_context_factory)

__doc__ = """
.. autoclass:: PyOpenCLArrayContext
"""


# {{{ PyOpenCLArrayContext

def make_loopy_program(
        domains, statements,
        kernel_data=None, *,
        name="sumpy_loopy_kernel",
        silenced_warnings=None,
        index_dtype=None,
        tags=None):
    """Return a :class:`loopy.LoopKernel` suitable for use with
    :meth:`ArrayContext.call_loopy`.
    """
    if kernel_data is None:
        kernel_data = [...]

    if silenced_warnings is None:
        silenced_warnings = []

    import loopy as lp
    from arraycontext.loopy import _DEFAULT_LOOPY_OPTIONS

    return lp.make_kernel(
            domains,
            statements,
            kernel_data=kernel_data,
            options=_DEFAULT_LOOPY_OPTIONS,
            default_offset=lp.auto,
            name=name,
            lang_version=lp.MOST_RECENT_LANGUAGE_VERSION,
            silenced_warnings=silenced_warnings,
            index_dtype=index_dtype,
            tags=tags)


class PyOpenCLArrayContext(PyOpenCLArrayContextBase):
    def transform_loopy_program(self, t_unit):
        for name in t_unit.entrypoints:
            options = t_unit[name].options
            if not (options.return_dict and options.no_numpy):
                raise ValueError(
                    f"loopy kernel '{name}' passed to call_loopy must "
                    "have 'return_dict' and 'no_numpy' options set. "
                    "Did you use 'make_loopy_program' to create this kernel?")

        return t_unit

# }}}


# {{{ pytest

def _acf():
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    return PyOpenCLArrayContext(queue, force_device_scalars=True)


class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext

    def __call__(self):
        # NOTE: prevent any cache explosions during testing!
        from sympy.core.cache import clear_cache
        clear_cache()

        return super().__call__()


register_pytest_array_context_factory(
    "sumpy.pyopencl",
    PytestPyOpenCLArrayContextFactory)

# }}}
