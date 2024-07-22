from __future__ import annotations


__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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

from importlib import metadata

from pytools import find_module_git_revision


def _parse_version(version: str) -> tuple[tuple[int, ...], str]:
    import re

    m = re.match("^([0-9.]+)([a-z0-9]*?)$", VERSION_TEXT)
    assert m is not None

    return tuple(int(nr) for nr in m.group(1).split(".")), m.group(2)


VERSION_TEXT = metadata.version("sumpy")
VERSION, VERSION_STATUS = _parse_version(VERSION_TEXT)

_GIT_REVISION = find_module_git_revision(__file__, n_levels_up=1)
KERNEL_VERSION = (*VERSION, _GIT_REVISION, 0)
