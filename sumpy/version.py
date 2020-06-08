from __future__ import division, absolute_import

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

# {{{ find install- or run-time git revision

import os
if os.environ.get("AKPYTHON_EXEC_FROM_WITHIN_WITHIN_SETUP_PY") is not None:
    # We're just being exec'd by setup.py. We can't import anything.
    _git_rev = None

else:
    import sumpy._git_rev as _git_rev_mod
    _git_rev = _git_rev_mod.GIT_REVISION

    # If we're running from a dev tree, the last install (and hence the most
    # recent update of the above git rev) could have taken place very long ago.
    from pytools import find_module_git_revision
    _runtime_git_rev = find_module_git_revision(__file__, n_levels_up=1)
    if _runtime_git_rev is not None:
        _git_rev = _runtime_git_rev

# }}}


VERSION = (2020, 1)
VERSION_STATUS = "beta1"
VERSION_TEXT = ".".join(str(x) for x in VERSION) + VERSION_STATUS

KERNEL_VERSION = (VERSION, _git_rev, 0)
