import os
from urllib.request import urlopen

_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2016-21, sumpy contributors"

os.environ["AKPYTHON_EXEC_FROM_WITHIN_WITHIN_SETUP_PY"] = "1"
ver_dic = {}
exec(compile(open("../sumpy/version.py").read(), "../sumpy/version.py", "exec"),
        ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
release = ver_dic["VERSION_TEXT"]

intersphinx_mapping = {
    "arraycontext": ("https://documen.tician.de/arraycontext/", None),
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "modepy": ("https://documen.tician.de/modepy/", None),
    "pyopencl": ("https://documen.tician.de/pyopencl/", None),
    "pymbolic": ("https://documen.tician.de/pymbolic/", None),
    "loopy": ("https://documen.tician.de/loopy/", None),
    "pytential": ("https://documen.tician.de/pytential/", None),
    "boxtree": ("https://documen.tician.de/boxtree/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

nitpick_ignore_regex = [
    ["py:class", r"symengine\.(.+)"],  # :cry:
]
