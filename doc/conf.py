from importlib import metadata
from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2016-21, sumpy contributors"
release = metadata.version("sumpy")
version = ".".join(release.split(".")[:2])

intersphinx_mapping = {
    "arraycontext": ("https://documen.tician.de/arraycontext/", None),
    "boxtree": ("https://documen.tician.de/boxtree/", None),
    "loopy": ("https://documen.tician.de/loopy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pymbolic": ("https://documen.tician.de/pymbolic/", None),
    "pyopencl": ("https://documen.tician.de/pyopencl/", None),
    "pytential": ("https://documen.tician.de/pytential/", None),
    "python": ("https://docs.python.org/3/", None),
    "pytools": ("https://documen.tician.de/pytools/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}

nitpick_ignore_regex = [
    ["py:class", r"symengine\.(.+)"],  # :cry:
]
