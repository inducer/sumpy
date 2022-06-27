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
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/modepy/": None,
    "https://documen.tician.de/pyopencl/": None,
    "https://documen.tician.de/pymbolic/": None,
    "https://documen.tician.de/loopy/": None,
    "https://documen.tician.de/pytential/": None,
    "https://documen.tician.de/boxtree/": None,
    "https://docs.sympy.org/latest/": None,
    "https://matplotlib.org/stable/": None,
}

nitpick_ignore_regex = [
        ["py:class", r"symengine\.(.+)"], # :cry:
        ]
