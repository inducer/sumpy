#!/usr/bin/env python
# -*- coding: latin1 -*-

from setuptools import setup

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py

ver_dic = {}
version_file = open("sumpy/version.py")
try:
    version_file_contents = version_file.read()
finally:
    version_file.close()

exec(compile(version_file_contents, "sumpy/version.py", 'exec'), ver_dic)

setup(name="sumpy",
      version=ver_dic["VERSION_TEXT"],
      description="Fast summation in Python",
      long_description="""
      Code-generating FMM etc.
      """,
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Intended Audience :: Other Audience',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Visualization',
          'Topic :: Software Development :: Libraries',
          'Topic :: Utilities',
          ],

      author="Andreas Kloeckner",
      author_email="inform@tiker.net",
      license="MIT",
      packages=["sumpy", "sumpy.expansion"],

      install_requires=[
          "loo.py>=2013.1beta",
          "pytools>=2013.5.6",
          "boxtree>=2013.1",
          "pytest>=2.3",

          # If this causes issues, see:
          # https://code.google.com/p/sympy/issues/detail?id=3874
          "sympy>=0.7.2",
          ],


      # 2to3 invocation
      cmdclass={'build_py': build_py})
