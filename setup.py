#!/usr/bin/env python
# -*- coding: latin1 -*-

from setuptools import setup

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    # 2.x
    from distutils.command.build_py import build_py

setup(name="sumpy",
      version="2013.1",
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
          "pytools>=2013.3",
          "pytest>=2.3",

          # FIXME leave out for now
          # https://code.google.com/p/sympy/issues/detail?id=3874
          #"sympy>=0.7.2",
          ],


      # 2to3 invocation
      cmdclass={'build_py': build_py})
