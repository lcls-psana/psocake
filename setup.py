#coding: utf8

"""
Setup script for psocake.
"""

from glob import glob
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='psocake',
      version='1.0.20',
      author="Chun Hong Yoon",
      author_email="yoon82@slac.stanford.edu",
      description='GUI for FEL analysis',
      url='https://github.com/lcls-psana/psocake',
      packages=["psocake"],
      package_dir={"psocake": "psocake"},
      package_data = {"psocake": ["data/graphics/*.gif"]},
      scripts=[s for s in glob('app/*') if not s.endswith('__.py')])
