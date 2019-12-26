from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

'''setup(name='Hello world app',
      ext_modules=cythonize("lib.pyx"),
      include_dirs=[numpy.get_include()])'''


setup(
    ext_modules=cythonize("std_c.pyx"),
    include_dirs=[numpy.get_include()]
)
