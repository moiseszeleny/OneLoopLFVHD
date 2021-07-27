from setuptools import setup
from Cython.Build import cythonize

setup(
    name='root_c',
    ext_modules=cythonize("roots_c.pyx"),
    zip_safe=False,
)