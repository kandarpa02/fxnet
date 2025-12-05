# build.py
from setuptools import setup
from Cython.Build import cythonize

# compile_opts = {
#     "boundscheck": False,
#     "wraparound": False,
#     "cdivision": True,
#     "initializedcheck": False,
#     "nonecheck": False,
#     "language_level": "3",
# }

# setup(
#     name="faketensor",
#     ext_modules=cythonize(
#         "faketensor/src/Cython/utils.pyx",
#         compiler_directives=compile_opts,
#     )
# )
