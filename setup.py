from setuptools import setup, find_packages, Extension
import os


setup(
    name="numfire",
    version="0.0.1",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="A highly efficient autodiff library with a NumPy-like API",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/numfire.git",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "gdown",
        "numpy>=2.3,<2.4",
        "xpy @ git+https://github.com/kandarpa02/xpy.git@main",
    ],
    extras_require = {
    "cuda": ["cupy-cuda12x>=13.6.0,<14.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
    ],
    license="Apache-2.0",
    zip_safe=False,
)
