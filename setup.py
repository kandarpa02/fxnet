from setuptools import setup, find_packages

setup(
    name="fxnet",
    version="0.0.1",
    author="Kandarpa Sarkar",
    author_email="kandarpaexe@gmail.com",
    description="A highly efficient autodiff library with a NumPy-like API",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kandarpa02/fxnet.git",
    packages=find_packages(),
    python_requires=">=3.8",

    install_requires=[
        "gdown",
        "numpy>=1.26,<2.3",
        "torch",
    ],

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
