from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize


def get_extensions():
    root = Path(__file__).parent
    PACKAGE = "faketensor"

    SRC_DIRS = [
        root / PACKAGE / "src",
        root / PACKAGE / "pyderiv",
        root / PACKAGE / "modules",
        root / PACKAGE / "utils",
    ]

    exts = []
    for d in SRC_DIRS:
        if not d.exists():
            continue
        for pyx in d.rglob("*.pyx"):
            rel = pyx.with_suffix("").relative_to(root / PACKAGE).parts
            mod = PACKAGE + "." + ".".join(rel)
            exts.append(Extension(mod, [str(pyx)], language="c++"))

    return exts



setup(
    name="faketensor",
    ext_modules=cythonize(get_extensions(), language_level="3"),
)
