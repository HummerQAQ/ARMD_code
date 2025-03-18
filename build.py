import os
from distutils.command.build_ext import build_ext
from pathlib import Path

from Cython.Build import cythonize


# This function will be executed in setup.py:
def build(setup_kwargs):
    # The files you want to compile
    extensions = [
        p.__str__()
        for p in Path(".").rglob("*.py")
        if any([d in p.__str__() for d in ["lstm", "scripts", "exodus_common"]])
    ]

    # gcc arguments hack: enable optimizations
    os.environ["CFLAGS"] = "-O3"

    # Build
    setup_kwargs.update(
        {
            "ext_modules": cythonize(
                extensions,
                language_level=3,
                compiler_directives={"linetrace": True},
            ),
            "cmdclass": {"build_ext": build_ext},
        }
    )
