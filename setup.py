from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "cdfbench._native",
        ["src/cdfbench.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
