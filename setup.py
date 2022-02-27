from setuptools import setup, find_packages, find_namespace_packages
import logging

p0 = find_packages(where="src")
p2 = find_namespace_packages(
    where="src",
    #include=["hydra_plugins.*"],
)

setup(
    packages=p0 + p2,
    package_dir={
        "": "src",
    },
    #install_requires=["pyttitools-adabins", "pyttitools-gma"],
)