from setuptools import find_packages, setup

subpackages = find_packages()

setup(
    name="panopticon",
    version="0.1",
    packages=["panopticon"] + [
        f"panopticon.{package}"
        for package in subpackages
    ],
    package_dir={"panopticon": "."},
)
