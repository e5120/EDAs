from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="eda",
    version="0.0.1",
    description="Estimation of Distribution Algorithms",
    author="sho shimazu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/e5120/EDAs",
    packages=["eda"],
    license="MIT",
)
