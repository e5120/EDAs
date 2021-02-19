from setuptools import setup


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="eda",
    version="1.0.0",
    description="Estimation of Distribution Algorithms",
    author="sho shimazu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/e5120/EDAs",
    packages=["eda"],
    license="MIT",
)
