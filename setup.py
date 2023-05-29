from setuptools import find_packages, setup

setup(
    name="rgpt",
    version="0.0.1",
    author="Louis Wendler",
    description="RecurrentGPT (unofficial)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="Apache License, Version 2.0",
    url="https://github.com/1ucky40nc3/recurrent_gpt",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
)
