#!/usr/bin/env python

import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="pydbow",
    version="0.1.2",
    description="Python Implementation of Bags of Binary Words",
    author="Göktuğ Karakaşlı",
    author_email="karakasligk@gmail.com",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/goktug97/PyDBoW",
    packages=["dbow"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy"],
)
