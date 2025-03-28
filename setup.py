"""
Setup script for TinySift.
"""

from setuptools import setup, find_packages

setup(
    name="tinysift",
    packages=find_packages(),
    package_data={"tinysift": ["py.typed"]},
)