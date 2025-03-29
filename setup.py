"""
Setup script for tiny-sift.
"""

from setuptools import setup, find_packages

setup(
    name="tiny-sift", packages=find_packages(), package_data={"tiny_sift": ["py.typed"]}
)
