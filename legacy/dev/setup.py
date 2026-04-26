"""
Implicit Correspondence Pose Estimation
========================================

A deep learning framework for 6-DOF camera pose estimation using
implicit 2D-3D correspondences between images and 3D Gaussian Splatting scenes.

Features:
- Keypoint-based pose regression aligned with ICL-I2PReg architecture
- Diversity loss to prevent keypoint collapse
- Support for both absolute and relative pose estimation
- Multi-GPU distributed training
- Integrated SplatLoc modules for feature decoding

Author: Your Name
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="implicit-correspondence",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="6-DOF camera pose estimation using implicit 2D-3D correspondences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/implicit_correspondence",
    packages=find_packages(exclude=["scripts", "docs", "output", "data"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ic-train=train:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.md"],
    },
)
