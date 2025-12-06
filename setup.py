"""
Setup script for Electronic Nose ML package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="electronic-nose-ml",
    version="1.0.0",
    author="Julia Vitiugova",
    author_email="vityugova.julia@physics.msu.ru",
    description="Neural network interpretability and calibration for gas sensor applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vitjuli/electronic-nose-ml",
    project_urls={
        "Bug Reports": "https://github.com/vitjuli/electronic-nose-ml/issues",
        "Source": "https://github.com/vitjuli/electronic-nose-ml",
    },
    packages=find_packages(where="."),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    keywords=[
        "machine learning",
        "neural networks",
        "interpretability",
        "calibration",
        "gas sensors",
        "feature importance",
        "deep learning",
        "PyTorch",
    ],
)
