from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gauNEGF",
    version="1.0.0",
    author="William Livernois",
    author_email="willll@uw.edu",
    description="A Python package for Non-Equilibrium Green's Function calculations with Gaussian",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wliverno/GaussianNEGF",
    packages=["gauNEGF"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.18.0,<=1.26",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
        "jax>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black",
            "flake8",
        ],
    },
    package_data={
        "gauNEGF": ["*.bethe"],
    },
) 
