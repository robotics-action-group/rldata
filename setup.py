from setuptools import setup, find_packages

setup(
    name="rldata",
    version="0.1.0",
    description="A Python package for RL data handling.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "tensorflow-datasets>=4.8.2",
        "torch>=1.13.1",
    ],
    python_requires=">=3.7",
    license="MIT",
    url="https://github.com/robotics-action-group/rldata",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
