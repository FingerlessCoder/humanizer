"""
Setup script for the humanizer package.
"""
from setuptools import setup, find_packages
import os
import sys

# Read the contents of README.md for long description
try:
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Text Humanizer - A package to make text more human-readable"

# Define package requirements
requirements = [
    "spacy>=3.5.0",
    "language-tool-python>=2.7.1",
    "nltk>=3.8.1",
    "scikit-learn>=1.0.2",
    "numpy>=1.21.0",
    "streamlit>=1.22.0",
]

setup(
    name="humanizer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    description="A package to make text more human-readable",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tay Wilson",
    python_requires=">=3.7",
    url="https://github.com/FingerlessCoder/humanizer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Text Processing :: Linguistic",
    ],
    # Add entry points for command-line usage
    entry_points={
        "console_scripts": [
            "humanize-text=humanizer.cli:main",  # This will create a 'humanize-text' command
        ],
    },
    # Include non-Python files
    package_data={
        "humanizer": ["config/*.json"],
    },
)
