from setuptools import setup, find_packages

setup(
    name="vgslify",
    version="0.10.0",
    author="Tim Koornstra",
    author_email="tim.koornstra@gmail.com",
    description="VGSLify is a Python toolkit designed for rapid prototyping "
    "and seamless conversion between TensorFlow models and the Variable-size "
    "Graph Specification Language (VGSL). Drawing inspiration from "
    "Tesseract's VGSL specs, VGSLify introduces enhancements and provides a "
    "streamlined approach to define, train, and interpret deep learning "
    "models using VGSL.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TimKoornstra/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
    ],
    python_requires=">=3.9",
)
