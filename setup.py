from setuptools import setup, find_packages

import setuptools

# this loads the version number from the dynamite/version.py module
version = open("popkinmocks/_version.py")
version = version.readlines()[-1].split()[-1].strip("\"'")

# load the readme as long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# load the package requirements from requirements.txt
with open("requirements.txt", "r") as fp:
    required = fp.read().splitlines()

setuptools.setup(
    name="popkinmocks",
    version=version,
    author="Prashin Jethwa",
    author_email="prashin.jethwa@univie.ac.at",
    description="mock IFU datacubes for stellar populations and kinematics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.univie.ac.at/dynamics/dynamite_docs/index.html",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    project_urls={
        "Source": "https://github.com/prashjet/popkinmocks/",
    },
    python_requires=">=3.6",
    # use the already parsed requirements from requirements.txt
    install_requires=required,
    # extra requirements for testing
    extras_require={
        "testing": [
            "pytest",
            "coverage",
        ],
        "docs": [
            "sphinx-book-theme",
            "myst-nb",
            "ipython",
        ],
    },
    package_data={"popkinmocks": ["../data/MILES_BASTI_CH_baseFe/*"]},
)
