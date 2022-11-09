import pathlib

import setuptools

DIR_PATH = pathlib.Path(__file__).parent
REQUIREMENTS_PATH = DIR_PATH / "requirements.txt"
README_PATH = DIR_PATH / "README.md"

setuptools.setup(
    name="memnet",
    version="0.0.1",
    packages=["memnet"],
    install_requires=REQUIREMENTS_PATH.read_text().splitlines(),
    url="https://github.com/joksas/memnet",
    project_urls={
        "Bug Tracker": "https://github.com/joksas/memnet/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    author="Dovydas Joksas",
    author_email="dovydas.joksas@ucl.ac.uk",
    description="A set of useful classes and functions for dealing with neural networks implented using memristive crossbar arrays.",
    long_description=README_PATH.read_text(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
)
