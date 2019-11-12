import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="calibration",
    version="0.0.1",
    author="Carl Andersson, David Widmann",
    author_email="carl.andersson@it.uu.se, david.widmann@it.uu.se",
    description="Tools for calibration evaluation",
    install_requires=['anytree', 'numpy'],
    test_suite="test",
    tests_require=['scipy'],
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uu-sml/calibration",
    packages=setuptools.find_packages(
        include=["calibration", "calibration.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering",
        "Operating System :: OS Independent",
    ],
)
