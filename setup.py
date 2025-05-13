from setuptools import find_packages, setup

setup(
    name="cmb_likelihoods",
    version="1.0",
    description="CMB likelihoods for cobaya",
    author="Srini Raghunathan",
    author_email="sriniraghuna@gmail.com",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "astropy",
        "cobaya>=3.1",
    ],
    package_data={
        f"{lkl}": ["*.yaml"]
        for lkl in ["cmb_likelihoods"]
    },
)

