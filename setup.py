from setuptools import find_packages, setup

setup(
    name="cmb_bao_sne_likelihoods",
    version="1.0",
    description="CMB_BAO_LSST-SNe likelihoods for cobaya",
    author="Srini Raghunathan",
    author_email="sriniraghuna@gmail.com",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "astropy",
        "cobaya>=3.5",
        "pyparsing>=2.0.2",
        "camb>=1.5",
    ],
    package_data={
        f"{lkl}": ["*.yaml"]
        for lkl in ["cmb_likelihoods", "sne_likelihoods", "bao_likelihoods"]
    },
)

