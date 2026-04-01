# Cobaya-based mock likelihoods for joint analyses of CMB, Type Ia Supernova, and BAO datasets.
## Paper:
* Raghunathan S., LSST-DESC 2026; arXiv:[2603.09973](https://arxiv.org/abs/2603.09973). 

## Overview
* ### Mock CMB likelihoods:
  * ### Available for SPT-3G, Advanced SO-Baseline, ASO-Goal and CMB-S4.
    * ###### Inspired and based on `cobaya`-based likelihood codes written by Jesus *Torrado*, Antony *Lewis*, Matthieu *Tristram* and Lennart *Balkenhol*.
  * ### Also supports the following:
    * #### SNe likelihoods (specifically DES and LSST SNe samples) using the same style as DES likelihoods implemented in `cobaya`.
    * #### BAO likelihoods (specifically DESI-DR2 and DESI-DR3 measurements) using the same style as DESI likelihoods implemented in `cobaya`.
     
---
## Data
* Input cosmology is $\Lambda {\rm CDM}$ based on Planck-2018 measurements $TT,TE,EE+lowE+lensing$ from Table-2 of [1807.06209](https://arxiv.org/pdf/1807.06209) but with $w_{0}=-1$ and $w_{a}=0$.
* ### CMB data:
  * Path: [data/cmb_data/binned_with_delta_l_100](https://github.com/sriniraghunathan/CMB_BAO_SNe_likelihoods/tree/main/data/cmb_data/binned_lmint300_lmaxt3500_lminp300_lmaxp3500_deltal100)
  * Mock bandpowers along with the bandpower covariance matrix and bandpower window function are included.
    * These use the internal linear combination datasets.
    * ILC weights for different freqeuency bands are included.
* ### SNe data:
  * Path: [data/sn_data.tar.gz](https://github.com/sriniraghunathan/CMB_BAO_SNe_likelihoods/blob/main/data/sn_data.tar.gz)
  * Contains LSST-Y3 mock data along with the covariances.
* ### BAO data:
  * Path: [data/bao_data](https://github.com/sriniraghunathan/CMB_BAO_SNe_likelihoods/tree/main/data/bao_data)
  * Contains DR2 and DR3.
---
## Installation
* #### `pip install .`
* To clean and reinstall:
  * `pip uninstall cmb_bao_sne_likelihoods` and then `pip install .`
---
## Requirements
* Should get automatically installed on doing `pip install .`. 
* `astropy`, `cobaya>=3.5`, `pyparsing>=2.0.2`, `camb>=1.5`
* Also, requires the standard python packages like `numpy`, `scipy`, etc.
---
## Example
* Try the following for a CMB-only likelihood:
  * `cobaya-run yamls/examples/so_baseline_TTEETEPP_w0walcdmsampler.yaml` to sample $w_{0} w_{a} \Lambda {\rm CDM}$ parameters with SO-Baseline-like data.
  * This also marginalises over nuisance (temperature and polarisation calibration parameters for all bands).
    * To fix them simply remove them from the yaml file. Check these excellent [cobaya_documentation] (https://cobaya.readthedocs.io/en/latest/cosmo_basic_runs.html) for more details.
* Try the following for a CMB+DESI-BAO likelihood:
  * `cobaya-run yamls/examples/so_baseline_TTEETEPP_desidr2baomock_w0walcdmsampler.yaml` to sample $w_{0} w_{a} \Lambda {\rm CDM}$ parameters with SO-Baseline-like and DESI-DR2-BAO-like data.
* Try the following for a CMB+DESI-BAO+LSST-SNe likelihood:
  * `cobaya-run yamls/examples/so_baseline_TTEETEPP_desidr2baomock_lssty3snemock_w0walcdmsampler.yaml` to sample $w_{0} w_{a} \Lambda {\rm CDM}$ parameters with SO-Baseline-like, DESI-DR2-BAO-like, and LSST-Y3-SNe-like data.

  
    
