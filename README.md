# CMB mock likelihoods for cobaya

## Overview
* ### Mock CMB likelihoods for SPT-3G, SO-Baseline, SO-Goal and CMB-S4.
  * ### Also supports SNe likelihoods (specifically LSST SNe sample) using the same style as DES likelihoods implemented in `cobaya`.
  * ##### Inspired and based on `cobaya`-based likelihood codes written by Jesus *Torrado*, Antony *Lewis*, Matthieu *Tristram* and Lennart *Balkenhol*.
---
## Data
* ### CMB data:
  * Path: [data/cmb_data/binned_with_delta_l_100](https://github.com/sriniraghunathan/CMB_cobaya_likelihoods_and_sampling/data/cmb_data/binned_with_delta_l_100/)
  * Mock bandpowers along with the bandpower covariance matrix and bandpower window function are included.
    * These use the internal linear combination datasets.
    * ILC weights for different freqeuency bands are included.
* ### SNe data:
  * Currently private.
* ### BAO data:
  * Path: [data/bao_data/desi_bao_dr2/desi_bao_dr2_mock](https://github.com/sriniraghunathan/CMB_cobaya_likelihoods_and_sampling/data/bao_data/desi_bao_dr2/desi_bao_dr2_mock))
  * Same as the ones released by DESI DR2 [link](https://github.com/CobayaSampler/bao_data/desi_bao_dr2)
  * But also including $w_{0} w_{a} \Lambda {\rm CDM}$ expectation. This can be found [here](https://github.com/sriniraghunathan/CMB_cobaya_likelihoods_and_sampling/data/bao_data/desi_bao_dr2/desi_bao_dr2_mock/desi_gaussian_bao_ALL_GCcomb_mean_camb.txt)
    * Input cosmology is {\it Planck}-2018 $TT,TE,EE+lowE+lensing$ from Table-2 of [1807.06209](https://arxiv.org/pdf/1807.06209) but with $w_{0}=-1$ and $w_{a}=0$.
    * This could be useful in generating combined mock-likelihoods for BAO, and also for other datasets.
---
## Installation
* #### `pip install .`
* To clean and reinstall:
  * `pip uninstall cmb_sne_likelihoods` and then `pip install .`
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

  
    
