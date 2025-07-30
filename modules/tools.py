import numpy as np, os, sys
import numpy as np, sys, os, copy, scipy as sc, pandas as pd
import sne_cmb_fisher_tools
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u
#import cobaya
#from cobaya.likelihoods import base_classes
#from cobaya.conventions import Const#, packages_path_input



def bao_model(param_dict, z_arr, observable_arr, cosmo = None, camb_results = None, camb_or_astropy = 'camb'):
    """
    need the following quantities
    transverse comoving distance D_M = cosmo.comoving_distance(z).value
    angular diameter distance D_a = D_M/(1+z). This is actually not needed.
    hubble distance D_H = c/H0 = cosmo.hubble
    Eq. (2) of https://arxiv.org/pdf/2503.14738.
    rd = 147.05 Mpc * (ombh2/0.02236)**-0.13 * (ommh2/0.1432)**-0.23 * (neff/3.04)**-0.1
    """
    from astropy import constants

    """
    if cosmo is None:
        param_values = [omch2, ws, wa]
        param_dict_sampler = copy.deepcopy( param_dict )
        for pcntr, ppp in enumerate( param_names ):
            param_dict_sampler[ppp] = param_values[pcntr]
        cosmo = sne_cmb_fisher_tools.set_cosmo(param_dict_sampler, baselinecosmo = baselinecosmo)
    """

    #compute the models
    ombh2 = param_dict['ombh2']
    ommh2 = param_dict['omch2'] + param_dict['ombh2']
    neff = param_dict['neff']
    model_arr =[]
    for (z, o) in zip( z_arr, observable_arr ):

        if camb_or_astropy == 'astropy':
            curr_DM = cosmo.comoving_distance(z).value #Transverse comoving distance
            curr_DH = ( constants.c.to('km/s')/cosmo.H(z) ).value #Hubble distance
        elif camb_or_astropy == 'camb':
            curr_DM = camb_results.angular_diameter_distance(z) * (1+z) #Transverse comoving distance
            curr_DH = ( constants.c.to('km/s')/camb_results.hubble_parameter(z) ).value #Hubble distance

        """
            return np.cbrt(
                ((1 + z) * self.provider.get_angular_diameter_distance(z)) ** 2 *
                Const.c_km_s * z / self.provider.get_Hubble(z,
                                                            units="km/s/Mpc")) / self.rs()
        """

        #sound drag - Eq. (2) of https://arxiv.org/pdf/2503.14738
        curr_rd = 147.05 * (ombh2/0.02236)**-0.13 * (ommh2/0.1432)**-0.23 * (neff/3.04)**-0.1
        curr_rd = 147.0330237648529
        
        #Iso BAO distance: Below Eq.(12) of https://arxiv.org/pdf/2503.14738
        curr_Dv = (z * curr_DH * curr_DM**2. )**(1/3.) 
        
        if o == 'DV_over_rs':
            curr_model = curr_Dv / curr_rd
        elif o == 'DM_over_rs':
            curr_model = curr_DM / curr_rd
        elif o == 'DH_over_rs':
            curr_model = curr_DH / curr_rd
        model_arr.append( curr_model )
    model_arr = np.asarray( model_arr )

    return model_arr

def get_bao_derivatives(redshift, observable, param_dict, params, stepsize_frac = 0.01):
    """    
    obtain derivatives of BAO w.r.t cosmological parameters

    redshift: float - redshift for which BAO distances are being measured.
    observable: list - BAO observables.
    param_dict: dictionary with parameter values
    params: params to calcualte the derivatives for
    stepsize_frac: stepsize to be used for parameters for the finite difference method.
        Default: 10 per cent of the fiducial parameter value (stepsize_frac * fiducial_parameter_value)

    Returns:
    bao_deriv_dict: BAO derivative dictionary. Keys are redshift values.
    """

    import copy


    bao_deriv_dic = {}
    for ppp in params:
        pval = param_dict[ppp]
        ##print(ppp, pval)
        if pval == 0:
            pval_step = stepsize_frac
        else:
            pval_step = pval * stepsize_frac

        #run twice for finite difference method
        param_dict_mod = copy.deepcopy( param_dict )
        bao_distance_arr = []
        for diter in range(2):
            if diter == 0: #pval - step
                param_dict_mod[ppp] = pval - pval_step
            elif diter == 1: #pval + step
                param_dict_mod[ppp] = pval + pval_step

            #set cosmo
            pars, camb_results = sne_cmb_fisher_tools.set_camb(param_dict_mod, lmax = 10, WantTransfer = False)
            cosmo = None

            #get BAO distance value now
            curr_model_val = bao_model(param_dict, [redshift], [observable], cosmo = cosmo, camb_results = camb_results)[0]
            ##print(curr_model_val); sys.exit()

            bao_distance_arr.append( curr_model_val )

        deriv_val = ( bao_distance_arr[1] - bao_distance_arr[0] ) / (2 * pval_step)

        bao_deriv_dic[ppp] = deriv_val
    return bao_deriv_dic

def get_bao_fisher(params, covariance, bao_deriv_dic, cov_inv = None):


    npar = len(params)
    F = np.zeros([npar,npar])

    if cov_inv is None:
        #cov_inv = sc.linalg.pinv2(covariance)
        cov_inv = sc.linalg.pinv(covariance)

    ##print( cov_inv ); sys.exit()

    param_combinations = []
    for pcnt,p in enumerate(params):
        for pcnt2,p2 in enumerate(params):
            param_combinations.append([p,p2, pcnt, pcnt2])

    bao_keys = list(bao_deriv_dic.keys())
    for (p1,p2, pcnt1, pcnt2) in param_combinations:

        if pcnt2<pcnt1:continue

        der1 = np.asarray( [bao_deriv_dic[k][p1] for k in bao_keys] )
        der2 = np.asarray( [bao_deriv_dic[k][p2] for k in bao_keys] )

        ###print( der1, der2 ); sys.exit()

        curr_val = der1 @ ( cov_inv @ der2 )

        F[pcnt2,pcnt1] += curr_val
        if pcnt1 != pcnt2:
            F[pcnt1,pcnt2] += curr_val


    return F     