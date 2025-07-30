#
import numpy as np, sys, os, copy, scipy as sc
import astropy
from astropy import constants as const
from astropy import units as u
from astropy import coordinates as coord
from pylab import *
import copy
import sne_cmb_fisher_tools

def get_param_dict(paramfile):
    """
    Read input params file and store in a dict.

    Parameters:
    paramfile: paramfile to be read.

    Returns:
    param_dict: dictionary with parameter values
    """

    params, paramvals = np.genfromtxt(paramfile, delimiter = '=', unpack = True, autostrip = True, dtype='unicode')
    param_dict = {}
    for p,pval in zip(params,paramvals):
        if pval in ['T', 'True']:
            pval = True
        elif pval in ['F', 'False']:
            pval = False
        elif pval == 'None':
            pval = None
        else:
            try:
                pval = float(pval)
                if int(pval) == float(pval):
                    pval = int(pval)
            except:
                pass
        # replace unallowed characters in paramname
        p = p.replace('(','').replace(')','')
        param_dict[p] = pval
    return param_dict

def get_lsst_3x2pt_params(lsst_config_key):
    srd_y1 = ['omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'omega_b', 'h', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'a_0', 'beta', 'eta_low_z', 'eta_high_z']
    srd_y3 = ['omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'omega_b', 'h', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'a_0', 'beta', 'eta_low_z', 'eta_high_z']
    srd_y10 = ['omega_m', 'sigma_8', 'n_s', 'w_0', 'w_a', 'omega_b', 'h', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'b_6', 'b_7', 'b_8', 'b_9', 'b_10', 'a_0', 'beta', 'eta_low_z', 'eta_high_z']
    csw_y1 = ['a_s', 'h', 'n_s', 'omega_c_h2', 'omega_b_h2', 'w_0', 'w_a', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'a_0', 'beta', 'eta_low_z', 'eta_high_z']
    csw_y3 = ['a_s', 'h', 'n_s', 'omega_c_h2', 'omega_b_h2', 'w_0', 'w_a', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'a_0', 'beta', 'eta_low_z', 'eta_high_z']
    csw_y10 = ['a_s', 'h', 'n_s', 'omega_c_h2', 'omega_b_h2', 'w_0', 'w_a', 'b_1', 'b_2', 'b_3', 'b_4', 'b_5', 'b_6', 'b_7', 'b_8', 'b_9', 'b_10', 'a_0', 'beta', 'eta_low_z', 'eta_high_z']
    lsst_3x2pt_param_dic = {'srd_y1': srd_y1, 
                      'srd_y3': srd_y3, 
                      'srd_y10': srd_y10, 
                      'csw_y1': csw_y1, 
                      'csw_y3': csw_y3, 
                      'csw_y10': csw_y10, 
                        }

    return lsst_3x2pt_param_dic[lsst_config_key]


def combine_fisher( F_mat_arr, param_names_arr, small_diag_element = 1e-3):

    param_names_diff_dic = {'a_s': 'As', 'n_s': 'ns', 'omega_c_h2': 'omch2',  'omega_b_h2': 'ombh2', 'w_0': 'ws', 'w_a': 'wa'}
    param_names_arr_mod = []
    for curr_param_names in param_names_arr:
        curr_param_names_mod = []
        for p in curr_param_names:
            if p in param_names_diff_dic:
                curr_param_names_mod.append( param_names_diff_dic[p] )
            else:
                curr_param_names_mod.append(p)
        param_names_arr_mod.append(curr_param_names_mod)
    param_names_arr = np.copy(param_names_arr_mod)
    combined_param_names = np.unique(np.concatenate( param_names_arr ) )
    nparams = len(combined_param_names)
    ##print(combined_param_names, nparams); sys.exit()

    combined_F_mat = np.zeros((nparams, nparams))
    for curr_param_names, curr_F_mat in zip(param_names_arr, F_mat_arr):
        for pcntr1, p1 in enumerate( curr_param_names ):            
            for pcntr2, p2 in enumerate( curr_param_names ):
                curr_pind1 = np.where(combined_param_names == p1)[0][0]
                curr_pind2 = np.where(combined_param_names == p2)[0][0]
                combined_F_mat[curr_pind2, curr_pind1] += curr_F_mat[pcntr2, pcntr1]
    
    F_mat_mod_arr = []
    for curr_param_names, curr_F_mat in zip(param_names_arr, F_mat_arr):
        F_mat_mod = np.zeros((nparams, nparams))
        if small_diag_element is not None:
            small_diag_mat = np.zeros_like( F_mat_mod )
            np.fill_diagonal(small_diag_mat, small_diag_element)
            F_mat_mod = F_mat_mod + small_diag_mat
            ##print(F_mat_mod); sys.exit()

        for pcntr1, p1 in enumerate( curr_param_names ):            
            for pcntr2, p2 in enumerate( curr_param_names ):
                curr_pind1 = np.where(combined_param_names == p1)[0][0]
                curr_pind2 = np.where(combined_param_names == p2)[0][0]
                F_mat_mod[curr_pind2, curr_pind1] += curr_F_mat[pcntr2, pcntr1]
        F_mat_mod_arr.append(F_mat_mod)

    return combined_F_mat, combined_param_names, F_mat_mod_arr

def fix_params(F_mat, param_names, fix_params):

    #remove parameters that must be fixed    
    F_mat_refined = []
    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 in fix_params or p2 in fix_params: continue
            F_mat_refined.append( (F_mat[pcntr2, pcntr1]) )

    totparamsafterfixing = int( np.sqrt( len(F_mat_refined) ) )
    F_mat_refined = np.asarray( F_mat_refined ).reshape( (totparamsafterfixing, totparamsafterfixing) )

    param_names_refined = []
    for p in param_names:
        if p in fix_params: continue
        param_names_refined.append(p)

    return F_mat_refined, param_names_refined

def get_cosmo_derivatives_using_camb_v1(new_param_names, param_dict, param_dict_derivatives, which_spectra = 'lensed_scalar', cosmo_params = ['h0', 'm_nu', 'neff', 'ombh2', 'omch2', 'w0', 'wa', 'As', 'sigma_8']):

    params_cmd_dic = {\
        #'As_e-2tau': '(pars.InitPower.As) * np.exp( -2. * pars.Reion.optical_depth )',\
        'omegam': 'pars.omegam',\
        #'h': 'pars.h',\
        'omegab': 'pars.omegab',\
        'omegac': 'pars.omegac',\
        #'cosmomc_theta': 'results.cosmomc_theta()',\
        'sigma_8': 'results.get_sigma8_0()',\
        #'sigma_8': 'results.get_sigmaR(R=8., hubble_units=False, return_R_z=False)[0]', \
        'As': 'pars.InitPower.As', \
        'h0': 'pars.h',\
        'ombh2': 'pars.ombh2',\
        'omch2': 'pars.omch2',\
        'm_nu': 'pars.omnuh2 * ( camb.constants.neutrino_mass_fac / ( (pars.num_nu_massless + pars.num_nu_massive + (pars.nu_mass_degeneracies[0] - pars.nu_mass_fractions[0])) /3.)**0.75 )', \
        'neff': 'pars.num_nu_massless + pars.num_nu_massive + (pars.nu_mass_degeneracies[0] - pars.nu_mass_fractions[0])', \
        'w0': 'pars.DarkEnergy.w',\
        'wa': 'pars.DarkEnergy.wa',\
        }


    dauorip_daunewp_tmp = {}
    print('\n\t%s:' %(p))#, end =' ')
    for new_p in sorted(new_param_names):

        if (1):#cosmo_params is not None:
            if new_p not in cosmo_params:
                dauorip_daunewp_tmp[new_p] = 0.

        if new_p not in dauorip_daunewp_tmp:
            if new_p == p: 
                dauorip_daunewp_tmp[new_p] = 1.
            elif p_cmd is None:
                dauorip_daunewp_tmp[new_p] = 0.
            else:
                tmpdic = {}
                tmpdic[new_p] = param_dict_derivatives[new_p][0]
                pval1 = set_run_camb(param_dict, which_spectra, param_dict_derivatives = tmpdic, high_low = 0, param_name_and_cmd = param_name_and_cmd)
                pval2 = set_run_camb(param_dict, which_spectra, param_dict_derivatives = tmpdic, high_low = 1, param_name_and_cmd = param_name_and_cmd)

                dauorip_daunewp_tmp[new_p] = (pval2 - pval1) / (2*tmpdic[new_p])
                #print(tmpdic, dauorip_daunewp_tmp[p])
            print('\t\td%s_d%s = %s: command = %s' %(p.replace('_',''), new_p.replace('_',''), dauorip_daunewp_tmp[new_p], p_cmd))#, end =' ')
    
    return dauorip_daunewp_tmp        

def get_mod_param_names(param_name):

    param_names_dict = {'a_s': 'As', 'as': 'As',                        
                        'omega_m': 'omegam', 
                        'omega_c_h2': 'omch2', 
                        'omega_b_h2': 'ombh2', 
                        'omega_b': 'omegab', 
                        'omega_c': 'omegac', 
                        'n_s': 'ns', 
                        'w_0': 'ws', 
                        'w_a': 'wa', 
                        }

    if param_name in param_names_dict:
        return param_names_dict[param_name]
    else:
        return param_name

def get_cosmo_param_derivatives(param_ori, param_new, param_dict, param_steps_dict = None, stepsize_frac = 0.01, thetastar_or_cosmomctheta_or_h = 'h', lmax = 1000):

    """
    We will calculate the change in param_ori for an unit change in param_new.
    For example, if the old parameter is sigma_8 and the new parameter is As, then we will get dau_sigma_8 / dau_As using CAMB.
    """
    import copy

    camb_params_cmd_dic = {\
        'As_e-2tau': '(pars.InitPower.As) * np.exp( -2. * pars.Reion.optical_depth )',\
        'omegam': 'pars.omegam',\
        'h': 'pars.h',\
        'omegab': 'pars.omegab',\
        'omegac': 'pars.omegac',\
        'cosmomc_theta': 'results.cosmomc_theta()',\
        'sigma_8': 'results.get_sigma8_0()',\
        #'sigma_8': 'results.get_sigmaR(R=8., hubble_units=False, return_R_z=False)[0]', \
        'As': 'pars.InitPower.As', \
        'h0': 'pars.h',\
        'ombh2': 'pars.ombh2',\
        'omch2': 'pars.omch2',\
        'm_nu': 'pars.omnuh2 * ( camb.constants.neutrino_mass_fac / ( (pars.num_nu_massless + pars.num_nu_massive + (pars.nu_mass_degeneracies[0] - pars.nu_mass_fractions[0])) /3.)**0.75 )', \
        'neff': 'pars.num_nu_massless + pars.num_nu_massive + (pars.nu_mass_degeneracies[0] - pars.nu_mass_fractions[0])', \
        'w0': 'pars.DarkEnergy.w',\
        'wa': 'pars.DarkEnergy.wa',\
        }

    astropy_params_cmd_dic = None

    '''
    param_ori = get_mod_param_names(param_ori)
    param_new = get_mod_param_names(param_new)
    '''
    
    ##print( param_ori, param_new ); sys.exit()

    print('\nGet derivative of %s w.r.t %s' %(param_ori, param_new)); ###sys.exit()

    assert param_new in param_dict
    assert param_ori in camb_params_cmd_dic or param_ori in astropy_params_cmd_dic

    pval = param_dict[param_new]
    param_dict_low = copy.deepcopy( param_dict )
    param_dict_high = copy.deepcopy( param_dict )
    if param_steps_dict is not None:
        pval_step = param_steps_dict[param_new]
    else:
        if param_dict[param_new] == 0: #if the param's fiducial value if zero
            pval_step = stepsize_frac
        else:
            pval_step = pval * stepsize_frac
    pval_low = pval - pval_step
    pval_high = pval + pval_step
    param_dict_low[param_new] = pval_low
    param_dict_high[param_new] = pval_high    
    print('\t\tvalues: fid=%g, low=%g, high=%g, step=%g' %(pval, pval_low, pval_high, pval_step)); ##sys.exit()

    if param_ori in camb_params_cmd_dic: #run camb
        camb_pars_low, camb_results_low =  sne_cmb_fisher_tools.set_camb(param_dict_low, thetastar_or_cosmomctheta_or_h = thetastar_or_cosmomctheta_or_h, lmax = lmax)
        camb_pars_high, camb_results_high = sne_cmb_fisher_tools.set_camb(param_dict_high, thetastar_or_cosmomctheta_or_h = thetastar_or_cosmomctheta_or_h, lmax = lmax)

        cmd = camb_params_cmd_dic[param_ori]
        if cmd.find('results')>-1:
            cmd_for_low = cmd.replace('results.', 'camb_results_low.')
            cmd_for_high = cmd.replace('results.', 'camb_results_high.')
        elif cmd.find('pars')>-1:
            cmd_for_low = cmd.replace('pars.', 'camb_pars_low.')
            cmd_for_high = cmd.replace('pars.', 'camb_pars_high.')

    elif param_ori in astropy_params_cmd_dic: #run astropy
        cosmo_low = set_cosmo(param_dict_low)
        cosmo_high = set_cosmo(param_dict_high)
        print( cosmo_low ); sys.exit()

        cmd_for_low = cmd.replace('cosmo.', 'cosmo_low.')
        cmd_for_high = cmd.replace('cosmo.', 'cosmo_high.')

    result_low, result_high = eval( cmd_for_low ), eval( cmd_for_high )
    dauparamori_dauparamnew = (result_high - result_low) / (2*pval_step)

    return dauparamori_dauparamnew

def get_jacobian_transformation(F_mat, param_names, param_dict, param_old_new_arr):

    ny, nx = F_mat.shape
    J_mat = np.eye(ny)
    param_names = np.asarray( param_names )
    param_names_mod = [p for p in param_names]
    for param_old_new in param_old_new_arr:
        param_old, param_new = param_old_new
        ###print( param_old_new ); sys.exit()
        pind = np.where( param_names == param_old)[0][0]

        if param_old_new in [['log(1e10_a_s)', 'As'], ['log(1e10_a_s)', 'a_s']]:
            derivval = 1./param_dict[param_new]   
        elif param_old_new in [['As', 'log(1e10_a_s)'], ['As', 'logA'], ['a_s', 'log(1e10_a_s)']]:
            #derivval = param_dict[param_new] 
            derivval = 1./param_dict[param_old] #20250522 - check this again  
            ##print( derivval )  
        elif param_old_new == ['H0', 'h']:
            derivval = 100.
        elif param_old_new == ['h', 'H0']:
            derivval = 1./100.
        elif param_old_new == ['omch2', 'h']:
            derivval = 2 * param_dict['h'] * param_dict['omega_m']
        elif param_old_new in [ ['omch2', 'omega_m'], ['omega_c_h2', 'omega_m'], ['ombh2', 'omega_b'], ['omega_b_h2', 'omega_b'] ]:
            derivval = param_dict['h']**2.
        elif param_old_new in [ ['sigma_8', 'As'], ['omegab', 'ombh2'], ['omegam', 'omch2'] ]: #run CAMB/astropy for this
            #derivval = 1./param_dict['h']**2.
            param_old, param_new = param_old_new
            derivval = get_cosmo_param_derivatives(param_old, param_new, param_dict = param_dict)
        elif param_old_new in [ ['omega_b', 'ombh2'], ['omega_m', 'omega_b_h2'] ]:
            derivval = 1./param_dict['h']**2.

        ####print( param_old_new, derivval ); sys.exit()
        J_mat[pind, pind] = derivval
        #print( J_mat, param_new ); sys.exit()
        param_names_mod[pind] = param_new
    
    return J_mat, param_names_mod

def rotate_fisher_mat(F_mat, J_mat):

    return np.dot( J_mat.T, np.dot(F_mat, J_mat))

def rotate_fisher_mat_parent(F_mat_dic, param_names, param_dict, param_old_new_arr, fix_params_arr = None):

    #get the Jacobian first
    tmp_F_mat = list( F_mat_dic.values() )[0]
    J_mat, param_names_mod = get_jacobian_transformation(tmp_F_mat, param_names, param_dict, param_old_new_arr)
    ####print( J_mat.shape, param_names_mod, J_mat ); sys.exit()

    F_mat_dic_mod = {}
    for fmat_keyname in F_mat_dic:
        F_mat = F_mat_dic[fmat_keyname]
        F_mat_mod = rotate_fisher_mat( F_mat, J_mat ) 

        if fix_params_arr is not None:
            F_mat_mod, param_names_mod = fix_params(F_mat_mod, param_names_mod, fix_params_arr)
            ##print( param_names_mod ); sys.exit()

        F_mat_dic_mod[fmat_keyname] = F_mat_mod

    return F_mat_dic_mod, param_names_mod

def get_sigma_of_a_parameter(F_mat, param_names, desired_param_arr, prior_dic = None, fix_params_arr = None):

    F_mat_mod = np.copy(F_mat)
    if np.sum(F_mat_mod) == 0:
        sigma_vals = {}
        if desired_param_arr is not None:
            for desired_param in desired_param_arr:
                sigma_vals[desired_param] = 0.
        return sigma_vals

    param_names_mod = np.copy(param_names)

    param_names_mod = np.asarray( param_names_mod )
    fix_params_arr = np.asarray( fix_params_arr )

    if prior_dic is not None: #add priors.
        F_mat_mod = fn_add_prior(F_mat_mod, param_names_mod, prior_dic)

    if fix_params_arr is not None:
        F_mat_mod, param_names_mod = fix_params(F_mat_mod, param_names_mod, fix_params_arr)

    cov_mat = np.linalg.inv(F_mat_mod)
    param_names_mod = np.asarray( param_names_mod )

    sigma_vals = {}
    if desired_param_arr is not None:
        for desired_param in desired_param_arr:
            #print('\textract sigma(%s)' %(desired_param))
            pind = np.where(param_names_mod == desired_param)
            pcntr1, pcntr2 = pind, pind
            cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]
            cov_extract = np.asarray( [cov_mat[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
            sigma = cov_extract[0,0]**0.5
            
            sigma_vals[desired_param] = sigma

    
    #pind = np.where(param_names_mod == desired_param)[0][0]
    #pcntr1, pcntr2 = pind, pind
    #cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]
    #cov_extract = np.asarray( [cov_mat[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
    #sigma_val = cov_extract[0,0]**0.5

    return sigma_vals  


def format_axis(ax, fx, fy, maxxloc=None, maxyloc = None):
    """
    function to format axis fontsize.


    Parameters
    ----------
    ax: subplot axis.
    fx: fontsize for xaxis.
    fy: fontsize for yaxis.
    maxxloc: total x ticks.
    maxyloc: total y ticks.

    Returns
    -------
    formatted axis "ax".
    """
    for label in ax.get_xticklabels(): label.set_fontsize(fx)
    for label in ax.get_yticklabels(): label.set_fontsize(fy)
    if maxyloc is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=maxxloc))
    if maxxloc is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=maxxloc))

    return ax

def convert_param_to_latex(param):
    greek_words_small = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 
                        'lambda', 'mu', 'nu', 'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega']
    greek_words_captial = [w.capitalize() for w in greek_words_small]
    greek_words = greek_words_small + greek_words_captial
    math_words = ['z']

    tmp_param_split = param.split('_')
    if len( tmp_param_split ) == 1:
        latex_param = r'$%s$' %(param)
    else:
        tmpval = tmp_param_split[0]
        if tmpval in greek_words:
            tmpval = '\%s' %(tmpval)
        latex_param = '%s' %(tmpval)
        braces_arr = ''
        for tmpval in tmp_param_split[1:]:
            if tmpval in greek_words:
                tmpval = '\%s' %(tmpval)
            if tmpval in math_words:
                latex_param = '%s_{%s' %(latex_param, tmpval)
            else:
                latex_param = '%s_{\\rm %s' %(latex_param, tmpval)
            braces_arr = '%s}' %(braces_arr)
        latex_param = '%s%s' %(latex_param, braces_arr)
        latex_param  = r'$%s$' %(latex_param)

    return latex_param

def get_latex_param_str(param):
    params_str_dic= {\
    'norm_YszM': r'${\rm log}(Y_{\ast})$', 'alpha_YszM': r'$\alpha_{_{Y}}$',\
    'beta_YszM': r'$\beta_{_{Y}}$', 'gamma_YszM': r'$\gamma_{_{Y}}$', \
    'alpha': r'$\eta_{\rm v}$', 'sigma_8': r'$\sigma_{\rm 8}$', \
    'one_minus_hse_bias': r'$1-b_{\rm SZ}$', 
    'omega_m': r'$\Omega_{\rm m}$', 'omegam': r'$\Omega_{\rm m}$', \
    'h0':r'$h$', 'm_nu':r'$\sum m_{\nu}$', \
    'ombh2': r'$\Omega_{b}h^{2}$', 'omch2': r'$\Omega_{c}h^{2}$', 'omega_lambda': r'$\Omega_{\Lambda}$',
    'omega_b_h2': r'$\Omega_{b}h^{2}$', 'omega_c_h2': r'$\Omega_{c}h^{2}$',
    'omega_k': r'$\Omega_{k}$',
    'w0': r'$w_{0}$', 'wa': r'$w_{a}$', \
    'tau': r'$\tau_{\rm re}$', 
    'As': r'$A_{\rm s}$', 
    #'As': r'log$A_{\rm s}$', 
    'ns': r'$n_{\rm s}$', 'neff': r'$N_{\rm eff}$', \
    'mnu': r'$\sum m_{\nu}$', 'thetastar': r'$\theta_{\ast}$', \
    'h': r'$h$', 'omk': r'$\Omega_{k}$', 'ws': r'$w_{0}$', \
    'w_0': r'$w_{0}$', 'w_a': r'$w_{a}$', \
    'yhe': r'$Y_{P}$','nnu': r'N$_{\rm eff}$','omegak': r'$\Omega_{k}$',\
    'w': r'$w_{0}$', 'nrun': r'$n_{run}$', 'Aphiphi':r'$A^{\phi\phi}$', \
    'nnu': r'$N_{\rm eff}$', 'H0': r'$H_0$', \
    #adding more
    'a_s': r'$A_{\rm s}$', 'h': r'$h$', 'n_s': r'$n_{\rm s}$', \
    'omega_m': r'$\Omega_{m}$', 
    'omega_b': r'$\Omega_{b}$', 'omegab': r'$\Omega_{b}$',\
    #SNe
    'M': r'$M$', 'alpha': r'$\alpha$', 'beta': r'$\beta$',\
    }

    if param not in params_str_dic:
        return convert_param_to_latex(param)
    else:
        return params_str_dic[param]

def add_prior(F_mat, param_names, prior_dic):

    for pcntr1, p1 in enumerate( param_names ):
        for pcntr2, p2 in enumerate( param_names ):
            if p1 == p2 and p1 in prior_dic:
                prior_val = prior_dic[p1]
                F_mat[pcntr2, pcntr1] += 1./prior_val**2.

    return F_mat

def get_gaussian(mean, sigma, minx, maxx, delx = None):

    if delx is None: 
        delx = (maxx - minx)/1000000.

    ##print(mean, sigma, minx, maxx, delx)#; sys.exit()
    x = np.arange(minx, maxx, delx)

    #return x, 1./(2*np.pi*sigma)**0.5 * np.exp( -(x - mean)**2. / (2 * sigma**2.)  )
    return x, np.exp( -(x - mean)**2. / (2 * sigma**2.)  )

def get_ellipse_specs(COV, howmanysigma = 1):
    """
    Refer https://arxiv.org/pdf/0906.4123.pdf
    """
    assert COV.shape == (2,2)
    confsigma_dic = {1:2.3, 2:6.17, 3: 11.8}

    sig_x2, sig_y2 = COV[0,0], COV[1,1]
    sig_xy = COV[0,1]
    
    t1 = (sig_x2 + sig_y2)/2.
    t2 = np.sqrt( (sig_x2 - sig_y2)**2. /4. + sig_xy**2. )
    
    a2 = t1 + t2
    b2 = t1 - t2

    a = np.sqrt(abs(a2))
    b = np.sqrt(abs(b2))

    t1 = 2 * sig_xy
    t2 = sig_x2 - sig_y2
    theta = np.arctan2(t1,t2) / 2.
    
    alpha = np.sqrt(confsigma_dic[howmanysigma])
    
    #return (a*alpha, b*alpha, theta)
    return (a*alpha, b*alpha, theta, alpha*(sig_x2**0.5), alpha*(sig_y2**0.5))


def make_triangle_plot(F_dic, tr, tc, param_names, param_values_dict, desired_params_to_plot, one_or_two_sigma = 1, fix_axis_range_to_xxsigma = 5., fsval = 12, ncol = 2, noofticks = 4, color_dic = None, ls_dic = None, lw_dic = None, show_one_sigma_lab = True, use_percent = False, bias_dic = None, sort_alphabetical = False, mark_fid_lines = True, filled = False):

    """
    F_dic: Fisher matrix dictionary with experiment names as keys.
    tr: total rows.
    tc: total rows.
    param_values_dict: dictionary containing Fiducial values of cosmological parameters.
    desired_params_to_plot: parameters to be plotted.
    one_or_two_sigma: one or two or three or XX sigma region to be shown. Default is 1\sigma.
    fix_axis_range_to_xxsigma: x and y limits of axis will be fixed to xx\sigma. Default is 5\sigma.
    fsval: fontsize.
    noofticks: noofticks on axis.
    color_dic: Colours to be used for different experiments. If None, choose it automatically.
    ls_dic: line style for experiments. If None, we will use "-".
    lw_dic: line width for experiments. If None, we will use 1..
    lwval: Line width.
    show_one_sigma_lab: If True, parameter errors will be reported on 1d posteriors.
    use_percent: If True, then parameter errors will be reported as per cent on 1d posteriors.
    bias_dic: Bias on parameters. Not used for 3G forecasting paper. Defualt is None.
    """

    import matplotlib.patches as patches
    import warnings, matplotlib.cbook
    warnings.filterwarnings('ignore', category=matplotlib.cbook.mplDeprecation)

    ################################################
    ################################################
    #pick colours
    if color_dic is None:
        color_arr = ['navy', 'darkgreen', 'goldenrod', 'orangered', 'darkred']
        color_dic = {}
        for expcntr, expname in enumerate( F_dic ):
            color_dic[expname] = color_arr[expcntr]

    #linestyles
    if ls_dic is None:
        ls_dic = {}
        for expname in F_dic:
            ls_dic[expname] = '-'

    #linewidths
    if lw_dic is None:
        lw_dic = {}
        for expname in F_dic:
            lw_dic[expname] = 1.
    param_names_to_plot = []
    cosmo_param_pl_chars_dict = {}
    pcntr_for_plotting = 1
    if sort_alphabetical:
        for pp in sorted( desired_params_to_plot ):
            param_names_to_plot.append(pp)
            cosmo_param_pl_chars_dict[pp] = pcntr_for_plotting
            pcntr_for_plotting += 1
    else:
        for pp in desired_params_to_plot:
            param_names_to_plot.append(pp)
            cosmo_param_pl_chars_dict[pp] = pcntr_for_plotting
            pcntr_for_plotting += 1

    totparamstoplot = len(param_names_to_plot)
    diag_matrix = np.arange( totparamstoplot**2 ).reshape((totparamstoplot, totparamstoplot)) + 1
    ##print(diag_matrix); sys.exit()


    sbpl_locs_dic = {}
    for p1 in param_names_to_plot:
        for p2 in param_names_to_plot:
            sbpl_locs_dic[(p1,p2)] = cosmo_param_pl_chars_dict[p1] + ((cosmo_param_pl_chars_dict[p2]-1) * totparamstoplot)

    ##print(sbpl_locs_dic); sys.exit()

    widthvalues_for_axis_limits = {} #used later to fix axis ranges
    for pcntr1, p1 in enumerate( param_names ):
        widthvalues_for_axis_limits[p1] = 0.
        for pcntr2, p2 in enumerate( param_names ):        

            if p1 not in desired_params_to_plot or p2 not in desired_params_to_plot: continue
            '''
            if p1 in fix_params or p2 in fix_params: continue
            '''

            sbpl = sbpl_locs_dic[(p1,p2)]
            if sbpl not in np.tril(diag_matrix): continue

            cov_inds_to_extract = [(pcntr1, pcntr1), (pcntr1, pcntr2), (pcntr2, pcntr1), (pcntr2, pcntr2)]

            #fiducial values
            x = param_values_dict[p1]
            y = param_values_dict[p2]

            #x and y extents; \eplison_x and \epsilon_y for 1d Posteriors.
            deltax, deltay = 5*x, 5*y #some large range
            ##epsilon_x, epsilon_y = abs(x/10000.), abs(y/10000.) #for Gaussian 1d curve.
            if x == 0:
                deltax = 20.
            if y == 0:
                deltay = 20.
            x1, x2 = x - deltax, x + deltax
            y1, y2 = y - deltay, y + deltay

            if fix_axis_range_to_xxsigma is not None:
                x1, x2 = x - deltax*fix_axis_range_to_xxsigma*3, x + deltax*fix_axis_range_to_xxsigma*3
                y1, y2 = y - deltay*fix_axis_range_to_xxsigma*3, y + deltay*fix_axis_range_to_xxsigma*3
            else:
                x1, x2 = x - deltax, x + deltax
                y1, y2 = y - deltay, y + deltay

            #latex parameter labels
            p1str = get_latex_param_str(p1)
            p2str = get_latex_param_str(p2)

            ###print(p1, p2, p1str, p2str)

            #create subplot first
            ax = subplot(tr, tc, sbpl)#, aspect = 'equal')

            if sbpl<=(tr*(tc-1)):
                setp(ax.get_xticklabels(), visible=False)
            else:
                xlabel(p1str, fontsize = fsval);

            if ((sbpl-1)%tc == 0) and totparamstoplot>1 and sbpl!= 1:
                ylabel(p2str, fontsize = fsval);
            else:
                setp(ax.get_yticklabels(), visible=False)

            #print(p1, p2, sbpl)
            for expcntr, exp in enumerate( F_dic ):

                F_mat = F_dic[exp]
                #exp_COV = sc.linalg.pinv(F_mat)
                ##print(F_mat); sys.exit()
                exp_COV = np.linalg.inv(F_mat)
                ###print( exp_COV )

                #cov_extract = np.asarray( [exp_COV[ii] for ii in cov_inds_to_extract] ).reshape((2,2))
                cov_extract = []
                for ii in cov_inds_to_extract:
                    cov_extract.append(exp_COV[ii])
                cov_extract = np.asarray( cov_extract ).reshape((2,2))

                #if np.sum(cov_extract)<=1e-20: print(p1,p2, cov_extract); continue
                #print(p1, p2, cov_extract)

                colorval = color_dic[exp]
                lsval = ls_dic[exp]
                lwval = lw_dic[exp]
                alphaarr = [0.8, 0.3]
                for ss in range(one_or_two_sigma):

                    if p1 == p2:

                        widthval = cov_extract[0,0]**0.5##/2.35
                        ###print(p1, cov_extract); sys.exit()
                        ##print(p1, x1, x2)
                        hor, ver = get_gaussian(x, widthval, x1, x2)#, epsilon_x)

                        #labval = r'%.4f' %(widthval)
                        labval = None
                        if show_one_sigma_lab:
                            if bias_dic is not None:
                                #labval = r'%.3g (%.2g$\sigma$)' %(widthval, bias_dic[p1][1]/widthval)
                                labval = r'%.2g$\sigma$ shift' %(bias_dic[p1][1]/widthval)
                                ###print(labval); sys.exit()
                                #print(p1, labval, bias_dic[p1], widthval)
                            else:
                                widthval_for_label = widthval
                                if abs(x)>0. and use_percent:
                                    labval = r'%.3f\%%' %(100. * abs(widthval_for_label/x))
                                else:
                                    labval = r'%.3g' %(widthval_for_label)

                        #print(labval)
                        if totparamstoplot==1 and exp_dic is not None:
                            labval = r'%s: %s' %(exp_dic[exp][0], labval)
                        plot(hor, ver, color = colorval, lw = lwval, label = labval, ls = lsval)
                        if mark_fid_lines and expcntr == 0:
                            axvline(x, lw = 0.5, color = 'gray')
                        if ss == 0:
                            legend(loc = 4, framealpha = 1, fontsize = fsval-2, ncol = ncol, edgecolor = 'None', handletextpad=0.5, handlelength = 1.3, numpoints = 1, columnspacing = 0.8)

                        xlim(x1, x2)
                        ylim(0., 1.)
                        title(p1str, fontsize = fsval+2);

                        if p1 in widthvalues_for_axis_limits:
                            widthvalues_for_axis_limits[p1] = max(widthvalues_for_axis_limits[p1], widthval)

                        tick_params(axis='y', which = 'both', length = 0., width = 0.)
                        setp(ax.get_yticklabels(), visible=False); 
                    else:

                        Ep = get_ellipse_specs(cov_extract, howmanysigma = ss + 1)
                        widthval, heightval = Ep[0], Ep[1]

                        #if widthval<=1e-10 or heightval<=1e-10: continue
                        #print(widthval, heightval, p1, p2)
                        ellipse = patches.Ellipse(xy=[x,y], width=2.*widthval, height=2.*heightval, angle=np.degrees(Ep[2]))

                        ax.add_artist(ellipse)
                        ellipse.set_clip_box(ax.bbox)
                        if filled:
                            ellipse.set_facecolor(colorval)
                        else:
                            ellipse.set_facecolor('None')
                        ellipse.set_edgecolor(colorval)
                        ellipse.set_linewidth(lwval)
                        ellipse.set_linestyle(lsval)
                        ellipse.set_alpha(alphaarr[ss])

                        if mark_fid_lines and expcntr == 0:
                            axvline(x, lw = 0.5, color = 'gray')
                            axhline(y, lw = 0.5, color = 'gray')

                        xlim(x1, x2)
                        ylim(y1, y2)

                        '''
                        if exp.find('bias') == -1:
                            axhline(y, lw = 0.1);axvline(x, lw = 0.1)
                        '''
            if noofticks is not None:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=noofticks))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=noofticks))

            for label in ax.get_xticklabels(): label.set_fontsize(fsval-3.5)
            for label in ax.get_yticklabels(): label.set_fontsize(fsval-3.5)

            if (0):
                grid(True, which='major', axis = 'x', lw = 0.5, alpha = 0.1)
                grid(True, which='major', axis = 'y', lw = 0.5, alpha = 0.1)
    
    #set axis limits now based on widths obtained
    ##print(widthvalues_for_axis_limits); ##sys.exit()
    if fix_axis_range_to_xxsigma is not None:
        for pcntr1, p1 in enumerate( param_names ):
            for pcntr2, p2 in enumerate( param_names ):
                if p1 not in desired_params_to_plot or p2 not in desired_params_to_plot: continue
                #if p1 in fix_params or p2 in fix_params: continue
                ##if (not show_diagonal) and p1 == p2: continue
                sbpl = sbpl_locs_dic[(p1,p2)]                            
                if sbpl not in np.tril(diag_matrix): continue
                deltax, deltay = widthvalues_for_axis_limits[p1], widthvalues_for_axis_limits[p2]
                if deltax == 0. and deltay == 0.: continue
                '''
                x, deltax = param_dict[p1]
                y, deltay = param_dict[p2]
                '''
                x = param_values_dict[p1]
                y = param_values_dict[p2]
                x1, x2 = x - deltax*fix_axis_range_to_xxsigma, x + deltax*fix_axis_range_to_xxsigma
                y1, y2 = y - deltay*fix_axis_range_to_xxsigma, y + deltay*fix_axis_range_to_xxsigma
                ##if p1 == p2: print(widthvalues_for_axis_limits[p1], x, x1, x2)
                ax = subplot(tr, tc, sbpl)#, aspect = 'equal')
                xlim(x1, x2)
                if p1 != p2:
                    ylim(y1, y2)

    return color_dic, ls_dic

def get_kde(samples, xcol, ycol, xmin = None, xmax = None, ymin = None, ymax = None, xgridlen = 100, ygridlen = 100):
    """curr_samples = samples_dic[chainkeyname]
    p = curr_samples.getParamNames()
    def get_kde(samples, xcol, ycol, xmin = None, xmax = None, ymin = None, ymax = None, xgridlen = ):
        from scipy.stats import gaussian_kde
        x, y = curr_samples.samples[:, xcol], curr_samples.samples[:, ycol]
        #counts, xedges, yedges = np.histogram2d(x, y, bins = 100)
        kde = gaussian_kde([x,y])

        if xmin is None: xmin = x.min()
        if xmax is None: xmax = x.max()
        if ymin is None: ymin = y.min()
        if ymax is None: ymax = y.max()

        xgrid, ygrid = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kde(positions).T, xx.shape)

    lev1 = np.std(f)
    lev2 = 2*np.std(f)
    lev3 = 3*np.std(f)
    threshold = [np.mean(f), lev1]#, lev2]
    threshold = [lev1, lev2]

    cmap_val = cm.RdYlBu_r
    contourf(xx, yy, f, levels = [lev1, 1000*np.std(f)], linewidths = 1., linestyles = '-', colors = ['crimson'])
    contourf(xx, yy, f, levels = [lev2, 1000*np.std(f)], linewidths = 1., linestyles = '-', colors = ['black'])
    #contour(xx, yy, f, levels = [lev2, lev3], linewidths = 1., linestyles = '-', colors = ['red'])
    show()
    """

    from scipy.stats import gaussian_kde
    x, y = samples.samples[:, xcol], samples.samples[:, ycol]
    #counts, xedges, yedges = np.histogram2d(x, y, bins = 100)
    kde = gaussian_kde([x,y], bw_method='silverman')

    if xmin is None: xmin = x.min()
    if xmax is None: xmax = x.max()
    if ymin is None: ymin = y.min()
    if ymax is None: ymax = y.max()

    xarr = np.linspace(xmin, xmax, xgridlen)
    yarr = np.linspace(ymin, ymax, ygridlen)
    xgrid, ygrid = np.meshgrid(xarr, yarr)
    xy = np.vstack([xgrid.ravel(), ygrid.ravel()])
    zgrid = np.reshape(kde(xy).T, xgrid.shape)
    
    return xgrid, ygrid, zgrid

def make_contour(ax, xgrid, ygrid, zgrid, threshold_arr = [1, 2], filled = True, max_threshold = 1e4, alpha_arr = [0.3, 0.8], color_arr = ['darkgreen', 'darkgreen']):
    threshold_arr.append(max_threshold)
    level_arr = [np.mean(zgrid)+np.std(zgrid) * thresh for thresh in threshold_arr]
    if filled:
        which_contour = ax.contourf
    else:
        which_contour = ax.contour
    for cntr, (curr_level, curr_col, curr_alpha) in enumerate( zip( level_arr[:-1], color_arr, alpha_arr) ):
        print(curr_level, curr_col, curr_alpha)
        which_contour(xgrid, ygrid, zgrid, levels = [curr_level, level_arr[cntr+1]], linewidths = 1., linestyles = ['-'], colors = [curr_col], alpha = curr_alpha)
        if filled:
            ax.contour(xgrid, ygrid, zgrid, levels = [curr_level, level_arr[cntr+1]], linewidths = 0.5, linestyles = '-', colors = [curr_col], alpha = curr_alpha)
    return ax
