def get_log_likelihood(model):
    diff = data - model
    logl = -0.5 * np.asarray( np.dot(diff.T, np.dot( cov_inv, diff ))).squeeze()
    return logl

def get_DM_model(sne_zarr, param_dict_for_model, baselinecosmo, use_hsq_units, theory = 'camb'):

    if theory == 'camb':
        pars, results = sne_cmb_fisher_tools.set_camb(param_dict_for_model, lmax = 10, WantTransfer = False)
        angular_diameter_distance = np.asarray( [results.angular_diameter_distance(z) for z in sne_zarr] )
        model = (5 * np.log10((1 + sne_zarr) * (1 + sne_zarr) * angular_diameter_distance * 1e6 / 10.))
    elif theory == 'astropy':
        cosmo = sne_cmb_fisher_tools.set_cosmo(param_dict_for_model, baselinecosmo = baselinecosmo, use_hsq_units = use_hsq_units)
        model = np.asarray( [cosmo.distmod(z).value for z in sne_zarr] )

    if 'scriptM' in param_dict_for_model:
        model += param_dict_for_model['scriptM']

    return model

def get_sne_dist_mod_likelihood(**param_values):
    import copy
    param_values = [param_values[p] for p in param_names]
    param_dict_sampler = copy.deepcopy( param_dict )
    for pcntr, ppp in enumerate( param_names ):
        param_dict_sampler[ppp] = param_values[pcntr]
    
    model = get_DM_model(sne_zarr, param_dict_sampler, baselinecosmo, use_hsq_units, theory = theory)

    res = get_log_likelihood(model)
    return res

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
#load modules
import numpy as np, sys, os, pandas as pd
import astropy
from astropy import constants as const
from astropy import units as u
from astropy import coordinates as coord
from astropy.cosmology import FlatLambdaCDM
import cobaya
import getdist
from getdist import plots, MCSamples
sys.path.append('modules/')
import sne_cmb_fisher_tools, misc
import argparse
import emcee

#-----------------------------------
parser = argparse.ArgumentParser(description='')

parser.add_argument('-sampler', dest='sampler', action='store', help='sampler', choices=['emcee', 'cobaya', 'grid_logl'], default= 'cobaya')
parser.add_argument('-theory', dest='theory', action='store', help='theory', choices=['astropy', 'camb'], default= 'astropy')
parser.add_argument('-force_resampling', dest='force_resampling', action='store', help='force_resampling', default= 1)
parser.add_argument('-debug_cobaya', dest='debug_cobaya', action='store', help='debug_cobaya', default= 0)
parser.add_argument('-tot_threads', dest='tot_threads', action='store', help='tot_threads', default= 20)
parser.add_argument('-which_cosmo', dest='which_cosmo', action='store', help='which_cosmo', type = str, choices=['hlcdm', 'lcdm', 'hw0lcdm', 'w0lcdm', 'hw0walcdm', 'w0walcdm'], default= 'w0walcdm')
parser.add_argument('-paramfile', dest='paramfile', action='store', help='paramfile', type = str, default= 'data/params_cobaya.ini')

#-----
#dataset
parser.add_argument('-datasets', dest='datasets', action='store', help='datasets',  choices=['sne', 'bao'], nargs = '+', default = ['sne'])

#-----

#-----
#BAO details
parser.add_argument('-bao_exp', dest='bao_exp', action='store', help='bao_exp', choices=['sdss', 'desi'], default = 'desi')
parser.add_argument('-bao_dr', dest='bao_dr', action='store', help='bao_dr', choices=[1, 2, 5], type = int, default = 2)
#-----

#-----
#sne details
parser.add_argument('-sne_exp', dest='sne_exp', action='store', help='sne_exp', type = str, choices=['lsst_binned', 'lsst_unbinned', 'des', 'des_cobaya', 'roman'])
parser.add_argument('-lsst_sim_no', dest='lsst_sim_no', action='store', help='lsst_sim_no', type = int, default= 1)
parser.add_argument('-cov_tag', dest='cov_tag', action='store', help='cov_tag', type = int, default= 0) #sys+stat
parser.add_argument('-use_ideal_data', dest='use_ideal_data', action='store', help='use_ideal_data', type = int, default= 0) #Ideal data or actual MU from the files
parser.add_argument('-fit_for_H0', dest='fit_for_H0', action='store', help='fit_for_H0', default= 0)
#DES stuff
parser.add_argument('-marginalize_abs_mag', dest='marginalize_abs_mag', action='store', help='marginalize_abs_mag', type = int, default= 0)
parser.add_argument('-fit_for_scriptM', dest='fit_for_scriptM', action='store', help='fit_for_scriptM', type = int, default= 1)
parser.add_argument('-add_weights', dest='add_weights', action='store', help='add_weights', type = int, default= 1)
#cuts
parser.add_argument('-zmin', dest='zmin', action='store', help='zmin', type = float, default= -1)
parser.add_argument('-zmax', dest='zmax', action='store', help='zmax', type = float, default= -1)
#-----

#-----
#convergence
parser.add_argument('-GRstat', dest='GRstat', action='store', help='GRstat', type = float, default= 0.05)
#-----


args = parser.parse_args()
args_keys = args.__dict__
for kargs in args_keys:
    param_value = args_keys[kargs]
    if isinstance(param_value, str):
        cmd = '%s = "%s"' %(kargs, param_value)
    else:
        cmd = '%s = %s' %(kargs, param_value)
    exec(cmd)
os.environ.get('OMP_NUM_THREADS',str(tot_threads))
if fit_for_scriptM: assert not marginalize_abs_mag

#-----------------------------------
print('\nget the params as dict')
param_dict = misc.get_param_dict(paramfile)
if 'H0' in param_dict:
    param_dict['h'] = param_dict['H0'] / 100.
param_dict['omega_m'] = param_dict['omch2'] / param_dict['h']**2.

if switch_to_sne_input_cosmo:
    """
       init_HzFUN_INFO  
             H0         = 70.00      # km/s/Mpc 
             OM, OL, Ok = 0.31500, 0.68500, 0.00000 
             w0, wa     = -1.000,  0.000 
    """    
    param_dict['h'] = 0.7
    param_dict['omega_m'] = 0.31500
    param_dict['omch2'] = param_dict['omega_m'] * param_dict['h']**2.
    #print(param_dict['omch2']); sys.exit()

#-----------------------------------
#variable / file names
#input folder for SNe data.
fd = 'data/sn_data/%s/' %(sne_exp)
#output folder and chainname
mcmc_input_params_info_dict, chainname, op_fd, output = sne_cmb_fisher_tools.get_params_chainame_folder(datasets, param_dict, which_cosmo, 
                                                    sne_exp, 
                                                    bao_exp, bao_dr, 
                                                    theory = theory, 
                                                    sampler = sampler, 
                                                    zmin = zmin, zmax = zmax, lsst_sim_no = lsst_sim_no,
                                                    add_weights = add_weights, fit_for_scriptM = fit_for_scriptM, marginalize_abs_mag = marginalize_abs_mag, use_ideal_data = use_ideal_data, 
                                                    )

print('chains will be under %s' %(output)); ##sys.exit()

if not os.path.exists( op_fd ): os.system('mkdir -p %s' %(op_fd))

baselinecosmo_dic = {'lcdm': 'FlatLambdaCDM', 'w0walcdm': 'Flatw0waCDM', 'w0lcdm': 'FlatwCDM', 
                     'hlcdm': 'FlatLambdaCDM', 'hw0walcdm': 'Flatw0waCDM', 'hw0lcdm': 'FlatwCDM', 
                     }
baselinecosmo = baselinecosmo_dic[which_cosmo]

#-----------------------------------
#load data
#get sne details
print('\nget sne details')
sne_details_dic = sne_cmb_fisher_tools.get_sne_details(sne_exp, unbinned_sne_sim_no = lsst_sim_no, obtain_covs = True, zmin = zmin, zmax = zmax)

sne_cov_tag_dic = sne_details_dic['sne_cov_tag_dic']
##print(sne_cov_tag_dic); sys.exit()
sne_zarr = sne_details_dic['sne_zarr']
sne_muarr = sne_details_dic['sne_muarr']
sne_muerr_stat = sne_details_dic['sne_muerr_stat']
print(sne_details_dic.keys())
###print(sne_details_dic['sne_tot_cov'].keys()); sys.exit()
sne_cov = sne_details_dic['sne_tot_cov'][cov_tag]
data = np.copy( sne_muarr )

if add_weights:
    weights = np.asarray( sne_muerr_stat**2. )
    np.fill_diagonal(sne_cov, sne_cov.diagonal() + weights)
cov_inv = np.linalg.inv(sne_cov)

if marginalize_abs_mag: #cobaya-style
    deriv = np.ones_like(sne_muarr)[:, None]
    derivp = cov_inv.dot(deriv)
    fisher = deriv.T.dot(derivp)
    cov_inv = cov_inv - derivp.dot(np.linalg.solve(fisher, derivp.T))

#-----------------------------------
#params

input_info = {}
input_info["params"] = mcmc_input_params_info_dict


param_names = list(input_info["params"].keys())
print( param_names ); ##sys.exit()

input_info["likelihood"] = {}
if 'sne' in datasets:
    input_info["likelihood"]["sne_dist_mod"] = {
                    "external": get_sne_dist_mod_likelihood, 
                    #"input_params": param_names,
                    }
if 'bao' in datasets:
    desi_likelihood_keyname = 'bao.%s_dr%s.desi_bao_all' %(bao_exp, bao_dr)
    input_info["likelihood"][desi_likelihood_keyname] = ''

input_info["sampler"] = {"mcmc": {"drag": False, "Rminus1_stop": GRstat, "max_tries": 5000}}
if 'bao' in datasets:
    input_info["theory"] = {
      "camb":{
            "extra_args": {
              "num_massive_neutrinos": 1,
              "nnu": 3.044,
              "halofit_version": "mead2020",
              "bbn_predictor": "PArthENoPE_880.2_standard.dat",
              "lens_potential_accuracy": 4,
              "lens_margin": 1250,
              "dark_energy_model": "ppf",
                        }
            }
    }       
input_info["output"] = output
input_info["output"] = output
updated_info, sampler = cobaya.run(input_info, force = force_resampling, debug = debug_cobaya)
print('Done.')
