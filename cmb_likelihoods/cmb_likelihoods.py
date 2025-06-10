"""General CMB likelihoods for forecasting
Author: Srini Raghunathan
email: sriniraghuna@gmail.com
Inspired from codes by Jesus Torrado, Antony Lewis, Matthieu Tristram and Lennart Balkenhol
"""

import itertools
import os
import re
from typing import Optional, Sequence

import numpy as np
from cobaya.conventions import packages_path_input
from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.log import LoggedError
from cobaya.theory import Theory

from . import tools

_do_plot = False #True #False ###True ##False ##True
if _do_plot:
    from pylab import *

class CMBmocks(InstallableLikelihood):
    install_options = {
        ##"download_url": "https://github.com/sriniraghunathan/CMB_SNIa_3x2pt_Fisher/cmb_likelihoods/",
        ##"data_path": "data/",
    }
    
    parent_data_folder: Optional[str] = 'data/'
    cmb_experiment_name: Optional[str] = 's4_wide'
    spectra_to_use: Optional[list] = ['TT', 'EE', 'TE']
    analytic_or_simbased_cov: Optional[str] = 'analytic'
    lmin: Optional[int] = 2
    lmax: Optional[int] = 5000
    cl_or_dl: Optional[str] = 'cl'
    delta_l: Optional[int]# = 100
    lmin_t: Optional[int] = 300
    lmin_p: Optional[int] = 300
    lmin_pp: Optional[int] = 30
    lmax_t: Optional[int] = 3500
    lmax_p: Optional[int] = 3500 #4000
    lmax_pp: Optional[int] = 3500 #4000
    """
    use_cosmopower: Optional[bool] = True
    cosmopowe_trained_dataset_fd: Optional[str] = 'data/SPT3G_2018_TTTEEE_cosmopower_trained_model_v1'
    """

    def initialize(self):
        if self.cmb_experiment_name in ['so_baseline', 'so_goal']:
            #self.freq_list = [30, 40, 90, 150, 220, 280]
            self.freq_list = [27, 39, 93, 145, 225, 280]
        elif self.cmb_experiment_name in ['s4_wide']:
            #self.freq_list = [30, 40, 90, 150, 220, 280]
            self.freq_list = [27, 39, 93, 145, 225, 278]
        elif self.cmb_experiment_name.find('spt3g_')>-1:
            self.freq_list = [90, 150, 220]
        self.spectra_to_use_str = ''.join(self.spectra_to_use)
        if self.delta_l == 1:
            self.unbinned = True
        else:
            self.unbinned = False

        '''
        if self.unbinned:
            self.data_folder = '%s/unbinned/%s/' %(self.parent_data_folder, self.cmb_experiment_name)
        else:
            self.data_folder = '%s/binned_with_delta_l_%s/%s/' %(self.parent_data_folder, self.delta_l, self.cmb_experiment_name)
        '''
        lmin_lmax_str = 'lmint%s_lmaxt%s_lminp%s_lmaxp%s' %(self.lmin_t, self.lmax_t, self.lmin_p, self.lmax_p)
        if 'PP' in self.spectra_to_use:
            lmin_lmax_str = '%s_lminphi%s_lmaxphi%s' %(lmin_lmax_str, self.lmin_pp, self.lmax_pp)

        if self.unbinned:
            self.data_folder = '%s/unbinned_%s/%s/' %(self.parent_data_folder, lmin_lmax_str, self.cmb_experiment_name)
        else:
            #self.data_folder = '%s/binned_with_delta_l_%s/%s/' %(self.parent_data_folder, self.delta_l, self.cmb_experiment_name)
            self.data_folder = '%s/binned_%s_deltal%s/%s/' %(self.parent_data_folder, lmin_lmax_str, self.delta_l, self.cmb_experiment_name)

        ###print(self.data_folder); quit()

        #self.ilc_weights_fname = '%s/binned_with_delta_l_%s/%s/%s_ilc_weights.npy' %(self.parent_data_folder, self.delta_l, self.cmb_experiment_name, self.cmb_experiment_name)
        self.ilc_weights_fname = '%s/binned_%s_deltal%s/%s/%s_ilc_weights.npy' %(self.parent_data_folder, lmin_lmax_str, self.delta_l, self.cmb_experiment_name, self.cmb_experiment_name)
        self.bp_file = '%s/%s_bandpowers_%s.txt' %(self.data_folder, self.cmb_experiment_name, self.spectra_to_use_str)
        assert self.analytic_or_simbased_cov in ['simbased', 'analytic']
        if self.analytic_or_simbased_cov == 'simbased':
            self.cov_file = '%s/%s_simbasedcovariance_%s.txt' %(self.data_folder, self.cmb_experiment_name, self.spectra_to_use_str)
            self.cov_inv_file = '%s/%s_simbasedcovariance_inv_%s.txt' %(self.data_folder, self.cmb_experiment_name, self.spectra_to_use_str)
        elif self.analytic_or_simbased_cov == 'analytic':
            self.cov_file = '%s/%s_covariance_%s.txt' %(self.data_folder, self.cmb_experiment_name, self.spectra_to_use_str)
            self.cov_inv_file = '%s/%s_covariance_inv_%s.txt' %(self.data_folder, self.cmb_experiment_name, self.spectra_to_use_str)
        self.window_file = '%s/%s_bpwf_%s.npy' %(self.data_folder, self.cmb_experiment_name, self.spectra_to_use_str)
        ###print(self.window_file); quit()

        # Read in bandpowers (remove index column)
        self.leff = np.loadtxt(self.bp_file, unpack=True)[0] #\ell_eff
        self.bandpowers_mat = np.loadtxt(self.bp_file, unpack=True)[1:] #TT, EE, TE

        #20250609 - set bandpower outside of the range to zero.
        bandpowers_mat_mod = []
        for speccntr, curr_bandpowers in enumerate( self.bandpowers_mat ):
            if speccntr == 0: #TT
                lmin_cut = self.lmin_t
                lmax_cut = self.lmax_t
            elif speccntr == 1: #EE
                lmin_cut = self.lmin_p
                lmax_cut = self.lmax_p
            elif speccntr == 3: #TE
                lmin_cut = min(self.lmin_t, self.lmin_p)
                lmax_cut = self.lmax_p
            elif speccntr == 3: #PP
                lmin_cut = self.lmin_pp
                lmax_cut = self.lmax_pp
            curr_bandpowers[self.leff<lmin_cut] = 0.
            curr_bandpowers[self.leff>lmax_cut] = 0.
            bandpowers_mat_mod.append( curr_bandpowers )
        self.bandpowers_mat = bandpowers_mat_mod


        if len(self.spectra_to_use) == 1:
            self.bandpowers = self.bandpowers_mat
        elif len(self.spectra_to_use) == 2:
            self.bandpowers = np.concatenate( (self.bandpowers_mat[0], self.bandpowers_mat[1]) )
        elif len(self.spectra_to_use) == 3:
            self.bandpowers = np.concatenate( (self.bandpowers_mat[0], self.bandpowers_mat[1], self.bandpowers_mat[2]) )
        elif len(self.spectra_to_use) == 4:
            self.bandpowers = np.concatenate( (self.bandpowers_mat[0], self.bandpowers_mat[1], self.bandpowers_mat[2], self.bandpowers_mat[3]) )

        #lensing
        self.add_dl_fac_for_lensing = False ##True

        # get BPWF
        if not self.unbinned:
            self.windows = np.load( self.window_file, allow_pickle = True).item()
        else:
            self.windows = None

        #make sure \ell ranges are fine.
        if self.use_cosmopower:
            assert self.lmax<=4999

        #covariance
        self.cov = np.loadtxt( self.cov_file )
        if self.cov_inv_file is not None:
            self.cov_inv = np.loadtxt( self.cov_inv_file )
        else:            
            self.cov_inv = np.linalg.inv( self.cov )

        #ILC weights
        if self.ilc_weights_fname is not None:
            self.ilc_weights_dic_full = np.load( self.ilc_weights_fname, allow_pickle = True ).item()

            #weights are defined over ell=0, 12000. Make them the same as ells here.
            ells = np.arange(self.lmin, self.lmax + 1)
            self.ilc_weights_dic = {} 
            for curr_spec in self.ilc_weights_dic_full:
                curr_ilc_weights_full = self.ilc_weights_dic_full[curr_spec]
                curr_ilc_weights = []
                for wl in curr_ilc_weights_full:
                    curr_ilc_weights.append( wl[ells] )
                self.ilc_weights_dic[curr_spec] = np.asarray( curr_ilc_weights )

        else:
            self.ilc_weights_dic = None

        if _do_plot:
            total_bins = len( self.leff )
            cl_err = np.diag( self.cov )**0.5
            cl_tt_err, cl_ee_err, cl_te_err = cl_err[:total_bins], cl_err[total_bins: 2*total_bins], cl_err[2*total_bins: 3*total_bins]
            ax = subplot(111, yscale = 'log')
            dl_fac = self.leff * (self.leff+1)/2/np.pi
            errorbar( self.leff, dl_fac * self.bandpowers_mat[0],  yerr = dl_fac * cl_tt_err, marker = '.', ls = 'None', capsize = 1.)
            errorbar( self.leff, dl_fac * self.bandpowers_mat[1],  yerr = dl_fac * cl_ee_err, marker = '.', ls = 'None', capsize = 1.)
            errorbar( self.leff, dl_fac * abs(self.bandpowers_mat[2]),  yerr = dl_fac * cl_te_err, marker = '.', ls = 'None', capsize = 1.)
            xlim(0, 5010); ylim(0., 1e4)
            show()
            quit()     

        ##from IPython import embed; embed();

    def get_requirements(self):
        # State requisites to the theory code
        return {"Cl": {cl: self.lmax for cl in self.spectra_to_use}}

    def apply_ilc_weights(self):
        pass

    def loglike(self, cl_cmb_dic, **params_values):

        lmin, lmax = self.lmin, self.lmax
        #ells = np.arange(lmin, lmax + 2)
        ells = np.arange(lmin, lmax + 1)

        ##from IPython import embed; embed(); quit

        cbs_or_dbs = [] ##np.empty_like(self.bandpowers)
        for i, curr_spec in enumerate(self.spectra_to_use):
            cl_cmb = cl_cmb_dic[curr_spec]
    
            #----
            #sum signals for bandpowers

            #CMB
            curr_cl_or_dl = cl_cmb[ells]
            #----

            #----
            """
            #apply an overall calibration as a check
            calibration = params_values.get(f"cal")
            curr_cl_or_dl *= calibration
            """
            #----

            #----
            #apply calibrations
            if curr_spec in ['TT', 'EE', 'TE']:
                def get_cal_arr(spec, bands, params_values):
                    if spec == 'TT':
                        cal_var_str = 'mapTCalxxx'
                    elif spec == 'EE':
                        cal_var_str = 'mapPCalxxx'
                    cal_arr = []
                    for bcntr, band in enumerate( bands ):
                        cal_var_name = cal_var_str.replace('xxx', str(bcntr+1))
                        cmd = 'params_values.get(f"%s")' %(cal_var_name)
                        curr_cal_val = eval( cmd )
                        ##print(cmd, curr_cal_val)
                        cal_arr.append( 1./curr_cal_val )
                    return np.asarray( cal_arr)

                if self.ilc_weights_fname is not None:
                    total_bands = len(self.freq_list)
                    if curr_spec in ['TT', 'EE']:
                        curr_ilc_weights = self.ilc_weights_dic[curr_spec]
                        assert len(curr_ilc_weights) == total_bands
                        map_cal_arr = get_cal_arr(curr_spec, self.freq_list, params_values)
                        ##from IPython import embed; embed()
                        cl_dic_for_ilc = tools.create_copies_of_cl_in_multiple_bands(curr_cl_or_dl, self.freq_list, keyname = curr_spec)
                        cl_dic_for_ilc = tools.apply_TPcal_to_cmb_spec_dic(cl_dic_for_ilc, curr_spec, self.freq_list, map_cal_arr = map_cal_arr)
                        curr_cl_or_dl_mod = tools.get_ilc_residual_using_weights(cl_dic_for_ilc, curr_ilc_weights, self.freq_list, el = ells)
                    elif curr_spec == 'TE':                        
                        ##from IPython import embed; embed()
                        curr_ilc_weights_T = self.ilc_weights_dic['TT']
                        curr_ilc_weights_P = self.ilc_weights_dic['EE']
                        assert len(curr_ilc_weights_T) == total_bands
                        assert len(curr_ilc_weights_P) == total_bands
                        map_cal_T_arr = get_cal_arr('TT', self.freq_list, params_values)
                        map_cal_P_arr = get_cal_arr('EE', self.freq_list, params_values)
                        cl_dic_for_ilc = tools.create_copies_of_cl_in_multiple_bands(curr_cl_or_dl, self.freq_list, keyname = curr_spec)
                        cl_dic_for_ilc = tools.apply_TPcal_to_cmb_spec_dic(cl_dic_for_ilc, curr_spec, self.freq_list, map_cal_arr = map_cal_T_arr, map_cal_arr2 = map_cal_P_arr)
                        curr_cl_or_dl_mod = tools.get_ilc_residual_using_weights(cl_dic_for_ilc, curr_ilc_weights, self.freq_list, el = ells)
                    curr_cl_or_dl = np.copy( curr_cl_or_dl_mod )
                else:
                    if curr_spec == "TT":
                        calibration = 1./( params_values.get(f"mapTCal") * params_values.get(f"mapTCal") )
                    elif curr_spec == "EE":
                        calibration = 1./( params_values.get(f"mapPCal") * params_values.get(f"mapPCal") )
                    elif curr_spec == "TE":
                        calibration = 0.5 * (
                            1
                            / (params_values.get(f"mapTCal") * params_values.get(f"mapPCal"))
                            + 1
                            / (params_values.get(f"mapTCal") * params_values.get(f"mapPCal"))
                        )
                    curr_cl_or_dl *= calibration
                ###print(curr_spec, calibration)
            #----

            #----
            if self.cl_or_dl == 'dl':
                dl_fac = ells * (ells+1)/2/np.pi
                curr_cl_or_dl = curr_cl_or_dl * dl_fac
            #----

            #----
            #apply cuts
            if curr_spec == 'TT':
                lmin_cut = self.lmin_t
                lmax_cut = self.lmax_t
            elif curr_spec == 'EE':
                lmin_cut = self.lmin_p
                lmax_cut = self.lmax_p
            elif curr_spec == 'TE':
                lmin_cut = min(self.lmin_t, self.lmin_p)
                lmax_cut = self.lmax_p
            elif curr_spec == 'PP':
                lmin_cut = self.lmin_pp
                lmax_cut = self.lmax_pp
            ##print(curr_spec, lmin_cut, lmax_cut); ###quit()
            curr_cl_or_dl[ells<lmin_cut] = 0.
            curr_cl_or_dl[ells>lmax_cut] = 0.
            
            #----

            #----
            # Binning via window and concatenate
            #from IPython import embed; embed()
            if not self.unbinned:
                window_reclen = np.shape(self.windows[curr_spec])[1]
                if len(curr_cl_or_dl)<window_reclen:
                    curr_cl_or_dl = np.interp(np.arange(window_reclen), ells, curr_cl_or_dl)
                curr_dbs = self.windows[curr_spec] @ curr_cl_or_dl
            else:
                curr_dbs = curr_cl_or_dl

            #20250609
            curr_dbs[self.leff<lmin_cut] = 0.
            curr_dbs[self.leff>lmax_cut] = 0.
            ###print(curr_dbs); quit()

            cbs_or_dbs.extend( curr_dbs )
            #----
        
        # Take the difference to the measured bandpower
        cbs_or_dbs = np.asarray( cbs_or_dbs )
        ###print(cbs_or_dbs); quit()
        
        if _do_plot:
            total_bins = len( self.leff )
            cl_err = np.diag( self.cov )**0.5
            cl_tt_err, cl_ee_err, cl_te_err = cl_err[:total_bins], cl_err[total_bins: 2*total_bins], cl_err[2*total_bins: 3*total_bins]
            if 'PP' in self.spectra_to_use:
                cl_pp_err = cl_err[3*total_bins:]
            clf()
            ax = subplot(111, yscale = 'log')
            dl_fac = 1. ##self.leff * (self.leff+1)/2/np.pi
            if self.cl_or_dl == 'dl':
                dl_fac = 1.
            errorbar( self.leff, dl_fac * self.bandpowers_mat[0],  yerr = dl_fac * cl_tt_err, marker = '.', ls = 'None', capsize = 1., color = 'black')
            if 'EE' in self.spectra_to_use:
                errorbar( self.leff, dl_fac * self.bandpowers_mat[1],  yerr = dl_fac * cl_ee_err, marker = '.', ls = 'None', capsize = 1., color = 'orangered')
            if 'TE' in self.spectra_to_use:
                errorbar( self.leff, dl_fac * abs(self.bandpowers_mat[2]),  yerr = dl_fac * cl_te_err, marker = '.', ls = 'None', capsize = 1., color = 'darkgreen')

            #theory now
            cl_tt_theory, cl_ee_theory, cl_te_theory = cbs_or_dbs[:total_bins], cbs_or_dbs[total_bins: 2*total_bins], cbs_or_dbs[2*total_bins: 3*total_bins]
            if 'PP' in self.spectra_to_use:
                cl_pp_theory = cbs_or_dbs[3*total_bins:]

            plot( self.leff, dl_fac * cl_tt_theory, color = 'black')
            if 'EE' in self.spectra_to_use:
                plot( self.leff, dl_fac * cl_ee_theory, color = 'orangered')
            if 'TE' in self.spectra_to_use:
                plot( self.leff, dl_fac * abs(cl_te_theory), color = 'darkgreen')

            xlim(0, 5010); #ylim(0.1, 1e4)
            show(); close()

            print(self.spectra_to_use)

            if 'PP' in self.spectra_to_use:
                clf()
                ax = subplot(111, yscale = 'log')
                dl_fac = (self.leff * (self.leff+1))**2./2/np.pi
                errorbar( self.leff, dl_fac * abs(self.bandpowers_mat[3]),  yerr = dl_fac * cl_pp_err, marker = '.', ls = 'None', capsize = 1., color = 'darkgreen')
                plot( self.leff, dl_fac * abs(cl_pp_theory), color = 'darkgreen')
                xlim(0, 5010); ylim(1e-10, 5e-7)
                show()
            quit()        
        
        delta_cb = cbs_or_dbs - self.bandpowers
        ###print(self.provider.get_param('ombh2')); print(delta_cb/self.bandpowers); quit()
        chi2 = (delta_cb @ self.cov_inv @ delta_cb.T)
        if np.ndim(chi2) == 1:
            chi2 = chi2[0]
        if np.ndim(chi2) == 2:
            chi2 = chi2[0][0]
        sign, slogdet = np.linalg.slogdet(self.cov)
        retval = -0.5 * (chi2 + slogdet )# + cal_prior)
        ###print( retval )
        ###print( "return value = %s" %(-0.5 * (chi2 + slogdet ) ))
        return retval


    def get_cmb_spectra_using_cosmopower(self, spectra):

        try:
            import cosmopower as cp
        except:
            pass

        if spectra == 'TT':
            trained_dataset_fname = '%s/cmb_spt_TT_NN' %(self.cosmopowe_trained_dataset_fd)
        elif spectra == 'EE':
            trained_dataset_fname = '%s/cmb_spt_EE_NN' %(self.cosmopowe_trained_dataset_fd)
        elif spectra == 'TE':
            trained_dataset_fname = '%s/cmb_spt_TE_PCAplusNN' %(self.cosmopowe_trained_dataset_fd)

        # load pre-trained PCA+NN model: maps cosmological parameters to CMB TE C_ell
        ###from IPython import embed; embed()
        if spectra in ['TT', 'EE']:
            cp_nn = cp.cosmopower_NN(restore=True, restore_filename=trained_dataset_fname)
        elif spectra == 'TE':
            cp_pca_nn = cp.cosmopower_PCAplusNN(restore=True, restore_filename=trained_dataset_fname)

        # create a dict of cosmological parameters
        params = {'ombh2': [self.provider.get_param('ombh2')],
                  'omch2': [self.provider.get_param('omch2')],
                  'h': [self.provider.get_param('H0')/100.],
                  'tau': [self.provider.get_param('tau')],
                  'ns': [self.provider.get_param('ns')],
                  'logA': [self.provider.get_param('logA')],
                  }

        
        # predictions (= forward pass through the network)
        if spectra in ['TT', 'EE']:
            spectra = cp_nn.ten_to_predictions_np(params)
        elif spectra == 'TE':
            spectra = cp_pca_nn.predictions_np(params)

        ###print(spectra); quit()

        return spectra[0]        

    def logp(self, **data_params):
        ###print(self.provider.get_param('ombh2'))
        if (0): #debug
            fix_cosmo_param_dic_debug = {'ombh2': 0.02237,
                                         'omch2': 0.1200, 
                                         'H0': 67.36, 
                                         'logA' :3.044, 
                                         'tau': 0.0544, 
                                         'ns': 0.9649, }
            self.provider.set_current_input_params(fix_cosmo_param_dic_debug)
            #from IPython import embed; embed();
            #quit()
        cl_cmb_specs = self.provider.get_Cl(ell_factor=False)
        cl_cmb_dic = {'TT': cl_cmb_specs.get("tt"), 'EE': cl_cmb_specs.get("ee"), 'TE': cl_cmb_specs.get("te")}
        if 'PP' in self.spectra_to_use:
            if self.add_dl_fac_for_lensing:
            ###from IPython import embed; embed(); sys.exit()
                el_ = cl_cmb_specs.get('ell')
                lensing_dl_fac = ( el_ * (el_+1) )**2. / 2/np.pi
                cl_cmb_dic['PP'] = cl_cmb_specs.get("pp") * lensing_dl_fac
            else:
                cl_cmb_dic['PP'] = cl_cmb_specs.get("pp")

        if _do_plot:
            total_bins = len( self.leff )
            cl_err = np.diag( self.cov )**0.5
            cl_tt_err, cl_ee_err, cl_te_err, cl_pp_err = cl_err[:total_bins], cl_err[total_bins: 2*total_bins], cl_err[2*total_bins: 3*total_bins], cl_err[3*total_bins:]

            clf()
            ax = subplot(111, yscale = 'log')
            dl_fac = self.leff * (self.leff+1)/2/np.pi
            errorbar( self.leff, dl_fac * self.bandpowers_mat[0],  yerr = dl_fac * cl_tt_err, marker = '.', ls = 'None', capsize = 1., color = 'black')
            errorbar( self.leff, dl_fac * self.bandpowers_mat[1],  yerr = dl_fac * cl_ee_err, marker = '.', ls = 'None', capsize = 1., color = 'orangered')
            errorbar( self.leff, dl_fac * abs(self.bandpowers_mat[2]),  yerr = dl_fac * cl_te_err, marker = '.', ls = 'None', capsize = 1., color = 'darkgreen')

            #theory now
            el_ = cl_cmb_specs.get('ell')
            dl_fac = el_ * (el_+1)/2/np.pi
            plot( el_, dl_fac * cl_cmb_dic['TT'], color = 'black')
            plot( el_, dl_fac * cl_cmb_dic['EE'], color = 'orangered')
            plot( el_, dl_fac * abs(cl_cmb_dic['TE']), color = 'darkgreen')

            xlim(0, 5010); ylim(0., 1e4)
            show()
            close()

            clf()
            ax = subplot(111, yscale = 'log')
            dl_fac = 1.
            errorbar( self.leff, dl_fac * self.bandpowers_mat[3],  yerr = dl_fac * cl_pp_err, marker = '.', ls = 'None', capsize = 1., color = 'black')

            #theory now
            el_ = cl_cmb_specs.get('ell')
            dl_fac = 1.
            plot( el_, dl_fac * cl_cmb_dic['PP'], color = 'black')

            xlim(0, 5010); 
            show()
            close()            
            quit()                   
        if _do_plot:
            total_bins = len( self.leff )
            cl_err = np.diag( self.cov )**0.5
            cl_tt_err, cl_ee_err, cl_te_err = cl_err[:total_bins], cl_err[total_bins: 2*total_bins], cl_err[2*total_bins: 3*total_bins]
            clf()
            ax = subplot(111, yscale = 'log')
            dl_fac = self.leff * (self.leff+1)/2/np.pi
            errorbar( self.leff, dl_fac * self.bandpowers_mat[0],  yerr = dl_fac * cl_tt_err, marker = '.', ls = 'None', capsize = 1., color = 'black')
            errorbar( self.leff, dl_fac * self.bandpowers_mat[1],  yerr = dl_fac * cl_ee_err, marker = '.', ls = 'None', capsize = 1., color = 'orangered')
            errorbar( self.leff, dl_fac * abs(self.bandpowers_mat[2]),  yerr = dl_fac * cl_te_err, marker = '.', ls = 'None', capsize = 1., color = 'darkgreen')

            #theory now
            el_ = cl_cmb_specs.get('ell')
            dl_fac = el_ * (el_+1)/2/np.pi
            plot( el_, dl_fac * cl_cmb_dic['TT'], color = 'black')
            plot( el_, dl_fac * cl_cmb_dic['EE'], color = 'orangered')
            plot( el_, dl_fac * abs(cl_cmb_dic['TE']), color = 'darkgreen')

            xlim(0, 5010); ylim(0., 1e4)
            show()
            close()
            quit()     

        return self.loglike(cl_cmb_dic, **data_params)


class spt3g_winter_TTEETE(CMBmocks):
    """
    Likelihood for SPT-3G Winter field ILC.
    """

class spt3g_winter_TTEETEPP(CMBmocks):
    """
    Likelihood for SPT-3G Winter field ILC with lensing.
    """

class spt3g_summer_TTEETE(CMBmocks):
    """
    Likelihood for SPT-3G summer field ILC.
    """

class spt3g_summer_TTEETEPP(CMBmocks):
    """
    Likelihood for SPT-3G summer field ILC with lensing.
    """

class spt3g_wide_TTEETE(CMBmocks):
    """
    Likelihood for SPT-3G wide field ILC.
    """

class spt3g_wide_TTEETEPP(CMBmocks):
    """
    Likelihood for SPT-3G wide field ILC with lensing.
    """

class spt3gpluswide_plus_spt3gwide_TTEETE(CMBmocks):
    """
    Likelihood for SPT-3G+ + SPT-3G wide field ILC.
    """

class spt3gpluswide_plus_spt3gwide_TTEETEPP(CMBmocks):
    """
    Likelihood for SPT-3G+ + SPT-3G wide field ILC with lensing.
    """


class so_baseline_TTEETE(CMBmocks):
    """
    Likelihood for SO-Baseline ILC.
    """

class so_baseline_TTEETEPP(CMBmocks):
    """
    Likelihood for SO-Baseline ILC with lensing.
    """

class so_goal_TTEETE(CMBmocks):
    """
    Likelihood for SO-Goal ILC.
    """

class so_goal_TTEETEPP(CMBmocks):
    """
    Likelihood for SO-Goal ILC with lensing.
    """

class s4_wide_TTEETE(CMBmocks):
    """
    Likelihood for S4-Wide ILC.
    """

class s4_wide_TTEETE_deltael50(CMBmocks):
    """
    Likelihood for S4-Wide ILC.
    """

class s4_wide_TTEETE_deltael200(CMBmocks):
    """
    Likelihood for S4-Wide ILC.
    """

class s4_wide_TTEETEPP(CMBmocks):
    """
    Likelihood for S4-Wide ILC with lensing.
    """

class advanced_so_baseline_TTEETE(CMBmocks):
    """
    Likelihood for Advanced-SO-Baseline ILC.
    """

class advanced_so_baseline_TTEETEPP(CMBmocks):
    """
    Likelihood for Advanced-SO-Baseline ILC with lensing.
    """

class advanced_so_goal_TTEETE(CMBmocks):
    """
    Likelihood for Advanced-SO-Goal ILC.
    """

class advanced_so_goal_TTEETEPP(CMBmocks):
    """
    Likelihood for Advanced-SO-Goal ILC with lensing.
    """

''' #in the works
class cosmopoweremulator(Theory):

    params = {'ombh2': None, 
              'nnu': None}


    def initialize(self):
        """called from __init__ to initialize"""

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """
        return {"Cl": {cl: CMBmocks.lmax for cl in CMBmocks.spectra_to_use}}

    def must_provide(self, **requirements):
        """
        """

    def get_cmb_spectra_using_cosmopower(self, spectra, **params_values):

        import cosmopower as cp

        if spectra == 'TT':
            trained_dataset_fname = '%s/cmb_spt_TT_NN' %(CMBmocks.cosmopowe_trained_dataset_fd)
        elif spectra == 'EE':
            trained_dataset_fname = '%s/cmb_spt_EE_NN' %(CMBmocks.cosmopowe_trained_dataset_fd)
        elif spectra == 'TE':
            trained_dataset_fname = '%s/cmb_spt_TE_PCAplusNN' %(CMBmocks.cosmopowe_trained_dataset_fd)

        # load pre-trained PCA+NN model: maps cosmological parameters to CMB TE C_ell
        ###from IPython import embed; embed()
        if spectra in ['TT', 'EE']:
            cp_nn = cp.cosmopower_NN(restore=True, restore_filename=trained_dataset_fname)
        elif spectra == 'TE':
            cp_pca_nn = cp.cosmopower_PCAplusNN(restore=True, restore_filename=trained_dataset_fname)

        # create a dict of cosmological parameters
        params = {'ombh2': [params_values.get('ombh2')],
                  'omch2': [params_values.get('omch2')],
                  'h': [params_values.get('H0')/100.],
                  'tau': [params_values.get('tau')],
                  'ns': [params_values.get('ns')],
                  'logA': [params_values.get('logA')],
                  }

        # predictions (= forward pass through the network)
        if spectra in ['TT', 'EE']:
            spectra = cp_nn.ten_to_predictions_np(params)
        elif spectra == 'TE':
            spectra = cp_pca_nn.predictions_np(params)

        ###print(spectra); quit()

        return spectra[0]

    def calculate(self, state, want_derived=True, **params_values):
        print(self.get_cmb_spectra_using_cosmopower('TT'))
        quit()
        state['Cl'] = {'TT': self.get_cmb_spectra_using_cosmopower('TT'),
                       'EE': self.get_cmb_spectra_using_cosmopower('EE'),
                       'TE': self.get_cmb_spectra_using_cosmopower('TE'),
                       }

        return self.current_state['Cl']
'''

