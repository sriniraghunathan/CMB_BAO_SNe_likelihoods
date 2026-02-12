#--------------------------------------------
def get_exp_details_using_ilc(els, cmb_experiment):
    if cmb_experiment == 'so_baseline':
        #ilc_fname = 'data/cmb/SO/sobaseline_ilc_cmb_27-39-93-145-225-280_TT-EE_fsky-1_-1years.npy'
        ilc_fname = 'data/cmb/ilc/sobaseline_ilc_cmb_27-39-93-145-225-280_TT-EE_fsky-1_-1years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'so_goal':
        #ilc_fname = 'data/cmb/SO/sogoal_ilc_cmb_27-39-93-145-225-280_TT-EE_fsky-1_-1years.npy'
        ilc_fname = 'data/cmb/ilc/sogoal_ilc_cmb_27-39-93-145-225-280_TT-EE_fsky-1_-1years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'advanced_so_baseline':
        ilc_fname = 'data/cmb/ilc/advanced_sobaseline_ilc_cmb_27-39-93-145-225-278_TT-EE_fsky-1_-1years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'advanced_so_goal':
        ilc_fname = 'data/cmb/ilc/advanced_sogoal_ilc_cmb_27-39-93-145-225-278_TT-EE_fsky-1_-1years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 's4_wide' or cmb_experiment == 's4_wide_cobaya_tester':
        #ilc_fname = 'data/cmb/cmbs4/s4wide_202310xx_pbdr_config/s4wide_202310xx_pbdr_config_ilc_galaxy0_27-39-93-145-225-278_TT-EE_lmax12000_for7years.npy'
        #ilc_fname = 'data/cmb/ilc/s4wide_202310xx_pbdr_config/s4wide_202310xx_pbdr_config_ilc_galaxy0_27-39-93-145-225-278_TT-EE_lmax12000_for7years.npy'
        ilc_fname = 'data/cmb/ilc/s4wide_202310xx_pbdr_config_ilc_cmb_27-39-93-145-225-278_TT-EE_fsky-1_-1years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'spt3g_winter':
        #ilc_fname = 'data/cmb/spt/spt_proposal_2023_ilc_cmb_90-150-220_TT-EE_fsky1500_7years.npy'
        ilc_fname = 'data/cmb/ilc/spt_proposal_2023_ilc_cmb_90-150-220_TT-EE_fsky1500_7years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'spt3g_summer':
        #ilc_fname = 'data/cmb/spt/spt_proposal_2023_summer_field_a_ilc_cmb_90-150-220_TT-EE_fsky1210_4years.npy'
        ilc_fname = 'data/cmb/ilc/spt_proposal_2023_summer_field_a_ilc_cmb_90-150-220_TT-EE_fsky1210_4years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'spt3g_wide':
        #ilc_fname = 'data/cmb/spt/spt_proposal_2023_ilc_cmb_90-150-220_TT-EE_fsky6000_1years.npy'
        ilc_fname = 'data/cmb/ilc/spt_proposal_2023_ilc_cmb_90-150-220_TT-EE_fsky6000_1years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'spt3g+wide_plus_spt3gwide':
        ilc_fname = 'data/cmb/ilc/spt3g+wide_plus_spt3gwide_202505xx_ilc_cmb_90-150-220_TT-EE_fsky6000_1years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'spt3g_summer2y':
        #ilc_fname = 'data/cmb/spt/spt_proposal_2023_summer_field_a_ilc_cmb_90-150-220_TT-EE_fsky1210_4years.npy'
        ilc_fname = 'data/cmb/ilc/spt_proposal_2023_summer_field_a_ilc_cmb_90-150-220_TT-EE_fsky1210_2years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500
    elif cmb_experiment == 'spt3g_winter2y':
        #ilc_fname = 'data/cmb/spt/spt_proposal_2023_ilc_cmb_90-150-220_TT-EE_fsky1500_7years.npy'
        ilc_fname = 'data/cmb/ilc/spt_proposal_2023_ilc_cmb_90-150-220_TT-EE_fsky1500_2years.npy'
        lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing = 300, 4000, 3500

    ilc_dic = np.load( ilc_fname, allow_pickle = True ).item()
    
    beam_noise_dic = ilc_dic['beam_noise_dic']['T']

    #details
    exp_details_dic = {}

    #els
    exp_details_dic['els'] = els

    #bands
    exp_details_dic['nu_arr'] = list(beam_noise_dic.keys())
    ###print(exp_details_dic['nu_arr']); sys.exit()

    #beams
    exp_details_dic['bl_dic'] = {}
    for nu in exp_details_dic['nu_arr']:
        beamval, noiseval = beam_noise_dic[nu]
        curr_bl = H.gauss_beam( np.radians( beamval/60. ), lmax = lmax)[:lmax]
        exp_details_dic['bl_dic'][nu] = curr_bl

    #ilc
    exp_details_dic['ilc_dic'] = {}
    el_ = ilc_dic['el']
    cl_residual_tt = ilc_dic['cl_residual']['TT']
    cl_residual_ee = ilc_dic['cl_residual']['EE']
    exp_details_dic['ilc_dic']['TT'] = np.interp(els, el_, cl_residual_tt)
    exp_details_dic['ilc_dic']['EE'] = np.interp(els, el_, cl_residual_ee)
    exp_details_dic['ilc_dic']['TE'] = np.zeros( len(els) )
    exp_details_dic['weights_dic'] = ilc_dic['weights']

    #lensing
    lensing_fname_suff = '_lmin%s_lmax%s_lmaxtt%s.npy' %(lmin_for_lensing, lmax_for_lensing, lmaxtt_for_lensing)
    lensing_fname = ilc_fname.replace('ilc/', 'lensing/').replace('.npy', lensing_fname_suff)
    lensing_dic = np.load( lensing_fname, allow_pickle = True, encoding = 'latin1' ).item()
    el_, nl_mv, clkk = lensing_dic['els'], lensing_dic['Nl_MV'], lensing_dic['cl_kk']

    #20250529 - these have the (l*(l+1.))**2/(2.*np.pi)
    lensing_dl_fac = (el_*(el_+1.))**2/(2.*np.pi)
    nl_mv = nl_mv / lensing_dl_fac
    clkk = clkk / lensing_dl_fac

    exp_details_dic['lensing_nl_mv'] = np.interp(els, el_, nl_mv)
    exp_details_dic['lensing_cl_kk'] = np.interp(els, el_, clkk)

    return exp_details_dic

def get_covariance_cmb_spectra(el, TT, EE, TE, PP = 0., TP = 0., EP = 0., fsky = 1., delta_el = 1., add_lensing = 1):

    if add_lensing:
        C = np.zeros( (4, 4) ) #TT, EE, TE, PP
    else:
        C = np.zeros( (3, 3) ) #TT, EE, TE, PP
    C[0,0] = TT**2.
    C[1,1] = EE**2
    C[2,2] = 1/2 * (TE**2. + TT*EE)
    if add_lensing:
        C[3,3] = PP**2.

    C[0,1] = C[1,0] = TE**2.
    C[0,2] = C[2,0] = TT*TE
    if add_lensing:
        C[0,3] = C[3,0] = 0.

    C[1,2] = C[2,1] = EE*TE
    if add_lensing:
        C[1,3] = C[3,1] = 0.

    if add_lensing:
        C[2,3] = C[3,2] = 0.

    C = C * (2 / (2 * el + 1) / fsky / delta_el)

    return C

def get_covariance_cmb_spectra_single(el, spec, fsky = 1., delta_el = 1.):

    C = np.zeros( (1,1) ) #TT
    C[0,0] = spec**2.
    C = C * (2 / (2 * el + 1) / fsky / delta_el)

    return C

#--------------------------------------------

import numpy as np, os, sys, healpy as H
#sys.path.append('modules/')
sys.path.append('/Users/sraghunathan/Research/SPTpol/analysis/git/CMB_BAO_SNe_likelihoods/modules/')
import misc, sne_cmb_fisher_tools

if os.path.exists('/data/spt/')>=-1:
    os.environ["OMP_NUM_THREADS"] = "10"

#--------------------------------------------
#specs
paramfile = 'data/params_cobaya.ini'
param_dict = misc.get_param_dict(paramfile)
save_files = 1 ##0 ##1

lmin, lmax = param_dict['min_l_limit'], param_dict['max_l_limit']
lmax = 5000

#lmin
lmin_t = param_dict['lmin_t']
lmin_p = param_dict['lmin_p']
lmin_phi = param_dict['lmin_phi']
lmin_dic = {'TT': lmin_t, 'EE': lmin_p, 'TE': min(lmin_t, lmin_p), 'PP': lmin_phi}

#lmax
lmax_t = param_dict['lmax_t']
lmax_p = param_dict['lmax_p']
lmax_phi = param_dict['lmax_phi']
lmax_dic = {'TT': lmax_t, 'EE': lmax_p, 'TE': lmax_p, 'PP': lmax_phi}

param_dict['lmax'] = param_dict['max_l_limit'] = lmax
els = np.arange( lmin, lmax+1 )
delta_l = 100 #100
add_lensing = 0 ##1 ###0 ##1 ##0 ##1 ##0 ##1 ###0 ##1 ##0
cmb_experiment_arr = ['so_baseline', 'so_goal', 'spt3g_winter', 'spt3g_summer', 'spt3g_wide', 's4_wide', 'advanced_so_baseline', 'advanced_so_goal']#, 'spt3g']#, 's4_wide']
##cmb_experiment_arr = ['s4_wide']
cmb_experiment_arr = ['advanced_so_baseline', 'advanced_so_goal']
cmb_experiment_arr = ['s4_wide_cobaya_tester']
##cmb_experiment_arr = ['spt3g_winter']
##cmb_experiment_arr = ['spt3g+wide_plus_spt3gwide']
##cmb_experiment_arr = ['spt3g_winter2y', 'spt3g_summer2y']

'''
#different delta_el
delta_l = 200 #50 #100
add_lensing = 0
cmb_experiment_arr = ['s4_wide']
'''

"""
if (1):
    cmb_experiment_arr = ['spt3g_winter']
    delta_l = 1
"""


which_spectra = 'lensed_scalar'
return_dl = False
required_spectra = ['TT', 'EE', 'TE']
lmin_lmax_str = 'lmint%s_lmaxt%s_lminp%s_lmaxp%s' %(lmin_t, lmax_t, lmin_p, lmax_p)
if add_lensing:
    required_spectra.append('PP')
    lmin_lmax_str = '%s_lminphi%s_lmaxphi%s' %(lmin_lmax_str, lmin_phi, lmax_phi)
required_spectra_str = ''.join(required_spectra)

fsky_dic = {'spt3g_winter': 0.036,
            'spt3g_summer': 2650./41253.,
            'spt3g_winter2y': 0.036,
            'spt3g_summer2y': 2650./41253.,            
            'spt3g_wide': 6000./41253.,
            'spt3g+wide_plus_spt3gwide': 6000./41253.,
            'so_baseline': 0.4, 
            'so_goal': 0.4, 
            'advanced_so_baseline': 0.4, 
            'advanced_so_goal': 0.4, 
            's4_wide': 0.57, 's4_wide_cobaya_tester': 0.57, }

#parent_fd = 'cobaya_likelihoods_and_sampling'
#parent_fd = '/Users/sraghunathan/Research/SPTpol/analysis/git/CMB_cobaya_likelihoods_and_sampling'
parent_fd = '/Users/sraghunathan/Research/SPTpol/analysis/git/CMB_BAO_SNe_likelihoods'
if delta_l == 1:
    parent_data_fd = '%s/data/cmb_data/unbinned_%s/' %(parent_fd, lmin_lmax_str)
else:
    #parent_data_fd = '%s/data/binned_with_delta_l_%s/' %(parent_fd, delta_l)
    parent_data_fd = '%s/data/cmb_data/binned_%s_deltal%s/' %(parent_fd, lmin_lmax_str, delta_l)
##parent_likelihood_fd = '%s/cmb_likelihoods/' %(parent_fd)

get_cov_from_sims = False ##True ##False #True
if get_cov_from_sims:
    total_sims_for_cov = 100
    total_sky_area = 41253. #sq. deg.
    fsky_in_sq_deg_dic = {}
    mapparams_dic = {}
    dx = 2. #arcmin pixels
    for cmb_experiment in fsky_dic:
        fsky_in_sq_deg_dic[cmb_experiment] = fsky_dic[cmb_experiment] * total_sky_area
    
        boxsize = int( np.sqrt(fsky_in_sq_deg_dic[cmb_experiment]) ) + 1
        boxsize_in_am = boxsize * 60.
        ny = nx = int( boxsize_in_am / dx )
        mapparams_dic[cmb_experiment] = [ny, nx, dx, dx]

#--------------------------------------------
#get CMB spectra
if get_cov_from_sims:
    pars, fid_cl_dic_unbinned = sne_cmb_fisher_tools.get_camb_cl(param_dict, which_spectra, raw_cl = True, required_spectra = required_spectra, return_dl = return_dl, lmin_dic = lmin_dic, lmax_dic = lmax_dic)
pars, fid_cl_dic = sne_cmb_fisher_tools.get_camb_cl(param_dict, which_spectra, raw_cl = True, delta_l = delta_l, required_spectra = required_spectra, return_dl = return_dl, lmin_dic = lmin_dic, lmax_dic = lmax_dic)
binned_el = fid_cl_dic['els']
#--------------------------------------------

print('\nloop through different experiments now\n')
for cmb_experiment in cmb_experiment_arr:
    print( cmb_experiment )
    data_fd = '%s/%s/' %(parent_data_fd, cmb_experiment)
    if not os.path.exists( data_fd ): os.system('mkdir -p %s' %(data_fd))
    

    exp_details_dic = get_exp_details_using_ilc(els, cmb_experiment)
    #print(exp_details_dic['nu_arr'])

    #get the covariance based on ILC residuals
    curr_ilc_dic = exp_details_dic['ilc_dic']
    curr_ilc_dic_binned = {}
    bpwf_dic = {}
    for which_spec in curr_ilc_dic:
        if which_spec == 'el': continue
        if delta_l > 1:
            curr_el, curr_ilc_cl, bpwf = sne_cmb_fisher_tools.perform_binning(els, curr_ilc_dic[which_spec], delta_el = delta_l, return_dl = return_dl, lmin = lmin_dic[which_spec], lmax = lmax_dic[which_spec])
        else:
            curr_el, curr_ilc_cl, bpwf = els, curr_ilc_dic[which_spec], None
        ###print( which_spec, curr_el, curr_ilc_cl ); sys.exit()
        curr_ilc_dic_binned[which_spec] = curr_ilc_cl
        bpwf_dic[which_spec] = bpwf


    if add_lensing: #get the lensing covariance
        lensing_nl_mv = exp_details_dic['lensing_nl_mv']
        if delta_l > 1:
            curr_el, lensing_nl_mv, bpwf = sne_cmb_fisher_tools.perform_binning(els, lensing_nl_mv, delta_el = delta_l, return_dl = return_dl, lmin = lmin_dic['PP'], lmax = lmax_dic['PP'])
        bpwf_dic['PP'] = bpwf
    ###sys.exit()

    """
    binned_cl_err = sne_cmb_fisher_tools.get_knox_errors_parent(binned_el, fid_cl_dic, curr_ilc_dic_binned, fsky_dic[cmb_experiment])

    if add_lensing: #get lensing bandpower errrors.
        pass
    """

    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    #save binned theory, covariance, bpwf, ilc weights

    #--------------------
    #ilc weights
    ilc_weights_opfname = '%s/%s_ilc_weights.npy' %(data_fd, cmb_experiment)
    if save_files:
        np.save( ilc_weights_opfname, exp_details_dic['weights_dic'])
    #--------------------

    #--------------------
    #bandpowers
    bp_opfname = '%s/%s_bandpowers_%s.txt' %(data_fd, cmb_experiment, required_spectra_str)
    print(bp_opfname)
    op_arr = binned_el
    header = 'ell'
    if 'TT' in required_spectra:
        op_arr = np.column_stack( (op_arr, fid_cl_dic['TT']))
        header = '%s TT' %(header)
        np.savetxt( bp_opfname, op_arr, header = header, fmt = '%g %g')
    if 'EE' in required_spectra:
        op_arr = np.column_stack( (op_arr, fid_cl_dic['EE']))
        header = '%s EE' %(header)
        np.savetxt( bp_opfname, op_arr, header = header, fmt = '%g %g %g')
    if 'TE' in required_spectra:
        op_arr = np.column_stack( (op_arr, fid_cl_dic['TE']))
        header = '%s TE' %(header)
        np.savetxt( bp_opfname, op_arr, header = header, fmt = '%g %g %g %g')
    if 'PP' in required_spectra:
        op_arr = np.column_stack( (op_arr, fid_cl_dic['PP']))
        header = '%s PP' %(header)
        np.savetxt( bp_opfname, op_arr, header = header, fmt = '%g %g %g %g %g')
    ###sys.exit()
    #--------------------
    #bandpower window function
    bpwf_opfname = '%s/%s_bpwf_%s.npy' %(data_fd, cmb_experiment, required_spectra_str)
    if save_files:
        np.save( bpwf_opfname, bpwf_dic)

    #--------------------
    #covariance
    ##print( fid_cl_dic['TT'].shape )
    ##print( curr_ilc_dic_binned['TT'].shape )
    cl_tt_final = fid_cl_dic['TT'] + curr_ilc_dic_binned['TT']
    if 'EE' in required_spectra:
        cl_ee_final = fid_cl_dic['EE'] + curr_ilc_dic_binned['EE']
    if 'TE' in required_spectra:
        cl_te_final = fid_cl_dic['TE'] + curr_ilc_dic_binned['TE']
    if 'PP' in required_spectra:
        cl_pp_final = fid_cl_dic['PP'] + lensing_nl_mv.real
    else:
        cl_pp_final = np.zeros( len(cl_tt_final) )
    spec_arr = required_spectra
    total_spec = len(spec_arr)
    total_ell_bins = len(binned_el)
    if (1):##delta_l>1:
        cov_mat = np.zeros( (total_ell_bins*total_spec, total_ell_bins*total_spec))
        for b, el_b in enumerate( binned_el ):
            #if b>2: continue
            if len(required_spectra) == 1:
                curr_el_cov = get_covariance_cmb_spectra_single(el_b, cl_tt_final[b], fsky = fsky_dic[cmb_experiment], delta_el = delta_l)
            else:
                curr_el_cov = get_covariance_cmb_spectra(el_b, cl_tt_final[b], cl_ee_final[b], cl_te_final[b], PP = cl_pp_final[b], fsky = fsky_dic[cmb_experiment], delta_el = delta_l, add_lensing = add_lensing)
            for s1 in range(total_spec):
                for s2 in range(total_spec):
                    i, j = (s1*total_ell_bins)+b, (s2*total_ell_bins)+b
                    cov_mat[i, j] = curr_el_cov[s1, s2]
                    #print(b, spec_arr[s1], spec_arr[s1], i, j)
        cov_mat_inv = np.linalg.inv( cov_mat )

        cov_opfname = '%s/%s_covariance_%s.txt' %(data_fd, cmb_experiment, required_spectra_str)
        cov_inv_opfname = '%s/%s_covariance_inv_%s.txt' %(data_fd, cmb_experiment, required_spectra_str)
        if save_files:
            np.savetxt( cov_opfname, cov_mat )
            np.savetxt( cov_inv_opfname, cov_mat_inv )

    else:
        pass #unbinned

    if (0): #plot
        from pylab import *
        binned_cl_err = np.sqrt( np.diag( cov_mat ) )
        binned_cl_err_tt = binned_cl_err[:total_ell_bins]
        binned_cl_err_ee = binned_cl_err[total_ell_bins: 2 * total_ell_bins]
        binned_cl_err_te = binned_cl_err[2* total_ell_bins: 3 * total_ell_bins]
        binned_cl_err_pp = binned_cl_err[3* total_ell_bins: ]

        clf()
        ax = subplot(111, yscale = 'log')
        binned_dl_fac = binned_el * (binned_el+1)/2/np.pi
        errorbar( binned_el-20, binned_dl_fac * fid_cl_dic['TT'], yerr = binned_dl_fac * binned_cl_err_tt, capsize = 1., color = 'black', marker = '.', ls = 'None' )
        errorbar( binned_el-20, binned_dl_fac * fid_cl_dic['EE'], yerr = binned_dl_fac * binned_cl_err_ee, capsize = 1., color = 'orangered', marker = '.', ls = 'None' )
        errorbar( binned_el-20, binned_dl_fac * abs(fid_cl_dic['TE']), yerr = binned_dl_fac * abs(binned_cl_err_te), capsize = 1., color = 'darkgreen', marker = '.', ls = 'None' )
        xlim(10., lmax+10); ylim(0.1, 5e3)
        show()


        clf()
        binned_dl_fac = (binned_el * (binned_el+1))**2./2/np.pi
        cl_kk_dic = {'PP': fid_cl_dic['PP']}
        nl_kk_dic = {'PP': real(lensing_nl_mv)}
        binned_cl_err_knox = sne_cmb_fisher_tools.get_knox_errors_parent(binned_el, cl_kk_dic, nl_kk_dic, fsky_dic[cmb_experiment], delta_el = delta_l)

        cl_kk_dic = {'PP': real(exp_details_dic['lensing_cl_kk'])}
        nl_kk_dic = {'PP': real(exp_details_dic['lensing_nl_mv'])}
        unbinned_cl_err_knox = sne_cmb_fisher_tools.get_knox_errors_parent(els, cl_kk_dic, nl_kk_dic, fsky_dic[cmb_experiment], delta_el = 1)

        tmpeldeltael10, tmpcl, tmpbpwf = sne_cmb_fisher_tools.perform_binning(els, real(exp_details_dic['lensing_cl_kk']) + real(exp_details_dic['lensing_nl_mv']), delta_el = 10, return_dl = return_dl, lmin = 1)
        cl_kk_dic = {'PP': real(tmpcl)}
        nl_kk_dic = {'PP': real(tmpcl)*0.}
        cl_err_knox_deltael10 = sne_cmb_fisher_tools.get_knox_errors_parent(tmpeldeltael10, cl_kk_dic, nl_kk_dic, fsky_dic[cmb_experiment], delta_el = 10)

        tmpeldeltael50, tmpcl, tmpbpwf = sne_cmb_fisher_tools.perform_binning(els, real(exp_details_dic['lensing_cl_kk']) + real(exp_details_dic['lensing_nl_mv']), delta_el = 50, return_dl = return_dl, lmin = 1)
        cl_kk_dic = {'PP': real(tmpcl)}
        nl_kk_dic = {'PP': real(tmpcl)*0.}
        cl_err_knox_deltael50 = sne_cmb_fisher_tools.get_knox_errors_parent(tmpeldeltael50, cl_kk_dic, nl_kk_dic, fsky_dic[cmb_experiment], delta_el = 50)

        cl_kk_dic = {'PP': real(cl_pp_final)}
        nl_kk_dic = {'PP': cl_pp_final*0.}
        binned_cl_err_knox_v2 = sne_cmb_fisher_tools.get_knox_errors_parent(binned_el, cl_kk_dic, nl_kk_dic, fsky_dic[cmb_experiment], delta_el = delta_l)

        dl_fac = (els * (els+1))**2./2/np.pi
        dl_fac_deltael10 = (tmpeldeltael10 * (tmpeldeltael10+1))**2./2/np.pi
        dl_fac_deltael50 = (tmpeldeltael50 * (tmpeldeltael50+1))**2./2/np.pi

        ax = subplot(111, yscale = 'log')#, xscale='log')
        #binned_dl_fac = 1.
        #plot( binned_el, binned_dl_fac * fid_cl_dic['PP'], color = 'black', ls = 'None')
        errorbar( binned_el, binned_dl_fac * fid_cl_dic['PP'], yerr = binned_dl_fac * binned_cl_err_pp, capsize = 1., color = 'black', marker = '.', ls = 'None' )
        #errorbar( binned_el-20, binned_dl_fac * fid_cl_dic['PP'], yerr = binned_dl_fac * binned_cl_err_knox['PP'], capsize = 1., color = 'goldenrod', marker = '.', ls = 'None' )
        errorbar( binned_el+20, binned_dl_fac * fid_cl_dic['PP'], yerr = binned_dl_fac * binned_cl_err_knox_v2['PP'], capsize = 1., color = 'tab:green', marker = '.', ls = 'None' )
        plot( els, dl_fac * exp_details_dic['lensing_nl_mv'], color = 'orangered', label = r'N0' )
        plot( els, dl_fac * unbinned_cl_err_knox['PP'], color = 'navy', label = r'Bandpower error: Unbinned' )
        plot( tmpeldeltael10, dl_fac_deltael10 * cl_err_knox_deltael10['PP'], color = 'green', label = r'Bandpower error: deltael = 10' )
        plot( tmpeldeltael50, dl_fac_deltael50 * cl_err_knox_deltael50['PP'], color = 'goldenrod', label = r'Bandpower error: deltael = 50' )
        plot( binned_el, binned_dl_fac * binned_cl_err_pp, color = 'darkred', label = r'Bandpower error: deltael = 100')
        legend(loc = 1)
        xlim(10., lmax+10); ylim(1e-10, 5e-7)#5e3)
        show(); sys.exit()


    if (0): #plot
        to_plot = np.log(cov_mat)
        to_plot[np.isinf(to_plot)] = None
        to_plot[np.isnan(to_plot)] = None
        imshow(to_plot); colorbar(); show()
    #--------------------

    #--------------------
    #add proposal covariance matrix
    if (0):
        import misc
        if cmb_experiment == 'so_baseline':
            fisher_fname = 'results/cmb/sobaseline/sobaseline_cmb_fisher_lensed_scalar_params--As--h--ns--omch2--ombh2--tau--ws--wa--mnu--neff--nrun_lmint300--lminp300--lmaxt3500--lmaxp4000_fsky0.4.npy'
        elif cmb_experiment == 'so_goal':
            fisher_fname = 'results/cmb/sogoal/sogoal_cmb_fisher_lensed_scalar_params--As--h--ns--omch2--ombh2--tau--ws--wa--mnu--neff--nrun_lmint300--lminp300--lmaxt3500--lmaxp4000_fsky0.4.npy'
        elif cmb_experiment in ['spt3g_winter', 'spt3g_summer', 'spt3g_wide', 'spt3g']:
            #same as SO-Baseline for now
            fisher_fname = 'results/cmb/sobaseline/sobaseline_cmb_fisher_lensed_scalar_params--As--h--ns--omch2--ombh2--tau--ws--wa--mnu--neff--nrun_lmint300--lminp300--lmaxt3500--lmaxp4000_fsky0.4.npy'
        elif cmb_experiment == 's4_wide':
            fisher_fname = 'results/cmb/cmbs4/cmbs4_cmb_fisher_lensed_scalar_params--As--h--ns--omch2--ombh2--tau--ws--wa--mnu--neff--nrun_lmint300--lminp300--lmaxt3500--lmaxp4000_fsky0.57.npy'


        fisher_dic = np.load(fisher_fname, allow_pickle = True).item()
        fisher_mat, params = fisher_dic['fisher_matrix'], fisher_dic['params']
        #fix_params_arr = ['ws', 'wa', 'neff', 'nrun', 'mnu']
        fix_params_arr = []#'ws', 'wa', 'neff', 'nrun', 'mnu']

        #get H0 from h.
        ##from IPython import embed; embed(); sys.exit()
        """
        param_dict['H0'] = param_dict['h'] * 100.
        param_dict['logA'] = np.log( param_dict['As'] * 1e10 )
        mod_param_arr = [['h', 'H0'], ['As', 'logA']]
        fisher_dic = {cmb_experiment: fisher_mat}
        if (1): 
            cov_mat_proposal = np.linalg.inv( fisher_mat )

            #J_mat, param_names_mod = misc.get_jacobian_transformation(fisher_mat, params, param_dict, mod_param_arr)

        fisher_dic, params = misc.rotate_fisher_mat_parent(fisher_dic, params, param_dict, mod_param_arr, fix_params_arr = fix_params_arr)
        fisher_mat = fisher_dic[cmb_experiment]
        """

        cov_mat_proposal = np.linalg.inv( fisher_mat )
        marg_params_str = ' '.join( params )
        proposal_cov_opfname = '%s/%s_proposal_covariance_lcdm.txt' %(data_fd, cmb_experiment)
        if save_files: 
            np.savetxt(proposal_cov_opfname, cov_mat_proposal, header = marg_params_str)
    #--------------------

    #--------------------
    """
    #create yaml files for the likelihoods
    def replace_str(opline, searchstr, replaceval):
        if opline.find(searchstr)>-1:
            opline = opline.replace(searchstr, replaceval)
        return opline

    analytic_or_simbased_cov_val = 'analytic'
    cl_or_dl_val = 'cl'
    use_cosmopower_val = False
    
    template_yaml_fname = 'data/templateforlikelihoods.yaml'
    template_yaml = open(template_yaml_fname, 'r')
    template_dic_for_cal = {'mapTPvalCal':
                                {'prior': {'min': 0.5, 'max': 1.5},
                                'ref': 1.,
                                'proposal': 0.001, 
                                'latex': "TPval_{\mathrm cal^\mathrm{bandval}}}}"
                                }
                            }

    yaml_opfname = '%s/%s_%s.yaml' %(parent_likelihood_fd, cmb_experiment, required_spectra_str)
    yaml_opf = open(yaml_opfname, 'w')

    for opline in template_yaml:
        opline = replace_str(opline, 'cmb_experiment_name_val', cmb_experiment)
        opline = replace_str(opline, 'analytic_or_simbased_cov_val', analytic_or_simbased_cov_val)
        opline = replace_str(opline, 'spectra_to_use_val', str(required_spectra))
        opline = replace_str(opline, 'freq_list_val', str(exp_details_dic['nu_arr']))
        opline = replace_str(opline, 'cl_or_dl_val', cl_or_dl_val)
        opline = replace_str(opline, 'use_cosmopower_val', str(use_cosmopower_val))
        opline = replace_str(opline, 'delta_l_val', str(delta_l))
        opline = replace_str(opline, 'lmin_t_val', str(lmin_dic['TT']))
        opline = replace_str(opline, 'lmin_p_val', str(lmin_dic['EE']))

        #lensing related
        if add_lensing:
            opline = replace_str(opline, 'lmin_pp_val', str(lmin_dic['PP']))
        else:
            continue

        print(opline)
        yaml_opf.writelines( '%s' %(opline) )


    #cal related
    opline = 'params:\n'; yaml_opf.writelines( '%s' %(opline) )
    for nucntr, curr_nu in enumerate( exp_details_dic['nu_arr'] ):
        for TP in ['T', 'P']:
            for k1 in template_dic_for_cal:                
                opline = '\t%s%s:\n' %(k1.replace('TP', TP), nucntr)
                for k2 in template_dic_for_cal[k1]:
                    opline = '\t\t%s:\n' %(k2)
                    for k3 in template_dic_for_cal[k1][k2]:
                        opline = '\t\t\t%s' %(k2)
                        print(opline)
                        sys.exit()


    yaml_opf.writelines( '%s\n' %(opline) )
    #yaml_opf.close()

    print(opline)
    sys.exit()
    """

    #--------------------

    """
    if get_cov_from_sims:
        ##sys.exit()
        import sim_tools_flatsky
        mapparams = mapparams_dic[cmb_experiment]
        ny, nx, dx, dx = mapparams
        sim_shape = [ny, nx]
        pixel_res_radians = np.radians(dx/60.)
        cl_dict = {}
        fid_els = np.arange(len(fid_cl_dic_unbinned['TT'] ))
        cl_dict[(0,0)] = np.interp(els, fid_els, fid_cl_dic_unbinned['TT']) + curr_ilc_dic['TT']
        if 'EE' in required_spectra:
            cl_dict[(1,1)] = np.interp(els, fid_els, fid_cl_dic_unbinned['EE']) + curr_ilc_dic['EE']
            cl_dict[(1,0)] = np.interp(els, fid_els, fid_cl_dic_unbinned['TE']) + curr_ilc_dic['TE']

        tmpbinsize = 50
        sim_cl_arr = []
        for sim_index in range( total_sims_for_cov ):
            print(sim_index)
            curr_sim = sim_tools_flatsky.make_gaussian_realisations(els, cl_dict, (sim_shape), pixel_res_radians)
            curr_map1 = curr_sim[0]
            if 'EE' in required_spectra:
                curr_map2 = curr_sim[1]

            curr_el, curr_cl_11 = sim_tools_flatsky.map2cl(sim_shape, pixel_res_radians, curr_map1, flatskymap2 = curr_map1, minbin = 0, maxbin = lmax, binsize = tmpbinsize)
            if 'EE' in required_spectra:
                curr_el, curr_cl_22 = sim_tools_flatsky.map2cl(sim_shape, pixel_res_radians, curr_map2, flatskymap2 = curr_map2, minbin = 0, maxbin = lmax, binsize = tmpbinsize)
                curr_el, curr_cl_12 = sim_tools_flatsky.map2cl(sim_shape, pixel_res_radians, curr_map1, flatskymap2 = curr_map2, minbin = 0, maxbin = lmax, binsize = tmpbinsize)
                
                curr_cl_arr = [curr_cl_11, curr_cl_22, curr_cl_12]
            else:
                curr_cl_arr = [curr_cl_11]

            curr_cl_arr_binned = []
            for curr_cl in curr_cl_arr:
                curr_cl = np.interp(els, curr_el, curr_cl)
                tmpel, tmpcl, tmp_bpwf = sne_cmb_fisher_tools.perform_binning(els, curr_cl, delta_el = delta_l, return_dl = return_dl, lmin_cut = lmin_dic['TT'])
                curr_cl_arr_binned.extend( tmpcl )

            curr_cl_arr_binned = np.asarray( curr_cl_arr_binned )

            sim_cl_arr.append( curr_cl_arr_binned.ravel() )

        sim_cl_arr = np.asarray( sim_cl_arr )
        sim_cov = np.cov( sim_cl_arr.T )

        sim_cov_inv = np.linalg.inv( sim_cov )

        cov_opfname = '%s/%s_simbasedcovariance_%s.txt' %(data_fd, cmb_experiment, required_spectra_str)
        cov_inv_opfname = '%s/%s_simbasedcovariance_inv_%s.txt' %(data_fd, cmb_experiment, required_spectra_str)
        np.savetxt( cov_opfname, sim_cov )
        np.savetxt( cov_inv_opfname, sim_cov_inv )
        #sys.exit()

        if (1):
            clf()
            ax = subplot(111, yscale = 'log')
            binned_dl_fac = binned_el * (binned_el+1)/2/np.pi
            errorbar( binned_el, binned_dl_fac * fid_cl_dic['TT'], color = 'black' )
            for sim_cl in sim_cl_arr:
                errorbar( binned_el, binned_dl_fac * sim_cl[:total_ell_bins], color = 'gray' )

            errorbar( binned_el, binned_dl_fac * curr_cl_arr_binned[:total_ell_bins], color = 'goldenrod' )
            tmp_dl_fac = curr_el * (curr_el+1)/2/np.pi
            errorbar( curr_el, tmp_dl_fac * curr_cl_11, color = 'orangered' )

            xlim(0., lmax+10); ylim(0.1, 1e4)
            show()

        if (1): #plotting

            binned_cl_err = sne_cmb_fisher_tools.get_knox_errors_parent(binned_el, fid_cl_dic, curr_ilc_dic_binned, fsky_dic[cmb_experiment], delta_el = delta_l)

            tmp_cl_mean = np.mean(sim_cl_arr, axis = 0)
            tmp_cl_dic = {'TT': tmp_cl_mean}
            tmp_nl_dic = {'TT': tmp_cl_mean*0.} 
            binned_cl_err_v2 = sne_cmb_fisher_tools.get_knox_errors_parent(binned_el, tmp_cl_dic, tmp_nl_dic, fsky_dic[cmb_experiment])
            
            binned_cl_err_from_cov_mat = np.sqrt( np.diag( cov_mat ) )
            binned_cl_err_from_sim_cov = np.sqrt( np.diag( sim_cov ) )
            binned_cl_err_from_cov_mat_tt = binned_cl_err_from_cov_mat[:total_ell_bins]
            binned_cl_err_from_sim_cov_tt = binned_cl_err_from_sim_cov[:total_ell_bins]

            clf()
            ax = subplot(111, yscale = 'log')
            binned_dl_fac = binned_el * (binned_el+1)/2/np.pi
            errorbar( binned_el-20, binned_dl_fac * fid_cl_dic['TT'], yerr = binned_dl_fac * binned_cl_err['TT'], capsize = 1., color = 'black', marker = '.', ls = 'None' )
            errorbar( binned_el-10, binned_dl_fac * fid_cl_dic['TT'], yerr = binned_dl_fac * binned_cl_err_v2['TT'], capsize = 1., color = 'purple', marker = '.', ls = 'None' )
            errorbar( binned_el, binned_dl_fac * fid_cl_dic['TT'], yerr = binned_dl_fac * binned_cl_err_from_cov_mat_tt, capsize = 1., color = 'orangered', marker = '.', ls = 'None' )
            errorbar( binned_el+20, binned_dl_fac * fid_cl_dic['TT'], yerr = binned_dl_fac * binned_cl_err_from_sim_cov_tt, capsize = 1., color = 'goldenrod', marker = '.', ls = 'None' )
            xlim(10., lmax+10); ylim(0.1, 5e3)
            show()

            clf()
            plot(binned_el, binned_cl_err['TT']/binned_cl_err_v2['TT'], color = 'purple')
            plot(binned_el, binned_cl_err['TT']/binned_cl_err_from_cov_mat_tt, color = 'orangered')
            plot(binned_el, binned_cl_err['TT']/binned_cl_err_from_sim_cov_tt, color = 'goldenrod')
            xlim(10., lmax+10); ylim(0.5, 1.5)
            axhline(1., lw = 0.2, alpha = 0.2)
            show()


            subplot(121); imshow( cov_mat ); colorbar(); 
            subplot(122); imshow( sim_cov ); colorbar(); 
            show()
        """


    
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
    #--------------------------------------------------------------------------------
print('All done.')
sys.exit()


