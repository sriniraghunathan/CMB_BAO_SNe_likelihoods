from pylab import *
import numpy as np, sys, os, copy, scipy as sc, pandas as pd
from astropy import constants as const
from astropy import units as u
from astropy import coordinates as coord
import copy
import astropy


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

def get_knox_errors_parent(els, cl_dic, nl11_dic, fsky, nl22_dic = None, nl12_dic = None, delta_el = None):

    delta_cl_dic = {}
    for XX in cl_dic:
        if XX not in ['TT', 'EE', 'TE', 'BB', 'PP']: continue
        print(XX)
        if XX == 'TT':
            nl11 = nl11_dic['TT']
            if nl22_dic is not None:
                nl22 = nl22_dic['TT']
                nl12 = nl12_dic['TT']
        elif XX == 'EE' or XX == 'BB':
            nl11 = nl11_dic['EE']
            if nl22_dic is not None:
                nl22 = nl22_dic['EE']
                nl12 = nl12_dic['EE']
        elif XX == 'TE':
            nl11 = np.copy(nl11_dic['TT']) * 0.
            if nl22_dic is not None:
                nl22 = np.copy(nl11) * 0.
                nl12 = np.copy(nl11) * 0.
        elif XX == 'PP':
            nl11 = np.copy(nl11_dic['PP'])
            if nl22_dic is not None:
                nl22 = np.copy(nl11) * 0.
                nl12 = np.copy(nl11) * 0.

        cl11 = cl_dic[XX] + nl11
        if nl22_dic is not None:
            cl22 = cl_dic[XX] + nl22
            cl12 = cl_dic[XX] + nl12
        else:
            cl22, cl12 = None, None

        delta_cl_dic[XX] = get_knox_errors(els, cl11, fsky, cl22 = cl22, cl12 = cl12, delta_el = delta_el)

    return delta_cl_dic

def get_knox_errors(els, cl11, fsky, cl22 = None, cl12 = None, delta_el = None):

    """
    get Knox bandpower errors.

    els: multipoles
    cl11: signal + noise in the first map.
    cl22: signal + noise in the second map.
    cl12: signal + noise cross spectrum of the two maps.
    """

    if delta_el is None:
        delta_el = np.diff(els)[0]
        ##print(np.diff(els)); sys.exit()

    if cl22 is not None and cl12 is not None:
        cl_total = np.sqrt( (cl12**2. + cl11 * cl22)/2. )
    else:
        cl_total = cl11

    cl_knox_err = np.sqrt(2./ (2.*els + 1.) / fsky / delta_el ) * (cl_total)
    cl_knox_err[np.isnan(cl_knox_err)] = 0.
    cl_knox_err[np.isinf(cl_knox_err)] = 0.

    return cl_knox_err

def set_cosmo(param_dict, baselinecosmo = 'Flatw0waCDM', use_hsq_units = True):
    """
    Set input cosmology.

    Parameters:
    param_dict: dictionary with parameter values
    baselinecosmo: Baseline cosmology to use.
        Default is Flatw0waCDM

    Returns:
    cosmo: cosmology that was set.
    """
    if 'H0' in param_dict:
        param_dict['h'] = param_dict['H0'] / 100.
    if baselinecosmo == 'FlatLambdaCDM':
        if use_hsq_units:
            cosmo = astropy.cosmology.FlatLambdaCDM(H0 = param_dict['h']*100., 
                                 Om0 = param_dict['omch2']/param_dict['h']**2., 
                                 Ob0 = param_dict['ombh2']/param_dict['h']**2., 
                                 Tcmb0 = param_dict['tcmb'], 
                                 m_nu = [0., 0., param_dict['mnu']] * u.eV,
                                 Neff = param_dict['neff'],
                                 )    

        else:
            cosmo = astropy.cosmology.FlatLambdaCDM(H0 = param_dict['h']*100., 
                                 Om0 = param_dict['omega_m'], 
                                 Ob0 = param_dict['ombh2']/param_dict['h']**2., 
                                 Tcmb0 = param_dict['tcmb'], 
                                 m_nu = [0., 0., param_dict['mnu']] * u.eV,
                                 Neff = param_dict['neff'], 
                                 )
    elif baselinecosmo == 'FlatwCDM':
        if use_hsq_units:
            cosmo = astropy.cosmology.Flatw0waCDM(H0 = param_dict['h']*100., 
                                 Om0 = param_dict['omch2']/param_dict['h']**2., 
                                 Ob0 = param_dict['ombh2']/param_dict['h']**2., 
                                 Tcmb0 = param_dict['tcmb'], 
                                 m_nu = [0., 0., param_dict['mnu']] * u.eV,
                                 Neff = param_dict['neff'], 
                                 w0 = param_dict['ws'], 
                                 )
        else:
            cosmo = astropy.cosmology.Flatw0waCDM(H0 = param_dict['h']*100., 
                                 Om0 = param_dict['omega_m'], 
                                 Ob0 = param_dict['ombh2']/param_dict['h']**2., 
                                 Tcmb0 = param_dict['tcmb'], 
                                 m_nu = [0., 0., param_dict['mnu']] * u.eV,
                                 Neff = param_dict['neff'], 
                                 w0 = param_dict['ws'], 
                                 )    
    elif baselinecosmo == 'Flatw0waCDM':
        if use_hsq_units:
            cosmo = astropy.cosmology.Flatw0waCDM(H0 = param_dict['h']*100., 
                                 Om0 = param_dict['omch2']/param_dict['h']**2., 
                                 Ob0 = param_dict['ombh2']/param_dict['h']**2., 
                                 Tcmb0 = param_dict['tcmb'], 
                                 m_nu = [0., 0., param_dict['mnu']] * u.eV,
                                 Neff = param_dict['neff'], 
                                 w0 = param_dict['ws'], 
                                 wa = param_dict['wa'], 
                                 )
        else:
            cosmo = astropy.cosmology.Flatw0waCDM(H0 = param_dict['h']*100., 
                                 Om0 = param_dict['omega_m'], 
                                 Ob0 = param_dict['ombh2']/param_dict['h']**2., 
                                 Tcmb0 = param_dict['tcmb'], 
                                 m_nu = [0., 0., param_dict['mnu']] * u.eV,
                                 Neff = param_dict['neff'], 
                                 w0 = param_dict['ws'], 
                                 wa = param_dict['wa'], 
                                 )
    return cosmo

def perform_redshift_cuts(zmin, zmax, z_arr, sne_cat):

    """
    Perform redshift cuts.

    Parameters
    ----------
    zmin: float
        minimum redshift.
        Default is -1 corresponding to no zmin cut.
    zmax: float
        maximum redshift.
        Default is -1 corresponding to no zmax cut.
    z_arr: array
        Redshift array of the sample with length = N_sources = len(z_arr).
    sne_cat: ndarray
        Array containing the catalogue.
        Dimension must be N_fields x N_sources.

    Returns
    -------
    sne_cat: ndarray
        Refined catalogue.
    passed_inds: array
        Selected indices based on the redshift cuts.
    """

    if zmin != -1 and zmax != -1:
        passed_inds =  np.where( (z_arr>=zmin) & (z_arr<=zmax) )[0]
    elif zmin != -1:
        passed_inds =  np.where( (z_arr>=zmin) )[0]
    elif zmax != -1:
        passed_inds =  np.where( (z_arr<=zmax) )[0]
    else:
        passed_inds = np.arange( len(z_arr) )

    sne_cat = sne_cat[:, passed_inds]

    return sne_cat, passed_inds

def refine_covs_based_on_cuts(cov_dic_or_cov, good_inds):

    """
    Refined the covariance matrix based on the good_inds.

    Parameters
    ----------
    cov_dic_or_cov: dict or array.
        original covariance matrix stored in a dictionary or just a array.

    good_inds: array.
        Indices to be selected in the covariance matrix.

    Returns
    -------
    cov_mod: dict or array.
        Refined covariance dict or array.
    """

    if isinstance(cov_dic_or_cov, dict):
        cov_mod = {}  
        for keyname in cov_dic_or_cov:
            cov_mod[keyname] = cov_dic_or_cov[keyname][good_inds[:, None], good_inds[None,:]]
        return cov_mod
    else:
        cov_mod = cov_dic_or_cov[good_inds[:, None], good_inds[None,:]]
        return cov_mod


def get_sne_details(sne_exp, add_stat_error, perform_checks_with_des = False, perform_random_z_selection = False, z_binning_kind = 'cumulative', unbinned_sne_sim_no = 1, obtain_covs = False, reqd_cov_tags = [0, 1, 2, 3, 4, 5, 6, 7], zmin = -1, zmax = -1):

    assert z_binning_kind in ['cumulative', 'individual']
    assert sne_exp in ['lsst_binned', 'lsst_unbinned', 'lsst_v2_unbinned', 'lsst_v2_binned', 'des', 'des_cobaya', 'test', 'roman']

    print('Reading and process details for sne_exp = %s' %(sne_exp))

    params = np.asarray( ['omch2', 'ws', 'wa', 'M'] )
    if perform_checks_with_des:
        sne_exp = 'des'
        perform_random_z_selection = False #True
        ###params = np.asarray( ['h', 'omch2', 'ws'] )
        params = np.asarray( ['omega_m', 'ws', 'wa'] )
        ###params = np.asarray( ['omega_m', 'ws'] )

        if perform_random_z_selection:
            params = np.asarray( ['omega_m', 'ws'] )

    #SNe details
    if sne_exp == 'lsst_binned':
        #params = np.asarray( ['omch2', 'ws', 'wa', 'M'] )
        sne_fd = 'data/lsst_SNe/'
        sne_details_fname = '%s/SNIa_distances.txt' %(sne_fd)
        #zarr = np.loadtxt(sne_details_fname, skiprows = 5, usecols = [2])
        sne_arr, z_arr, mu_arr, muerr_stat_arr, muerr_sys_arr = np.loadtxt(sne_details_fname, skiprows = 5, usecols = [1, 2, 4, 5, 6], unpack = True)
        stretch_x1_arr, color_c_arr = np.zeros(len(sne_arr)), np.zeros(len(sne_arr))
        
        sne_cov_tag_dic = {0: 'Stat + sys-All', 
                           # 1: 'Stat-only', 
                           # 2: 'Stat + sys-ZSHIFT', 
                           # 3: 'Stat + sys-ZERRSCALE', 
                           # 4: 'Stat + sys-Photo_shift', 
                           # 5: 'Stat + sys-MWEBV', 
                           # 6: 'Stat + sys-Cal_ZP', 
                           # 7: 'Stat + sys-Cal_wave', 
                           # 8: 'Stat + sys-Cal'
                          }        
    
    elif sne_exp in ['lsst_unbinned', 'lsst_v2_unbinned', 'lsst_v2_binned']:
        '''
        sne_fd = 'data/lsst_SNe_unbinned/'
        sne_details_fname = '%s/FITOPT000_MUOPT000.FITRES' %(sne_fd)
        unbinned_rec = pd.read_csv( sne_details_fname, comment='#', delim_whitespace=True)
        
        sne_arr = unbinned_rec['CID']
        z_arr, mu_arr, muerr_stat_arr = unbinned_rec['zHEL'], unbinned_rec['MU'], unbinned_rec['MUERR']
        stretch_x1_arr, color_c_arr = unbinned_rec['x1'], unbinned_rec['c']
        '''
        #params = np.concatenate( (params, ['alpha', 'beta']) )
        if sne_exp == 'lsst_unbinned':
            parent_sne_fd = 'data/lsst_SNe_unbinned/PLASTICC_COMBINED_CWR/'
            sne_fd = '%s/7_CREATE_COV/LSST_UNBINNED_COV_BBC_SIMDATA_PHOTOZ_%d/output/' %(parent_sne_fd, unbinned_sne_sim_no)
            skiprows = 5
        elif sne_exp == 'lsst_v2_unbinned':
            parent_sne_fd = 'data/lsst_v2_SNe_unbinned/PLASTICC_COMBINED_CWR/'
            sne_fd = '%s/7_CREATE_COV/LSST_UNBINNED_COV/output/LSST_UNBINNED_COV_BBC_SIMDATA_PHOTOZ_OUTPUT_BBCFIT-%04d/' %(parent_sne_fd, unbinned_sne_sim_no)
            skiprows = 10
        elif sne_exp == 'lsst_v2_binned':
            parent_sne_fd = 'data/lsst_v2_SNe_unbinned/PLASTICC_COMBINED_CWR/'
            sne_fd = '%s/7_CREATE_COV/LSST_BINNED_COV/output/LSST_BINNED_COV_BBC_SIMDATA_PHOTOZ_OUTPUT_BBCFIT-%04d/' %(parent_sne_fd, unbinned_sne_sim_no)
            skiprows = 9

        def get_sne_color_stretch(sne_arr, sim_no = 1):

            #unbinned details
            fitsrec_sne_details_fname = '%s/6_BIASCOR/BBC_SIMDATA_PHOTOZ/output/OUTPUT_BBCFIT-%04d/FITOPT000_MUOPT000.FITRES' %(parent_sne_fd, sim_no)
            #print( fitsrec_sne_details_fname, os.path.exists( fitsrec_sne_details_fname) )
            unbinned_rec = pd.read_csv( fitsrec_sne_details_fname, comment='#', delim_whitespace=True)
            fitsrec_sne_arr = list( unbinned_rec['CID'] )
            fitsrec_z_arr, fitsrec_mu_arr, fitsrec_muerr_stat_arr = unbinned_rec['zHEL'], unbinned_rec['MU'], unbinned_rec['MUERR']
            fitsrec_stretch_x1_arr, fitsrec_color_c_arr = unbinned_rec['x1'], unbinned_rec['c']

            #get the association index for sne_cid_arr in fitsrec_sne_cid_arr
            association_inds = [fitsrec_sne_arr.index(cid) for cid in sne_arr]
            return np.asarray( fitsrec_stretch_x1_arr[association_inds] ), np.asarray( fitsrec_color_c_arr[association_inds] )

        sne_details_fname = '%s/hubble_diagram.txt' %(sne_fd)
        if sne_exp in ['lsst_unbinned', 'lsst_v2_unbinned']: #hubble diagram
            sne_arr, z_arr, mu_arr, muerr_stat_arr, muerr_vpec_arr, muerr_sys_arr = np.loadtxt(sne_details_fname, skiprows = skiprows, usecols = [1, 3, 5, 6, 7, 8], unpack = True)
            #stretch_x1_arr, color_c_arr = get_sne_color_stretch(sne_arr, sim_no = unbinned_sne_sim_no)
        elif sne_exp in ['lsst_v2_binned']:
            sne_arr, z_arr, mu_arr, muerr_stat_arr, muerr_sys_arr = np.loadtxt(sne_details_fname, skiprows = skiprows, usecols = [1, 2, 4, 5, 6], unpack = True)
        stretch_x1_arr, color_c_arr = np.zeros(len(sne_arr)), np.zeros(len(sne_arr))

        '''
        plot( z_arr, mu_arr, 'k,', ls = 'None'); 
        xlabel(r'Redshift'); ylabel(r'Distance modulus $\mu$'); show(); sys.exit()
        '''
        
        if unbinned_sne_sim_no==1:
            sne_cov_tag_dic = {0: 'Stat + sys-All',
                           # 1: 'Stat-only', 
                           # 2: 'Stat + sys-ZSHIFT', 
                           # 3: 'Stat + sys-ZERRSCALE', 
                           # 4: 'Stat + sys-Photo_shift', 
                           # 5: 'Stat + sys-MWEBV', 
                           # 6: 'Stat + sys-Cal_ZP', 
                           # 7: 'Stat + sys-Cal_wave', 
                           # 8: 'Stat + sys-Cal'
                          } 
        else:       
            sne_cov_tag_dic = {0: 'Stat + sys-All'}

    elif sne_exp in ['des', 'des_cobaya']:
        if sne_exp == 'des':
            sne_fd = 'data/DES_SNIa/'
            sne_details_fname = '%s/hubble_diagram.txt' %(sne_fd)
            sne_arr, z_arr, mu_arr, muerr_stat_arr, muerr_vpec_arr, muerr_sys_arr = np.loadtxt(sne_details_fname, skiprows = 9, usecols = [1, 4, 5, 6, 7, 8], unpack = True)
        elif sne_exp == 'des_cobaya':
            sne_fd = 'data/DES_SNIa_cobaya/'
            sne_data_fname = '%s/DES-SN5YR_HD.csv' %(sne_fd)
            sne_rec = pd.read_csv( sne_data_fname )
            
            sne_arr = sne_rec['CID']
            z_arr = sne_rec['zCMB']
            mu_arr = sne_rec['MU']
            muerr_stat_arr = sne_rec['MUERR_FINAL']
            muerr_vpec_arr = np.zeros( len( muerr_stat_arr) )
            muerr_sys_arr = np.zeros( len( muerr_stat_arr) )

        stretch_x1_arr, color_c_arr = None, None
        sne_cov_tag_dic = {0: 'Stat + sys-All', 
                          }

    elif sne_exp == 'roman':
        sne_fd = 'data/roman/'
        sne_details_fname = '%s/NGR_fCPL.npz' %(sne_fd)

        NGR=np.load( sne_details_fname ) #fiducial cosmology : flat CPL
        z_original_ngrt=NGR['z']
        mb_original_ngrt=NGR['mb']
        sigma_original_ngrt=NGR['sigma']

        z_arr = z_original_ngrt
        mu_arr = mb_original_ngrt
        muerr_stat_arr = sigma_original_ngrt
        ###print( muerr_stat_arr ); sys.exit()
        params = np.asarray( ['omega_m', 'ws', 'wa', 'M'] )
        

        total_roman_sne = len( z_arr )
        sne_arr = np.arange( total_roman_sne )
        muerr_vpec_arr = np.zeros( total_roman_sne )
        muerr_sys_arr = np.zeros( total_roman_sne )
        stretch_x1_arr, color_c_arr = None, None
        sne_cov_tag_dic = {0: 'Stat', 
                          }

    elif sne_exp == 'test':
        sne_fd = 'data/test_SNe/'
        #sne_arr = ['SN2002dc', 'SN2003bd', 'SN2001hb']
        sne_arr = [1, 2, 3]
        z_arr = np.asarray( [0.475, 0.67, 1.] )
        mu_arr = np.asarray( [42.10, 43.14, 44.25] )
        muerr_stat_arr = np.asarray( [0.25, 0.21, 0.14] )
        muerr_vpec_arr = np.zeros( len(z_arr))
        muerr_sys_arr = np.zeros( len(z_arr))
        params = np.asarray( ['omega_m', 'b'] )
        stretch_x1_arr, color_c_arr = None, None
        sne_cov_tag_dic = {0: 'Stat + sys-All', 
        }

    total_zbins = len(z_arr)

    #z-bins
    if z_binning_kind == 'cumulative':
        z_bin_dic = {0: [0., 0.4], 1: [0., 0.6], 2: [0.,1.], 3: [0.,3.]}
    elif z_binning_kind == 'individual':
        z_bin_dic = {0: [0., 0.4], 1: [0.4, 0.6], 2: [0.6,1.], 3: [1.,3.]}
    if sne_exp in ['des', 'des_cobaya', 'test']:##, 'roman']:
        z_bin_dic = {0: [0., 3.0]}
    if sne_exp in ['roman']:
        z_bin_dic = {0: [0., 0.5], 1: [0., 1.], 2: [0.,3.]}

    if (0): #20250207 - LSST checks
        params = np.asarray( ['omega_m', 'ws', 'wa'] )
        z_bin_dic = {0: [0., 3.0]}
        sne_cov_tag_dic = {0: 'Stat + sys-All', 
                          }

    print(obtain_covs); ##sys.exit()
    if obtain_covs:
        #stat cov
        sne_stat_cov = np.diag(muerr_stat_arr**2.)
        ##print(sne_stat_cov); sys.exit()

        sne_tot_cov_inv_dic = {}
        sne_tot_cov_inv_fname = '%s/sne_%s_cov_inv.npy' %(sne_fd, sne_exp)

        ###print(sne_cov_tag_dic.keys()); sys.exit()

        print('\tCov inv file = %s. Exists = %s' %(sne_tot_cov_inv_fname, os.path.exists(sne_tot_cov_inv_fname)))
        sne_sys_cov_dic = {}
        sne_tot_cov_dic = {}
        if os.path.exists( sne_tot_cov_inv_fname ): #sys cov iv
            sne_tot_cov_inv_dic = np.load( sne_tot_cov_inv_fname, allow_pickle = True ).item()
        else:
            #sys cov
            sne_tot_cov_inv_dic = {}
        
        sne_tot_cov_inv_fname_modified = False
        for curr_cov_tag in sne_cov_tag_dic:

            if curr_cov_tag not in reqd_cov_tags: continue

            print('\t\tgetting the covariance and cov inv for tag = %s (%s)' %(curr_cov_tag, sne_cov_tag_dic[curr_cov_tag]))

            if sne_exp == 'lsst_binned':
                sne_cov_fname = '%s/covsys_%03d.txt' %(sne_fd, curr_cov_tag)
            elif sne_exp in ['lsst_unbinned', 'lsst_v2_unbinned', 'lsst_v2_binned']:
                #sne_cov_fname = '%s/covsys_%03d.txt' %(sne_fd, curr_cov_tag)
                sne_cov_fname = '%s/covsys_%03d.txt' %(sne_fd, curr_cov_tag)
            elif sne_exp in ['des', 'des_cobaya']:
                sne_cov_fname = '%s/covsys_%03d.txt' %(sne_fd, curr_cov_tag)
                sne_cov_inv_fname = '%s/covtot_inv_%03d.txt' %(sne_fd, curr_cov_tag)
            elif sne_exp == 'test':
                sne_cov_fname = '%s/covsys_%03d.txt' %(sne_fd, curr_cov_tag)
                sne_cov_inv_fname = '%s/covtot_inv_%03d.txt' %(sne_fd, curr_cov_tag)
            elif sne_exp == 'roman':
                sne_cov_fname = None
                sne_cov_inv_fname = None

            if not os.path.exists(sne_cov_fname): #20250420 - check if .gz exists
                sne_cov_fname_zipped = '%s.gz' %(sne_cov_fname)
                os.system('gunzip %s' %(sne_cov_fname_zipped))

            if sne_cov_fname is not None:
                sne_sys_cov = np.loadtxt(sne_cov_fname, skiprows = 1).reshape(total_zbins, total_zbins)
            else:
                sne_sys_cov = np.zeros( sne_stat_cov.shape )

            sne_sys_cov_dic[curr_cov_tag] = sne_sys_cov
            if curr_cov_tag == 0:
                if add_stat_error:
                    sne_tot_cov = sne_stat_cov + np.copy(sne_sys_cov)
                else:
                    sne_tot_cov = np.copy(sne_sys_cov)
            else:
                sne_tot_cov = sne_stat_cov + sne_sys_cov
            sne_tot_cov_dic[curr_cov_tag] = sne_tot_cov

            if curr_cov_tag in sne_tot_cov_inv_dic: continue

            sne_tot_cov_inv_dic[curr_cov_tag] = np.linalg.inv( sne_tot_cov )
            sne_tot_cov_inv_fname_modified = True

        if (0):##sne_tot_cov_inv_fname_modified:
            print('\t\t\tdumping cov inv file = %s' %(sne_tot_cov_inv_fname))
            np.save( sne_tot_cov_inv_fname, sne_tot_cov_inv_dic )

    #---------
    #20250422 - z-cuts
    if stretch_x1_arr is None: stretch_x1_arr = np.zeros( len(z_arr) )
    if color_c_arr is None: color_c_arr = np.zeros( len(z_arr) )
    sne_cat = np.asarray( [sne_arr, z_arr, mu_arr, muerr_stat_arr, muerr_sys_arr, stretch_x1_arr, color_c_arr] )
    sne_cat, passed_inds = perform_redshift_cuts(zmin, zmax, z_arr, sne_cat)
    sne_arr, z_arr, mu_arr, muerr_stat_arr, muerr_sys_arr, stretch_x1_arr, color_c_arr = sne_cat

    if obtain_covs:
        sne_sys_cov_dic = refine_covs_based_on_cuts(sne_sys_cov_dic, passed_inds)
        sne_stat_cov = refine_covs_based_on_cuts(sne_stat_cov, passed_inds)
        sne_tot_cov_dic = refine_covs_based_on_cuts(sne_tot_cov_dic, passed_inds)
        sne_tot_cov_inv_dic = None #20240422 - inverse yet to be implemented. Only needed for Fisher.
    #---------

    ret_dic = {}
    ret_dic['sne_zarr'] = z_arr
    ret_dic['sne_muarr'] = mu_arr
    ret_dic['sne_muerr_sys'] = muerr_sys_arr
    ret_dic['sne_muerr_stat'] = muerr_stat_arr
    ret_dic['sne_stretch_x1'] = stretch_x1_arr
    ret_dic['sne_color_c'] = color_c_arr
    ret_dic['sne_cov_tag_dic'] = sne_cov_tag_dic
    ret_dic['sne_z_bin_dic'] = z_bin_dic
    ret_dic['sne_params'] = params
    if obtain_covs:
        ret_dic['sne_sys_cov_dic'] = sne_sys_cov_dic
        ret_dic['sne_stat_cov'] = sne_stat_cov
        ret_dic['sne_tot_cov_dic'] = sne_tot_cov_dic
        ret_dic['sne_tot_cov_inv_dic'] = sne_tot_cov_inv_dic
        ret_dic['sne_tot_cov'] = sne_tot_cov_dic

    return ret_dic


def get_distance_modulus_derivatives(zarr, param_dict, params, stretch_x1arr = None, color_carr = None, stepsize_frac = 0.01, baselinecosmo = 'Flatw0waCDM', use_hsq_units = True):
    """    
    obtain derivatives of distance modules (DM) w.r.t cosmological and SNe parameters

    Latex str:
    Eq. (1) in page 6 bottom of https://arxiv.org/abs/2401.02929
    $\mu_{{\rm obs}, i} = m_{x, i} + \alpha x_{1, i} - \beta c_{i} + \gamma G_{{\rm host}, i} - M - \Delta \mu_{{\rm bias}, i}$

    zarr: float array - redshift for which DM derivatives are required
    param_dict: dictionary with parameter values
    params: params to calcualte the derivatives for
    stretch_x1arr: SNe stretch x1. Necessary for alpha.
    color_carr: SNe colour c. Necessary for beta.
    stepsize_frac: stepsize to be used for parameters for the finite difference method.
        Default: 10 per cent of the fiducial parameter value (stepsize_frac * fiducial_parameter_value)
    baselinecosmo: Baseline cosmology to use.
        Default is Flatw0waCDM

    Returns:
    distance_modulus_deriv_dict: distance_modulus derivative dictionary. Keys are the redshift values.
    """

    cosmo_params = ['omch2', 'ws', 'wa', 'omega_m']
    global_additive_bias_params = ['M', 'b']
    sne_additive_bias_params = ['alpha', 'beta']

    if 'alpha' in params:
        assert stretch_x1arr is not None
    if 'beta' in params:
        assert color_carr is not None

    distance_modulus_deriv_dict = {}
    for zcntr, z in enumerate(zarr):
        ###print(z, end = '  ')
        #print('Redshift = %g' %(z))
        distance_modulus_deriv_dict[z] = {}
        for ppp in params:
            pval = param_dict[ppp]
            ##print(ppp, pval)
            if pval == 0:
                pval_step = stepsize_frac
            else:
                pval_step = pval * stepsize_frac

            if ppp in cosmo_params:

                #run twice for finite difference method
                param_dict_mod = copy.deepcopy( param_dict )
                distance_modulus_arr = []
                for diter in range(2):
                    if diter == 0: #pval - step
                        param_dict_mod[ppp] = pval - pval_step
                    elif diter == 1: #pval + step
                        param_dict_mod[ppp] = pval + pval_step

                    #set cosmo
                    ##print(zcntr, z, ppp, param_dict[ppp], param_dict_mod[ppp], set(param_dict.items()) ^ set(param_dict_mod.items())); ##sys.exit()
                    cosmo = set_cosmo(param_dict_mod, baselinecosmo = baselinecosmo, use_hsq_units = use_hsq_units)

                    #get distance modules
                    curr_dist_mod = cosmo.distmod(z).value
                    ##print(ppp, param_dict_mod[ppp], curr_dist_mod)

                    distance_modulus_arr.append( curr_dist_mod )

                #print(distance_modulus_arr); sys.exit()

                ##sys.exit()
                #get derivative
                ##print(distance_modulus_arr); sys.exit()
                deriv_val = ( distance_modulus_arr[1] - distance_modulus_arr[0] ) / (2 * pval_step)
            elif ppp in global_additive_bias_params:
                deriv_val = 1.
            elif ppp in sne_additive_bias_params:
                if ppp == 'alpha':
                    deriv_val = stretch_x1arr[zcntr]
                elif ppp == 'beta': #equation is negative beta
                    deriv_val = -color_carr[zcntr]

            distance_modulus_deriv_dict[z][ppp] = deriv_val

            ##sys.exit()

    return distance_modulus_deriv_dict

def get_binning_operators(ells, ell_bins, ell_weights = None, min_ell = 2, use_dl_for_binning_operators = True, epsilon_for_diag = 1e-8):
    
    #Eqs. (20) and (21) of https://arxiv.org/pdf/astro-ph/0105302.pdf
    total_ells = len(ells)

    if ell_weights is None: ell_weights = np.ones_like(ells)
    total_ell_bins = len( ell_bins )
    pbl = np.zeros( (total_ell_bins, total_ells) ) #N_binned_el x N_ells (Eq. 20 of https://arxiv.org/pdf/astro-ph/0105302.pdf)
    qlb = np.zeros( (total_ells, total_ell_bins) ) #N_ells x N_binned_el (Eq. 21 of https://arxiv.org/pdf/astro-ph/0105302.pdf)
    
    '''    
    epsilon_diag_mat_for_pbl = np.eye( total_ell_bins, total_ells  ) * epsilon_for_diag
    epsilon_diag_mat_for_qlb = np.eye( total_ells, total_ell_bins  ) * epsilon_for_diag

    pbl = pbl + epsilon_diag_mat_for_pbl
    qlb = qlb + epsilon_diag_mat_for_qlb
    '''    

    for bcntr, (b1, b2) in enumerate(ell_bins):
        ##linds = np.where( (ells>=b1) & (ells<=b2) )[0]
        linds = np.where( (ells>=b1) & (ells<=b2) )[0]
        b3 = b2+1
        if len(linds) == 0 or b2<min_ell: continue #make sure \ell >= min_ell.
        if use_dl_for_binning_operators:
            dl_fac = ell_weights[linds] * ells[linds] * (ells[linds]+1)/2/np.pi
        else:
            dl_fac = 1. * ell_weights[linds]

        #pbl[bcntr, linds] = dl_fac * ell_weights[linds] / np.sum(ell_weights[linds]) ##(b2-b1)
        pbl[bcntr, linds] = dl_fac / (b3-b1)
        qlb[linds, bcntr] = 1./dl_fac
        ##print(bcntr, b1, b2, b3, len(linds), pbl[bcntr, linds]/dl_fac)#, ells[linds], dl_fac)#, pbl[bcntr])
        ##print(bcntr, b1, b2, ells[linds]); ##sys.exit()

    return pbl, qlb

def get_bpwf(pbl, qlb, mll, bl = None, fl = None):
    kll = mll
    if bl is not None:
        kll = np.dot( kll, bl**2.)
    if fl is not None:
        kll = np.dot( kll, fl)        
    kbb = np.dot(pbl, np.dot(kll, qlb))
    kbb_inv = np.linalg.inv(kbb) #inverse of kbb
    bpwf = np.dot(kbb_inv, np.dot(pbl, kll) ) #Eq. (25) of https://arxiv.org/pdf/1707.09353.

    return bpwf, kbb, kbb_inv

def get_ell_bin(el_unbinned, delta_el):
    el_binned = np.arange(1, max(el_unbinned)+delta_el, delta_el)
    ell_bins = [(b-delta_el/2, b+delta_el/2) for b in el_binned]

    return el_binned, ell_bins

def perform_binning(el_unbinned, cl_unbinned, delta_el = 100, return_dl = False, bl = None, fl = None, lmin = 30, lmax = 5000, epsilon_for_diag = 1e-8, debug = False):
    #ell_bins = [(b, b+delta_el) for b in el_binned]
    el_binned, ell_bins = get_ell_bin(el_unbinned, delta_el)

    reclen_binned = len( el_binned )
    reclen_unbinned = len( el_unbinned )
    ell_weights = np.ones( reclen_unbinned )
    ell_weights[el_unbinned<lmin] = epsilon_for_diag
    ell_weights[el_unbinned>lmax] = epsilon_for_diag
    pbl, qlb = get_binning_operators(el_unbinned, ell_bins, ell_weights = ell_weights, use_dl_for_binning_operators = return_dl)
    mll = np.diag( np.ones( reclen_unbinned ) )

    #lmin/lmax cuts
    unbinned_inds_to_cut = np.where( (el_unbinned<lmin) & (el_unbinned>lmax) )
    binned_inds_to_cut = np.where( (el_binned<lmin) & (el_binned>lmax))
    cl_unbinned[(el_unbinned<lmin) | (el_unbinned>lmax)] = 0. #lmin/lmax cut
    
    epsilon_diag_mat = np.eye( reclen_unbinned ) * epsilon_for_diag
    mll[(el_unbinned<lmin) | (el_unbinned>lmax)] = 0. #adding a lmin/lmax cut.
    mll = mll + epsilon_diag_mat
    #pbl[[el_binned<lmin, None], el_unbinned<lmin] = 0.
    #qlb[el_unbinned<lmin] = 0.

    ##from IPython import embed; embed()
    bpwf, kbb, kbb_inv = get_bpwf(pbl, qlb, mll, bl = bl, fl = fl)

    pspec_binned = np.dot(kbb_inv, np.dot(pbl, cl_unbinned) ) #Eq. (26) of https://arxiv.org/pdf/astro-ph/0105302.pdf. Note that there is no noise bias here.
    ##pspec_binned[(el_binned<lmin) | (el_binned>lmax)] = 0. #lmin/lmax cut
    ##print( el_binned, lmin, lmax, pspec_binned ); ##sys.exit()

    ###from IPython import embed; embed()

    if debug:
        ax = subplot(111, yscale = 'log')
        plot( el_unbinned, cl_unbinned, color = 'black' )
        plot( el_binned, pspec_binned, color = 'orangered' )
        show()
        color_arr = [cm.jet(int(d)) for d in np.linspace(0., 255, len(bpwf))]

        for b in range(len(bpwf)):
            plot( el_unbinned, bpwf[b], color = color_arr[b])
        show()

    ##from IPython import embed; embed();
    
    return el_binned, pspec_binned, bpwf

def perform_binning_simple(el, cl, delta_el = 50): 
    binned_el = np.arange(1, lmax+delta_el, delta_el)
    binned_cl = np.zeros( len(binned_el) )   
    for lll, l1 in enumerate( binned_el ):
        l2 = l1 + delta_el
        linds = np.where( (el>=l1) & (el<l2) )[0]
        binned_cl[lll] = np.mean( cl[linds])
    return binned_el, binned_cl

def perform_cl_binning(els, cl, delta_l = 1):
    if delta_l == 1:
        return els, cl
    binned_el = np.arange(min(els), max(els)+delta_l, delta_l)
    binned_cl = np.asarray( [np.mean(cl[el:el+delta_l]) for el in binned_el] )
    binned_cl[np.isnan(binned_cl)] = 1e10

    return binned_el, binned_cl

def set_camb(param_dict, thetastar_or_cosmomctheta_or_h = 'h', lmax = None, WantTransfer = True):

    """
    set CAMB cosmology.
    """

    import camb, copy
    from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid
    assert thetastar_or_cosmomctheta_or_h in ['thetastar', 'cosmomc_theta', 'h']

    if lmax is None:
        pars = camb.CAMBparams(max_l_tensor = param_dict['max_l_tensor'], max_eta_k_tensor = param_dict['max_eta_k_tensor'], WantTransfer = WantTransfer)
        #20200623 - setting accuracy/lmax seprately
        pars.set_accuracy(AccuracyBoost = param_dict['AccuracyBoost'], lAccuracyBoost = param_dict['lAccuracyBoost'], lSampleBoost = param_dict['lSampleBoost'],\
            DoLateRadTruncation = param_dict['do_late_rad_truncation'])
        ###pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'])
    else:
        pars = camb.CAMBparams(WantTransfer = WantTransfer)
        pars.set_for_lmax(int(lmax))

    #pars.set_dark_energy(param_dict['ws'])#, wa=param_dict['wa'])
    pars.DarkEnergy = DarkEnergyPPF(w=param_dict['ws'], wa=param_dict['wa'])
    ###pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = param_dict['As'])
    if thetastar_or_cosmomctheta_or_h == 'thetastar':
        pars.set_cosmology(thetastar=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
            omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
            num_massive_neutrinos = param_dict['num_nu_massive'])
        #pars.set_cosmology(cosmomc_theta=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], num_massive_neutrinos = param_dict['num_nu_massive'])
    elif thetastar_or_cosmomctheta_or_h == 'cosmomc_theta':
        pars.set_cosmology(cosmomc_theta=param_dict['cosmomc_theta'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
            omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
            num_massive_neutrinos = param_dict['num_nu_massive'])
        #pars.set_cosmology(cosmomc_theta=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], num_massive_neutrinos = param_dict['num_nu_massive'])
    elif thetastar_or_cosmomctheta_or_h == 'h':
        pars.set_cosmology(H0=param_dict['h']*100., ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
            omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
            num_massive_neutrinos = param_dict['num_nu_massive'])
    
    #20200619
    #print('\n\tswitching order on 20200619 following https://camb.readthedocs.io/en/latest/camb.html\n')
    if lmax is None:
        #pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'])
        pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'],\
            max_eta_k = param_dict['max_eta_k'],\
            #lens_k_eta_reference = param_dict['max_eta_k'],\
            )

    if param_dict['As']>3.:
        pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = np.exp(param_dict['As'])/1e10, nrun = param_dict['nrun'])
    else:
        pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = param_dict['As'], nrun = param_dict['nrun'])

    results = camb.get_results(pars)
    
    return pars, results

def get_camb_cl(param_dict, which_spectra, raw_cl = True, thetastar_or_cosmomctheta_or_h = 'h', delta_l = 1, required_spectra = ['TT', 'EE', 'TE', 'BB', 'PP', 'Tphi', 'Ephi'], return_dl = False, lmin_dic = None, lmax_dic = None):


    """
    set CAMB cosmology and get power spectra in uK^2 units
    """

    print('set CAMB cosmology and get power spectra in uK^2 units')

    import camb, copy
    from camb.dark_energy import DarkEnergyPPF, DarkEnergyFluid
    assert thetastar_or_cosmomctheta_or_h in ['thetastar', 'cosmomc_theta', 'h']

    pars = camb.CAMBparams(max_l_tensor = param_dict['max_l_tensor'], max_eta_k_tensor = param_dict['max_eta_k_tensor'])
    #20200623 - setting accuracy/lmax seprately
    pars.set_accuracy(AccuracyBoost = param_dict['AccuracyBoost'], lAccuracyBoost = param_dict['lAccuracyBoost'], lSampleBoost = param_dict['lSampleBoost'],\
        DoLateRadTruncation = param_dict['do_late_rad_truncation'])
    ###pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'])

    #pars.set_dark_energy(param_dict['ws'])#, wa=param_dict['wa'])
    pars.DarkEnergy = DarkEnergyPPF(w=param_dict['ws'], wa=param_dict['wa'])
    ###pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = param_dict['As'])
    if thetastar_or_cosmomctheta_or_h == 'thetastar':
        pars.set_cosmology(thetastar=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
            omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
            num_massive_neutrinos = param_dict['num_nu_massive'])
        #pars.set_cosmology(cosmomc_theta=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], num_massive_neutrinos = param_dict['num_nu_massive'])
    elif thetastar_or_cosmomctheta_or_h == 'cosmomc_theta':
        pars.set_cosmology(cosmomc_theta=param_dict['cosmomc_theta'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
            omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
            num_massive_neutrinos = param_dict['num_nu_massive'])
        #pars.set_cosmology(cosmomc_theta=param_dict['thetastar'], ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], num_massive_neutrinos = param_dict['num_nu_massive'])
    elif thetastar_or_cosmomctheta_or_h == 'h':
        pars.set_cosmology(H0=param_dict['h']*100., ombh2=param_dict['ombh2'], omch2=param_dict['omch2'], nnu = param_dict['neff'], mnu=param_dict['mnu'], \
            omk=param_dict['omk'], tau=param_dict['tau'], YHe = param_dict['YHe'], Alens = param_dict['Alens'], \
            num_massive_neutrinos = param_dict['num_nu_massive'])
    #20200619
    #print('\n\tswitching order on 20200619 following https://camb.readthedocs.io/en/latest/camb.html\n')
    #pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'])
    pars.set_for_lmax(int(param_dict['max_l_limit']), lens_potential_accuracy=param_dict['lens_potential_accuracy'],\
        max_eta_k = param_dict['max_eta_k'],\
        #lens_k_eta_reference = param_dict['max_eta_k'],\
        )

    if param_dict['As']>3.:
        pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = np.exp(param_dict['As'])/1e10, nrun = param_dict['nrun'])
    else:
        pars.InitPower.set_params(ns=param_dict['ns'], r=param_dict['r'], As = param_dict['As'], nrun = param_dict['nrun'])


    els = np.arange(param_dict['min_l_limit'], param_dict['max_l_limit']+1)

    results = camb.get_results(pars)

    #get dictionary of CAMb power spectra
    powers = results.get_cmb_power_spectra(pars, lmax = param_dict['max_l_limit'], raw_cl = raw_cl)#, spectra = [which_spectra])#, CMb_unit=None, raw_cl=False)

    #get only the required ell range since powerspectra start from ell=0 by default
    for keyname in powers:
        powers[keyname] = powers[keyname][param_dict['min_l_limit']:, :]

    if not raw_cl: #20200529: also valid for lensing (see https://camb.readthedocs.io/en/latest/_modules/camb/results.html#CAMbdata.get_lens_potential_cls)
        powers[which_spectra] = powers[which_spectra] * 2 * np.pi / (els[:,None] * (els[:,None] + 1 ))


    #tcmb factor
    if pars.OutputNormalization == 1:
        powers[which_spectra] = param_dict['tcmb']**2. *  powers[which_spectra]

    #uK
    powers[which_spectra] *= 1e12
    cl_tt, cl_ee, cl_bb, cl_te = powers[which_spectra].T

    #lensing
    cl_phiphi, cl_tphi, cl_ephi = powers['lens_potential'].T
    #cl_tphi *= 1e6
    #cl_ephi *= 1e6
    cl_phiphi = cl_phiphi# * (els * (els+1))**2. /(2. * np.pi)
    cl_tphi = cl_tphi# * (els * (els+1))**1.5 /(2. * np.pi)
    cl_ephi = cl_ephi# * (els * (els+1))**1.5 /(2. * np.pi)

    cl_dic = {}
    cl_dic['els'] = els
    if 'TT' in required_spectra: cl_dic['TT'] = cl_tt
    if 'EE' in required_spectra: cl_dic['EE'] = cl_ee
    if 'BB' in required_spectra: cl_dic['BB'] = cl_bb
    if 'TE' in required_spectra: cl_dic['TE'] = cl_te
    if 'PP' in required_spectra: cl_dic['PP'] = cl_phiphi
    if 'Tphi' in required_spectra: cl_dic['Tphi'] = cl_tphi
    if 'Ephi' in required_spectra: cl_dic['Ephi'] = cl_ephi

    #binning
    if delta_l > 1:
        cl_dic_binned = {}
        for XX in cl_dic:
            ###from IPython import embed; embed()
            if lmin_dic is not None and XX != 'els':
                lmin = lmin_dic[XX]
            else:
                lmin = param_dict['min_l_limit']
            if lmax_dic is not None and XX != 'els':
                lmax = lmax_dic[XX]
            else:
                lmax = param_dict['max_l_limit']
            #binned_el, binned_cl = perform_cl_binning(els, cl_dic[XX], delta_l = delta_l)
            binned_el, binned_cl, bpwf = perform_binning(els, cl_dic[XX], delta_el = delta_l, return_dl = return_dl, lmin = lmin, lmax = lmax)
            cl_dic_binned[XX] = binned_cl
        els = binned_el
        cl_dic = cl_dic_binned


    if (0):
        #from IPython import embed; embed()  
        ax = subplot(111, yscale = 'log');
        dls_fac = (els * (els+1)) /(2. * np.pi)
        plot(cl_tt * dls_fac, 'k-'); 
        plot(cl_ee * dls_fac, 'r-'); plot(cl_te * dls_fac, 'g-'); 
        plot(cl_bb * dls_fac, 'b-'); 
        show()
        sys.exit()

    return pars, cl_dic


def get_camb_cl_and_derivatives(params, param_dict, param_steps_dict = None, stepsize_frac = 0.01, which_spectra = 'lensed_scalar', delta_l = 1., thetastar_or_cosmomctheta_or_h = 'h', get_derivatives = True):

    import copy

    #Fiducial Cl
    print('\nGet fiducial CAMB power spectra. Spectra = %s' %(which_spectra))
    pars, cl_dic = get_camb_cl(param_dict, which_spectra, thetastar_or_cosmomctheta_or_h = thetastar_or_cosmomctheta_or_h)

    cl_deriv_dic = None
    if get_derivatives: #derivatives
        print('\nGet derivatives of CAMB power spectra. Spectra = %s' %(which_spectra))
        cl_deriv_dic = {}
        for ppp in sorted(params):

            print('\tparam for derivative = %s' %(ppp))###; sys.exit()

            #modify param values        
            pval = param_dict[ppp]
            param_dict_low = copy.deepcopy( param_dict )
            param_dict_high = copy.deepcopy( param_dict )
            if param_steps_dict is not None:
                pval_step = param_steps_dict[ppp]
            else:
                if param_dict[ppp] == 0: #if the param's fiducial value if zero
                    pval_step = stepsize_frac
                else:
                    pval_step = pval * stepsize_frac
            pval_low = pval - pval_step
            pval_high = pval + pval_step
            param_dict_low[ppp] = pval_low
            param_dict_high[ppp] = pval_high
            print('\t\tvalues: fid=%g, low=%g, high=%g, step=%g' %(pval, pval_low, pval_high, pval_step)); ##sys.exit()

            dummypars, cl_mod_dic_low = get_camb_cl(param_dict_low, which_spectra, thetastar_or_cosmomctheta_or_h = thetastar_or_cosmomctheta_or_h)
            dummypars, cl_mod_dic_high = get_camb_cl(param_dict_high, which_spectra, thetastar_or_cosmomctheta_or_h = thetastar_or_cosmomctheta_or_h)

            cl_deriv_dic[ppp] = {}
            for XX in cl_dic: #loop over TT, EE, BB, TE, and lensing
                cl_deriv_dic[ppp][XX] = (cl_mod_dic_high[XX] - cl_mod_dic_low[XX]) / (2*pval_step)
                print('\t\t\t%s:' %(XX), cl_deriv_dic[ppp][XX])

    return pars, cl_dic, cl_deriv_dic


def get_sne_distance_modulus_fisher(params, covariance, zarr, distance_modulus_deriv_dict, cov_inv = None):


    npar = len(params)
    F = np.zeros([npar,npar])

    if cov_inv is None:
        #cov_inv = sc.linalg.pinv2(covariance)
        cov_inv = sc.linalg.pinv(covariance)

    ###print( cov_inv ); sys.exit()

    param_combinations = []
    for pcnt,p in enumerate(params):
        for pcnt2,p2 in enumerate(params):
            param_combinations.append([p,p2, pcnt, pcnt2])


    for (p1,p2, pcnt1, pcnt2) in param_combinations:

        if pcnt2<pcnt1:continue
        ##print(p1, p2)

        der1 = np.asarray( [distance_modulus_deriv_dict[z][p1] for z in zarr] )
        der2 = np.asarray( [distance_modulus_deriv_dict[z][p2] for z in zarr] )

        ###curr_val = np.dot(der1, np.dot( cov_inv, der2 ))
        curr_val = der1 @ ( cov_inv @ der2 )
        #print(curr_val); sys.exit()

        F[pcnt2,pcnt1] += curr_val
        if pcnt1 != pcnt2:
            F[pcnt1,pcnt2] += curr_val

        '''
        for zcntr, z in enumerate( zarr ):
            der1 = distance_modulus_deriv_dict[z][p1]
            der2 = distance_modulus_deriv_dict[z][p2]
            sigma_sq = covariance[zcntr, zcntr]

            curr_val = der1 * der2 / sigma_sq

            F[pcnt2,pcnt1] += curr_val
        '''


    return F 

def get_cmb_cl_cov(TT, EE, TE, PP, TP, EP):

    C = np.zeros( (3,3) ) #TT, EE, PP
    C[0,0] = TT
    C[1,1] = EE
    C[0,1] = C[1,0] = TE

    C[2,2] = PP
    C[0,2] = C[2,0] = TP
    C[1,2] = C[2,1] = EP ##0. ##EP

    return np.mat( C )

def get_cmb_fisher(els, cl_deriv_dic, delta_cl_dic, params, pspectra_to_use, lmin_t = 0, lmax_t = 10000, lmin_p = 0, lmax_p = 10000, lmin_phi = 0, lmax_phi = 10000):

    import scipy.linalg 
    npar = len(params)
    F = np.zeros([npar,npar])
    #els = np.arange( len( delta_cl_dic.values()[0] ) )

    if 'PP' in pspectra_to_use:
        with_lensing = 1
    else:
        with_lensing = 0

    all_pspectra_to_use = []
    for tmp in pspectra_to_use:
        if isinstance(tmp, list):      
            all_pspectra_to_use.extend(tmp)
        else:
            all_pspectra_to_use.append(tmp)

    for lcntr, l in enumerate( els ):

        TT, EE, TE = 0., 0., 0.
        Tphi = Ephi = PP = 0.
        if 'TT' in delta_cl_dic:
            TT = delta_cl_dic['TT'][lcntr]
        if 'EE' in delta_cl_dic:
            EE = delta_cl_dic['EE'][lcntr]
        if 'TE' in delta_cl_dic:
            TE = delta_cl_dic['TE'][lcntr]
        if with_lensing:
            #Tphi, Ephi, PP = delta_cl_dic['Tphi'][lcntr], delta_cl_dic['Ephi'][lcntr], delta_cl_dic['PP'][lcntr]
            if 'Tphi' in delta_cl_dic:
                Tphi = delta_cl_dic['Tphi'][lcntr]
            else:
                Tphi = 0.

            if 'Ephi' in delta_cl_dic:
                Ephi = delta_cl_dic['Ephi'][lcntr]
            else:
                Ephi  = 0.
            PP = delta_cl_dic['PP'][lcntr]

        null_TT, null_EE, null_TE = 0, 0, 0
        if l<lmin_t or l>lmax_t:
            null_TT = 1
        if l<lmin_p or l>lmax_p: 
            null_EE = 1
            null_TE = 1
        null_PP = 0
        if l<lmin_phi or l>lmax_phi:
            null_PP = 1 #Lensing noise curves already have pretty large noise outside desired L range
        #if l<min_l_TE or l>max_l_TE:  
        #    null_TE = 1

        #20200611
        if 'TT' not in all_pspectra_to_use:
            null_TT = 1
        if 'EE' not in all_pspectra_to_use:
            null_EE = 1
        if 'TE' not in all_pspectra_to_use:
            #if 'TT' not in pspectra_to_use and 'EE' not in pspectra_to_use:
            #    null_TE = 1
            if 'TT' in pspectra_to_use and 'EE' in pspectra_to_use:
                null_TE = 0
            else:
                null_TE = 1
        if ['TT', 'EE', 'TE'] in pspectra_to_use:
            null_TT = 0
            null_EE = 0
            null_TE = 0
        #20200611

        ##if (null_TT and null_EE and null_TE): continue# and null_PP): continue
        if (null_TT and null_EE and null_TE and null_PP): continue

        param_combinations = []
        for pcnt,p in enumerate(params):
            for pcnt2,p2 in enumerate(params):
                ##if [p2,p,pcnt2,pcnt] in param_combinations: continue
                param_combinations.append([p,p2, pcnt, pcnt2])

        #nulling unwanted fields
        if null_TT and null_TE: TT = 0
        if null_EE and null_TE: EE = 0
        #if null_TE and (null_TT and null_EE): TE = 0
        if null_TE: 
            if not null_TT and not null_EE:
                pass
            else:
                TE = 0
        if null_PP: PP = Tphi = EPhi = 0
        if null_TT: Tphi = 0
        if null_EE: Ephi = 0
        #nulling unwanted fields

        COV_mat_l = get_cmb_cl_cov(TT, EE, TE, PP, Tphi, Ephi)
        if np.sum( COV_mat_l ) == 0.: continue
        inv_COV_mat_l = linalg.pinv(COV_mat_l)
        ##inv_COV_mat_l = np.linalg.inv(COV_mat_l)
        
        if (0):##l%500 == 0: 
            from IPython import embed; embed()
            print(l, null_TT, null_EE, null_TE, null_PP)
            print(COV_mat_l)

        for (p,p2, pcnt, pcnt2) in param_combinations:


            TT_der1, EE_der1, TE_der1 = 0., 0., 0.
            TT_der2, EE_der2, TE_der2 = 0., 0., 0.

            if 'TT' in cl_deriv_dic[p]:
                TT_der1 = cl_deriv_dic[p]['TT'][lcntr]
                TT_der2 = cl_deriv_dic[p2]['TT'][lcntr]
            if 'EE' in cl_deriv_dic[p]:
                EE_der1 = cl_deriv_dic[p]['EE'][lcntr]
                EE_der2 = cl_deriv_dic[p2]['EE'][lcntr]
            if 'TE' in cl_deriv_dic[p]:
                TE_der1 = cl_deriv_dic[p]['TE'][lcntr]
                TE_der2 = cl_deriv_dic[p2]['TE'][lcntr]


            if with_lensing:
                PP_der1, TPhi_der1, EPhi_der1 = cl_deriv_dic[p]['PP'][lcntr], cl_deriv_dic[p]['Tphi'][lcntr], cl_deriv_dic[p]['Ephi'][lcntr]
                PP_der2, TPhi_der2, EPhi_der2 = cl_deriv_dic[p2]['PP'][lcntr], cl_deriv_dic[p2]['Tphi'][lcntr], cl_deriv_dic[p2]['Ephi'][lcntr]
            else:
                PP_der1 = PP_der2 = 0.
                TPhi_der1 = TPhi_der2 = 0. 
                EPhi_der1 = EPhi_der2 = 0.


            if null_TT: TT_der1 = TT_der2 = TPhi_der1 = TPhi_der2 = 0
            if null_EE: EE_der1 = EE_der2 = EPhi_der1 = EPhi_der2 = 0
            if null_TE: TE_der1 = TE_der2 = 0
            if null_PP: PP_der1 = PP_der2 = 0

            fprime1_l_vec = get_cmb_cl_cov(TT_der1, EE_der1, TE_der1, PP_der1, TPhi_der1, EPhi_der1)
            fprime2_l_vec = get_cmb_cl_cov(TT_der2, EE_der2, TE_der2, PP_der2, TPhi_der2, EPhi_der2)

            curr_val = np.trace( np.dot( np.dot(inv_COV_mat_l, fprime1_l_vec), np.dot(inv_COV_mat_l, fprime2_l_vec) ) )

            F[pcnt2,pcnt] += curr_val

    return F   


def get_bias_vector(F_mat, params, z_arr, covariance, mu_arr, mu_sys_arr, distance_modulus_deriv_dict, cov_inv = None, ignore_sys_cov = False):

    if cov_inv is None:
        #cov_inv = sc.linalg.pinv2(covariance)
        cov_inv = sc.linalg.pinv(covariance)
    if covariance is None:
        covariance = sc.linalg.pinv(cov_inv)
    if (0):
        cov_sqrt = np.sqrt( covariance )
        cov_sqrt_inv = sc.linalg.pinv(cov_sqrt)
        print(cov_sqrt); sys.exit()

    bias_vector = []
    for pcntr, p in enumerate( params ):
        der_vec = np.asarray( [distance_modulus_deriv_dict[z][p] for z in z_arr] )
        '''
        tmp1 = np.dot(cov_inv, der_vec) 
        tmp2 = np.dot(cov_inv, mu_sys_arr)
        print(der_vec); 
        print(np.diag(cov_inv))
        print(mu_sys_arr)
        sys.exit()
        '''
        '''
        ##print( der_vec.shape, cov_inv.shape, mu_sys_arr.shape ); sys.exit()
        curr_bias_val_wrong = np.dot( np.dot(cov_inv, mu_sys_arr), np.dot(cov_inv, der_vec) )
        curr_bias = np.dot( np.dot(cov_sqrt_inv, mu_sys_arr), np.dot(cov_sqrt_inv, der_vec) )
        print( curr_bias );
        if (0):
            tmp_cov = np.diag( np.diag(covariance) )
            tmp_cov_inv = sc.linalg.pinv(tmp_cov)
            curr_bias_val_wrong = np.dot( np.dot(tmp_cov_inv, mu_sys_arr), np.dot(tmp_cov_inv, der_vec) )
            print(curr_bias_val_wrong)
        '''

        if ignore_sys_cov:
            covariance = np.diag( np.diag(covariance) )
        cov_inv = sc.linalg.pinv(covariance)
        tmp_mu_sys_arr = np.mat(mu_sys_arr)
        tmp_der_vec = np.mat(der_vec).T
        curr_bias_val = tmp_mu_sys_arr @ (cov_inv @ tmp_der_vec)
        curr_bias_val = np.array(curr_bias_val)[0][0]

        if (0):
            tmp_arr = []
            for zcntr in range( len(z_arr) ):
                tmpval = mu_sys_arr[zcntr] * distance_modulus_deriv_dict[z_arr[zcntr]][p] / covariance[zcntr, zcntr]

                tmp_arr.append( tmpval )
            curr_bias_val = np.sum( tmp_arr )

        ##print(curr_bias_val); sys.exit()
        
        bias_vector.append( curr_bias_val )
        ##print('\t\t%s, %s' %(p, bias_vector[pcntr]))
    ##sys.exit()

    bias_vector = np.asarray(bias_vector)
    C_mat = np.linalg.inv(F_mat)
    final_bias_vector = np.asarray( np.dot( np.mat(bias_vector), C_mat ) )[0]
    ##print( final_bias_vector ); sys.exit()
    
    return final_bias_vector

def logL_to_L(logL):
    tmp_logL = np.copy(logL) - np.max(logL)
    L = np.exp(tmp_logL)
    L = L / np.max(L)        
    return L

def process_2D_loglike_like(logL_or_L_grid, logL_or_L = 'logL', return_delta_chisq = False):

    if logL_or_L == 'logL': #convert to like
        L_grid = logL_to_L(logL_or_L_grid)
        delta_chi_sq_grid = -2 * logL_or_L_grid
    else:
        L_grid = np.copy(logL_or_L_grid)
        delta_chi_sq_grid = None

    L_arr_for_ax1 = np.mean( L_grid, axis = 0 )
    L_arr_for_ax1 /= np.max(L_arr_for_ax1)
    L_arr_for_ax0 = np.mean( L_grid, axis = 1 )
    L_arr_for_ax0 /= np.max(L_arr_for_ax0)

    if return_delta_chisq:
        return delta_chi_sq_grid, L_grid, L_arr_for_ax0, L_arr_for_ax1
    else:
        return L_grid, L_arr_for_ax0, L_arr_for_ax1
    
def process_3D_loglike_like(logL_or_L_grid, logL_or_L = 'logL', return_delta_chisq = False):

    if logL_or_L == 'logL': #convert to like
        L_grid = logL_to_L(logL_or_L_grid)
        delta_chi_sq_grid = -2 * logL_or_L_grid
    else:
        L_grid = np.copy(logL_or_L_grid)
        delta_chi_sq_grid = None

    L_arr_for_ax01 = np.mean( L_grid, axis = (2) )
    L_arr_for_ax01 /= np.max(L_arr_for_ax01)
    L_arr_for_ax02 = np.mean( L_grid, axis = (1) )
    L_arr_for_ax02 /= np.max(L_arr_for_ax02)
    L_arr_for_ax12 = np.mean( L_grid, axis = (0) )
    L_arr_for_ax12 /= np.max(L_arr_for_ax12)

    L_arr_for_ax0 = np.mean( L_grid, axis = (1,2) )
    L_arr_for_ax0 /= np.max(L_arr_for_ax0)
    L_arr_for_ax1 = np.mean( L_grid, axis = (0,2) )
    L_arr_for_ax1 /= np.max(L_arr_for_ax1)
    L_arr_for_ax2 = np.mean( L_grid, axis = (0,1) )
    L_arr_for_ax2 /= np.max(L_arr_for_ax2)
    
    if return_delta_chisq:
        return delta_chi_sq_grid, L_arr_for_ax01, L_arr_for_ax02, L_arr_for_ax12, L_arr_for_ax0, L_arr_for_ax1, L_arr_for_ax2
    else:
        return L_arr_for_ax01, L_arr_for_ax02, L_arr_for_ax12, L_arr_for_ax0, L_arr_for_ax1, L_arr_for_ax2
    
def get_and_process_logL_grid(chains_dic):

    parent_logL_grid = {}
    for chainname in chains_dic:
        print('\n%s' %(chainname))
        fd = chains_dic[chainname][0].replace('cobaya', 'grid_logl')
        ###fd = fd.replace('chains/', 'chains/testing/')
        fname = '%s/%s_logl.npy' %(fd, chainname)
        if not os.path.exists( fname ):
            #print('%s: %s does not exist' %(chainame, fname))
            print('\t%s does not exist' %(chainname))
            continue

        res_dic = np.load( fname, allow_pickle = True ).item()
        logl_grid, param_value_arr_dic = res_dic['logl_grid'], res_dic['param_value_arr_dic']
        ##print(logl_grid); #sys.exit()
        ##minval = np.min( logl_grid[ np.isnan(logl_grid) == False] )
        ##logl_grid[np.isnan(logl_grid)] = minval
        curr_param_names = list( param_value_arr_dic.keys() )
        curr_total_params = len( curr_param_names )

        if curr_total_params == 1:
            xarr = param_value_arr_dic[ curr_param_names[0] ]
            loglarr = logl_grid
            Larr = logL_to_L(loglarr)
            ##print( np.unique(Larr)); sys.exit()
            ##plot(xarr, Larr); show(); sys.exit()
            parent_logL_grid[chainname] = [xarr, Larr]
        elif curr_total_params == 2:
            xarr = param_value_arr_dic[ curr_param_names[0] ]
            yarr = param_value_arr_dic[ curr_param_names[1] ]
            loglarr = logl_grid.reshape( (len(xarr), len(yarr) ) )
            ##print(loglarr.shape, xarr.shape, yarr.shape); sys.exit()
            delta_chi_sq_grid, L_grid, L_arr_for_x, L_arr_for_y = process_2D_loglike_like(loglarr, logL_or_L = 'logL', return_delta_chisq = True)
            imshow( loglarr, aspect = 'auto', extent = [min(xarr), max(xarr), min(yarr), max(yarr)] ); 
            colorbar(); show(); sys.exit()
            ##plot(xarr, L_arr_for_x); show(); sys.exit()
            parent_logL_grid[chainname] = [xarr, yarr, L_grid, L_arr_for_x, L_arr_for_y]
        elif curr_total_params == 3:
            xarr = param_value_arr_dic[ curr_param_names[0] ]
            yarr = param_value_arr_dic[ curr_param_names[1] ]
            zarr = param_value_arr_dic[ curr_param_names[2] ]
            loglarr = logl_grid.reshape( (len(xarr), len(yarr), len(zarr) ) )
            parent_logL_grid[chainame] = [xarr, yarr, zarr, loglarr]
            delta_chi_sq_grid, L_arr_for_ax01, L_arr_for_ax02, L_arr_for_ax12, L_arr_for_ax0, L_arr_for_ax1, L_arr_for_ax2 = process_3D_loglike_like(loglarr, logL_or_L = 'logL', return_delta_chisq = True)            
            parent_logL_grid[chainname] = [xarr, yarr, L_arr_for_ax01, L_arr_for_ax02, L_arr_for_ax12, L_arr_for_ax0, L_arr_for_ax1, L_arr_for_ax2]

        print(res_dic.keys(), curr_param_names, loglarr.shape )

    return parent_logL_grid   

def get_params_chainame_folder(datasets, param_dict, which_cosmo, sne_exp, bao_exp, bao_dr, chains_fd = 'chains/', zmin = -1, zmax = -1, lsst_sim_no = 3, add_weights = 1, fit_for_scriptM = 1, marginalize_abs_mag = 0, use_ideal_data = 0, testing = 0, switch_to_sne_input_cosmo = 0, theory = 'astropy', sampler = 'cobaya'):

    '''
    params_for_dataset_dic = {'sne': ['omch2', 'ws', 'wa'], 
                              'cmb': ['As', 'ns', 'ombh2', 'omch2', 'h', 'tau', 'ws', 'wa'], 
                              'bao': ['ombh2', 'omch2', 'h', 'ws', 'wa'], 
                              }
    '''

    datasets_str = '_'.join(datasets)
    exp_str = ''
    if 'sne' in datasets:
        exp_str = '%s-sne_%s' %(exp_str, sne_exp)
    if 'bao' in datasets:
        exp_str = '%s-bao_%s%s' %(exp_str, bao_exp, bao_dr)
    exp_str = exp_str.strip('-')

    #chain names
    chainname = "%s__%s" %(which_cosmo, exp_str)

    #output folder
    op_fd = '%s/%s_likelihoods/' %(chains_fd, datasets_str)
    op_fd = '%s/%s/%s/%s/' %(op_fd, exp_str, theory, sampler)

    if testing:
        op_fd = '%s/testing/' %(op_fd)

    if 'sne' in datasets:

        #sim number
        if sne_exp == 'lsst_unbinned':
            op_fd = '%s/sim%s' %(op_fd, lsst_sim_no)

        #change input cosmology
        if switch_to_sne_input_cosmo:
            op_fd = '%s/snana_cosmo/' %(op_fd)    
        
        #redshift cuts
        if zmin != -1 and zmax != -1:
            op_fd =  '%s/sne_zmin%g_zmax%g/' %(op_fd, zmin, zmax)
        elif zmin != -1:
            op_fd =  '%s/sne_zmin%g/' %(op_fd, zmin)
        elif zmax != -1:
            op_fd =  '%s/sne_zmax%g/' %(op_fd, zmax)
        
        #weighted SNe or unweighted
        if not add_weights:
            op_fd = '%s/unweighted/' %(op_fd)

        #marginalise or fit for abs_mag
        if marginalize_abs_mag:
            op_fd = '%s/marginalize_abs_mag/' %(op_fd)
        elif fit_for_scriptM:
            op_fd = '%s/fit_for_scriptM/' %(op_fd)

        if use_ideal_data:
            op_fd = '%s/ideal_data/' %(op_fd)

    output = '%s/%s' %(op_fd, chainname)

    #----
    #params
    mcmc_input_params_info_dict = {}
    mcmc_input_params_info_dict["omch2"] = {
                    "prior": {"min": 0.001, "max": 0.99},
                    #"prior": {"min": 0.05, "max": 0.25},
                    "ref": {"dist": "norm", "loc": 0.12, "scale": 0.001},
                    "proposal": 0.0005,
                    "drop": False, 
                    "latex": r"\Omega_\mathrm{c} h^2", 
                    }
    if which_cosmo in ['hlcdm', 'hw0lcdm', 'hw0walcdm']:
        """
        mcmc_input_params_info_dict["h"] = {
                        "prior": {"min": 0.3, "max": 1.},
                        "ref": {"dist": "norm", "loc": 0.7, "scale": 0.1},
                        "proposal": 0.001,
                        "drop": False, 
                        "latex": r"h", 
                        }
        """
        mcmc_input_params_info_dict["H0"] = {
                        "prior": {"min": 30., "max": 100.},
                        "ref": {"dist": "norm", "loc": 70., "scale": 1.},
                        "proposal": 0.001,
                        "drop": False, 
                        "latex": r"H_{0}", 
                        }

    if which_cosmo in ['w0lcdm', 'w0walcdm', 'hw0lcdm', 'hw0walcdm']:
        mcmc_input_params_info_dict["w"] = {
                         "prior": {"min": -10., "max": 10.},
                         #"prior": {"min": -3., "max": 3.},
                         "ref": {"dist": "norm", "loc": -1, "scale": 0.2},
                         "proposal": 0.001,
                         "drop": False, 
                         "latex": r"w_0",
                         }

    if which_cosmo in ['w0walcdm', 'hw0walcdm']:
        mcmc_input_params_info_dict["wa"] = {
                        "prior": {"min": -30., "max": 20.},
                        #"prior": {"min": -3., "max": 3.},
                        "ref": {"dist": "norm", "loc": 0., "scale": 0.2},
                        "proposal": 0.001,
                        "drop": False, 
                        "latex": r"w_a"
                        }

    if fit_for_scriptM and 'sne' in datasets:
        mcmc_input_params_info_dict["scriptM"] = {
                    "prior": {"min": -10, "max": 10.},
                    "ref": {"dist": "norm", "loc": 0., "scale": 0.2},
                    "proposal": 0.001,
                    "drop": False, 
                    "latex": r"\mathcal{M}",                    
                    }
    
    mcmc_input_params_info_dict["tau"] = {
        "latex": r'\tau_\mathrm{reio}', 
        "value": param_dict['tau'],
        }
    mcmc_input_params_info_dict["logA"] = {
        "value": param_dict['logA'],
        "latex": r"ln10^{10}A_s",
        "drop": True,
        }
    mcmc_input_params_info_dict["As"] = {
        "value": 'lambda logA: 1e-10*np.exp(logA)',
        "latex": r"A_\mathrm{s}",
        "derived": True,
        }
    mcmc_input_params_info_dict["ns"] = {
        "value": param_dict['ns'], 
        "latex": r"n_\mathrm{s}",
        }
    """
    mcmc_input_params_info_dict['rdrag'] = {
                "derived": True,
                "latex": "r_\mathrm{drag}",
                }
    """


    #other derived params:
    """
    mcmc_input_params_info_dict['omegam'] = {
                    "derived": True,
                    "latex": "\Omega_\mathrm{m}",
                    }
    """
    #----

    return mcmc_input_params_info_dict, chainname, op_fd, output


def get_likelihood_from_loglikelihood(logL):
    logL = np.asarray( logL )
    logL = logL - np.max(logL)
    L = np.exp( logL )
    L /= np.max(L)
    return L

