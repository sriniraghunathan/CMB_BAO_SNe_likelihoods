import numpy as np, sys, os, glob
from scipy.io import readsav
from scipy import interpolate as intrp
import scipy as sc
from scipy import integrate, stats

def get_teb_spec_combination(cl_dict):

    """
    uses cl_dict to determine if we are using ILC jointly for T/E/B.

    Parameters
    ----------
    cl_dict : dict
        dictionary containing (signal+noise) auto- and cross- spectra of different freq. channels.

    Returns
    -------
    nspecs : int
        tells if we are performing ILC for T alone or T/E/B together.
        default is 1. For only one map component.

    specs : list
        creates ['TT', 'EE', 'TE', ... etc.] based on cl_dict that is supplied.
        For example:
        ['TT'] = ILC for T-only
        ['EE'] = ILC for E-only
        ['TT', 'EE'] = ILC for T and E separately.
        ['TT', 'EE', 'TE'] = ILC for T and E jointly.
    """

    # fix-me. Do this in a better way.
    specs = sorted(list(cl_dict.keys()))

    if specs == ['TT'] or specs == ['EE'] or specs == ['TE'] or specs == ['BB']:  # only TT is supplied
        nspecs = 1
    elif specs == sorted(['TT', 'EE']) or specs == sorted(
        ['TT', 'EE', 'TE']
    ):  # TT/EE/TE are supplied
        nspecs = 2
    elif specs == sorted(['TT', 'EE', 'BB']) or specs == sorted(
        ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    ):  # TT/EE/BB are supplied
        nspecs = 3
    else:
        logline = 'cl_dict must contain TT/EE/BB spectra or some combination of that'
        raise ValueError(logline)

    return nspecs, specs

def create_covariance(bands, elcnt, cl_dict):

    """
    Creates band-band covariance matrix at each el

    Parameters
    ----------
    bands : array
        array of frequency bands for which we need the covariance.
    elcnt : int
        ell index.
    cl_dict : dict
        dictionary containing (signal+noise) auto- and cross- spectra of different freq. channels.

    Returns
    -------
    cov: array
        band-band covariance matrix at each ell. dimension is nband x nband.
    """

    nc = len(bands)
    nspecs, specs = get_teb_spec_combination(cl_dict)
    cov = np.zeros((nspecs * nc, nspecs * nc))

    for specind, spec in enumerate(specs):
        curr_cl_dict = cl_dict[spec]
        if nspecs == 1:  # cov for TT or EE or BB
            for ncnt1, band1 in enumerate(bands):
                for ncnt2, band2 in enumerate(bands):
                    j, i = ncnt2, ncnt1
                    cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
        else:  # joint or separate TT/EE constraints #fix me: include BB for joint constraints.
            if spec == 'TT':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2, ncnt1
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
            elif spec == 'EE':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2 + nc, ncnt1 + nc
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
            elif spec == 'TE':
                for ncnt1, band1 in enumerate(bands):
                    for ncnt2, band2 in enumerate(bands):
                        j, i = ncnt2 + nc, ncnt1
                        cov[j, i] = curr_cl_dict[(band1, band2)][elcnt]
                        cov[i, j] = curr_cl_dict[(band1, band2)][elcnt]

    return cov

def apply_TPcal_to_cmb_spec_dic(cl_dic, spec, bands, map_cal_arr = None, map_cal_arr2 = None):
    """
    #Check Eq.(3) of https://arxiv.org/pdf/2102.03661.
    """
    if spec == 'TE': assert map_cal_arr2 is not None
    import copy
    cl_dic_with_cal = copy.deepcopy( cl_dic )
    for keyname in cl_dic_with_cal:
        for b1ind, b1 in enumerate( bands ):
            for b2ind, b2 in enumerate( bands ):
                if spec in ['TT', 'EE']:
                    map_cal1, map_cal2 = map_cal_arr[b1ind], map_cal_arr[b2ind]
                    map_cal = 1./ (map_cal1 * map_cal2)
                elif spec == 'TE':
                    map_calT1, map_calT2 = map_cal_arr[b1ind], map_cal_arr[b2ind]
                    map_calP1, map_calP2 = map_cal_arr2[b1ind], map_cal_arr2[b2ind]
                    map_cal = 0.5 * ( 
                        1./(map_calT1 * map_calP2) + 1./(map_calT2 * map_calP1 )
                    )
                cl_dic_with_cal[keyname][(b1, b2)] = cl_dic[keyname][(b1, b2)] * map_cal
    return cl_dic_with_cal

def create_copies_of_cl_in_multiple_bands(cl, bands, keyname = 'TT', map_cal_arr = None):
    cl_dic = {}
    if keyname is None: keyname = 'TT'
    cl_dic[keyname] = {}
    for b1ind, b1 in enumerate( bands ):
        for b2ind, b2 in enumerate( bands ):
            cl_dic[keyname][(b1, b2)] = np.copy( cl )
    return cl_dic

def get_ilc_residual_using_weights(cl_dic, wl, bands, wl2 = None, lmax = 10000, el = None):

    """
    get ILC residuals for a given compnent given the freqeuency dependent weights.
    If wl2 is None, then this function returns the auto-ILC residuals.

    Parameters
    ----------
    cl_dict : dict
        dictionary containing signal auto- and cross- spectra of the component in 
        different freq. bands.

    wl: array
        freqeuency dependent weights.

    bands: array
        array of frequency bands.

    wl2: array
        same as wl but for a different ILC.
        Default is None.

    lmax: int
        Maximum multipole for computation.
        Default is 10000.

    el: array
        Multipole array. 
        Default is None.

    Returns
    -------
    avec : array
        freq. dependene of the respective sky component.
        for example: CMB will be  [1., 1., ...., 1.] in all bands.
    """

    if wl2 is None:
        wl2 = wl
    if el is None:
        el = np.arange(lmax)
    res_ilc = []
    for elcnt, currel in enumerate(el):
        clmat = np.mat( create_covariance(bands, elcnt, cl_dic) )
        currw_ilc1 = np.mat( wl[:, elcnt] )
        currw_ilc2 = np.mat( wl2[:, elcnt] )
        curr_res_ilc = np.asarray(np.dot(currw_ilc1, np.dot(clmat, currw_ilc2.T)))[0][0]
        res_ilc.append( curr_res_ilc )

    res_ilc = np.asarray(res_ilc)
    res_ilc[np.isnan(res_ilc)] = 0.
    res_ilc[np.isinf(res_ilc)] = 0.

    return res_ilc
