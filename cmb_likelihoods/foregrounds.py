import numpy as np, sys, os, glob
from scipy.io import readsav
from scipy import interpolate as intrp
import scipy as sc
from scipy import integrate, stats

h, k_B, c=6.62607004e-34, 1.38064852e-23, 2.99792458e8

def get_foregrounds(els, freqarr, params_values):
    cl_dic_fg = {'TT': {}}

    for f1ind, freq1 in enumerate( freqarr ):
        for f2ind, freq2 in enumerate( freqarr ):

            #tSZ
            Atsz = eval( 'params_values.get(f"Atsz")' )
            alphatsz = eval( 'params_values.get(f"alphatsz")' )
            cl_tsz = get_cl_tsz_battaglia(els, Atsz, alphatsz, freq1, freq2)

            #kSZ
            Aksz = eval( 'params_values.get(f"Aksz")' )
            cl_ksz = get_cl_ksz(els, Aksz)

            #radio
            Aradio = eval( 'params_values.get(f"Aradio")' )
            alpharadio = eval( 'params_values.get(f"alpharadio")' )
            cl_radio = get_cl_radio(els, Aradio, alpharadio, freq1, freq2)

            #CIB
            Acibpo = eval( 'params_values.get(f"Acibpo")' )
            betacibpo = eval( 'params_values.get(f"betacibpo")' )
            Tcibpo = eval( 'params_values.get(f"Tcibpo")' )
            Acibclus = eval( 'params_values.get(f"Acibclus")' )
            alphacibclus = eval( 'params_values.get(f"alphacibclus")' )
            betacibclus = eval( 'params_values.get(f"betacibclus")' )
            Tcibclus = eval( 'params_values.get(f"Tcibclus")' )
            cl_cib = get_cl_cib(els, Acibpo, betacibpo, Acibclus, betacibclus, freq1, freq2, alphacibclus = alphacibclus, Tcibpo = Tcibpo, Tcibclus = Tcibclus)

            cl_fg_all = cl_tsz + cl_ksz + cl_radio + cl_cib
            cl_dic_fg['TT'][(freq1, freq2)] = cl_fg_all

    return cl_dic_fg

def get_dB_dT(nu, nu0 = None, temp = 2.725):
    if nu<1e4: nu *= 1e9

    x=h*nu/(k_B*temp)
    dBdT = x**4. * np.exp(x) / (np.exp(x)-1)**2.

    if nu0 is not None:
        nu0 *= 1e9
        x0=h*nu0/(k_B*temp)
        dBdT0 = x0**4 * np.exp(x0) / (np.exp(x0)-1)**2.
        return  dBdT / dbdT0
    else:
        return dBdT

def get_BnuT(nu, temp = 2.725):
    if nu<1e4: nu *= 1e9
    x=h*nu/(k_B*temp)

    t1 = 2 * h * nu**3./ c**2.
    t2 = 1./ (np.exp(x)-1.)

    return t1 * t2

def coth(x):
    return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x))

def compton_y_to_delta_Tcmb(freq1, freq2 = None, Tcmb = 2.73):

    """ad
    c.f:  table 1, sec. 3 of arXiv: 1303.5081; 
    table 8 of http://arxiv.org/pdf/1303.5070.pdf
    no relativistic corrections included.
    freq1, freq2 = frequencies in GHz to cover the bandpass
    freq2 = None will force freq1 to be the centre frequency
    """

    if freq1<1e4: freq1 = freq1 * 1e9

    if not freq2 is None:
        if freq2<1e4: freq2 = freq2 * 1e9
        freq = np.arange(freq1,freq2,delta_nu)
    else:
        freq = np.asarray([freq1])

    x = (h * freq) / (k_B * Tcmb)
    g_nu = x * coth(x/2.) - 4.

    return Tcmb * np.mean(g_nu)

def get_cl_tsz_cib(rho_tsz_cib, freq1, freq2, cl_tsz_freq1_freq1, cl_tsz_freq2_freq2, cl_cib_freq1_freq1, cl_cib_freq2_freq2, fg_model = 'george15'):

    if freq1 >= 217 and freq2 >=217:
        rho_tsz_cib = rho_tsz_cib * -1.

    cl_tsz_cib = -rho_tsz_cib * ( np.sqrt(cl_tsz_freq1_freq1 * cl_cib_freq2_freq2) + np.sqrt(cl_tsz_freq2_freq2 * cl_cib_freq1_freq1) )

    return cl_tsz_cib

def get_cl_radio(els, Aradio, alpharadio, freq1, freq2, freq0 = 150, null_highfreq_radio = True, el_norm = 3000):

    #dls_fac
    dl_fac = els * (els+1)/2/np.pi

    nr = ( get_dB_dT(freq0) )**2.
    dr = get_dB_dT(freq1) * get_dB_dT(freq2)

    epsilon_nu1_nu2 = nr/dr

    dl_rg = Aradio * epsilon_nu1_nu2 * (1.*freq1 * freq2/freq0/freq0)**alpharadio * (els*1./el_norm)**2

    cl_rg = dl_rg / dl_fac

    cl_rg[np.isnan(cl_rg)] = 0.

    if null_highfreq_radio and (freq1>230 or freq2>230):
        cl_rg *= 0.
    cl_rg[np.isnan(cl_rg)] = 0.
    cl_rg[np.isinf(cl_rg)] = 0.

    return cl_rg

def get_cl_tsz_battaglia(els, Atsz, alphatsz, freq1, freq2, freq0 = 148, el_norm = 3000.):

    dl_fac = els * (els+1)/2/np.pi

    el_, dlyy_battaglia = np.loadtxt('data/cmb_data/foregrounds/dl_tsz_148_batt.dat', unpack = True)
    dlyy_battaglia = np.interp(els, el_, dlyy_battaglia)
    
    tsz_fac_freq0 = compton_y_to_delta_Tcmb(148*1e9)
    tsz_fac_freq1 = compton_y_to_delta_Tcmb(freq1*1e9)
    tsz_fac_freq2 = compton_y_to_delta_Tcmb(freq2*1e9)

    scalefac = tsz_fac_freq1 * tsz_fac_freq2 / tsz_fac_freq0**2.

    dl_tsz = Atsz * dlyy_battaglia * scalefac * (els*1./el_norm)**alphatsz
    cl_tsz = dl_tsz/dl_fac
    cl_tsz[np.isnan(cl_tsz)] = 0.
    cl_tsz[np.isinf(cl_tsz)] = 0.

    return cl_tsz

def get_cl_ksz(els, Aksz):

    dl_fac = els * (els+1)/2/np.pi
    dl_ksz = np.tile( Aksz, len(els) )
    cl_ksz = dl_ksz / dl_fac
    cl_ksz[np.isnan(cl_ksz)] = 0.
    cl_ksz[np.isinf(cl_ksz)] = 0.

    return cl_ksz

def get_cl_cib(els, Acibpo, betacibpo, Acibclus, betacibclus, freq1, freq2, alphacibclus = 0.8, Tcibpo = 20., Tcibclus = 20., freq0 = 150, el_norm = 3000):    

    #conert to Dls
    dl_fac = els * (els+1)/2/np.pi

    nr = ( get_dB_dT(freq0) )**2.
    dr = get_dB_dT(freq1) * get_dB_dT(freq2)

    epsilon_nu1_nu2 = nr/dr

    def get_cib_eta_terms(Tcib, betacib):
        bnu1 = get_BnuT(freq1, temp = Tcib)
        bnu2 = get_BnuT(freq2, temp = Tcib)
        bnu0 = get_BnuT(freq0, temp = Tcib)

        etanu1 = ((1.*freq1*1e9)**betacib) * bnu1
        etanu2 = ((1.*freq2*1e9)**betacib) * bnu2
        etanu0 = ((1.*freq0*1e9)**betacib) * bnu0

        return etanu0, etanu1, etanu2

    etanu0_dg_po, etanu1_dg_po, etanu2_dg_po = get_cib_eta_terms(Tcibpo, betacibpo)
    etanu0_dg_clus, etanu1_dg_clus, etanu2_dg_clus = get_cib_eta_terms(Tcibclus, betacibclus)

    dl_dg_po = Acibpo * epsilon_nu1_nu2 * (1.*etanu1_dg_po * etanu2_dg_po/etanu0_dg_po/etanu0_dg_po) * (els*1./el_norm)**2
    dl_dg_clus = Acibclus * epsilon_nu1_nu2 * (1.*etanu1_dg_clus * etanu2_dg_clus/etanu0_dg_clus/etanu0_dg_clus) * (els*1./el_norm)**alphacibclus

    ###from IPython import embed; embed()

    cl_dg_po = dl_dg_po / dl_fac
    cl_dg_clus = dl_dg_clus / dl_fac

    cl_dg_po[np.isnan(cl_dg_po)] = 0.
    cl_dg_po[np.isinf(cl_dg_po)] = 0.
    cl_dg_clus[np.isnan(cl_dg_clus)] = 0.
    cl_dg_clus[np.isinf(cl_dg_clus)] = 0.

    cl_cib = cl_dg_po + cl_dg_clus

    return cl_cib

