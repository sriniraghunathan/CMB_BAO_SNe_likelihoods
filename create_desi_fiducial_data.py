import numpy as np, os, sys
sys.path.append( 'modules' )
import misc, sne_cmb_fisher_tools
from astropy.cosmology import FlatLambdaCDM
from astropy import constants as const
from astropy import units as u

#which_bao_data = 'desi_dr2'
#which_bao_data = 'desi_dr3'
#which_bao_data = 'desi_dr3_lowz'
which_bao_data = 'desi_dr3_highz'

ignore_first_entry_for_dr3_highz = True ##False ##True
if which_bao_data in ['desi_dr3', 'desi_dr3_lowz', 'desi_dr3_highz']:
    camb_or_astropy = 'camb'
else:
    camb_or_astropy = 'camb'
    #camb_or_astropy = 'astropy'

paramfile = 'data/params_cobaya.ini'
param_dict = misc.get_param_dict(paramfile)
if (0):
    param_dict['ws'] = -0.888
    param_dict['wa'] = -0.17
    param_dict['omch2'] = 0.298


#pars, camb_results = sne_cmb_fisher_tools.set_camb(param_dict, lmax = 100)
if camb_or_astropy == 'astropy':
    baselinecosmo = 'Flatw0waCDM'
    cosmo = sne_cmb_fisher_tools.set_cosmo(param_dict, baselinecosmo = baselinecosmo)
    camb_results = None
elif camb_or_astropy == 'camb':
    camb_results = None
    #pars, camb_results = sne_cmb_fisher_tools.set_camb(param_dict, lmax = 10, WantTransfer = True)
    cosmo = None
#sys.exit()

if which_bao_data == 'desi_dr2':
    bao_data_fd = 'data/bao_data/desi_bao_dr2'
    bao_data_fname = '%s/desi_gaussian_bao_ALL_GCcomb_mean.txt' %(bao_data_fd)
    bao_data_cov_fname = '%s/bao_data/desi_gaussian_bao_ALL_GCcomb_cov.txt' %(bao_data_fd)
    bao_data_opfd = 'data/bao_data/desi_bao_dr2'
elif which_bao_data == 'desi_dr3_highz':
    bao_data_fd = 'data/bao_data/desi_bao_dr3_stuffs/'
    bao_data_fname = '%s/Tab7_data_vector_bao_lya_zbin_1p9_3p7.ecsv' %(bao_data_fd)
    bao_data_cov_fname = '%s/Tab7_covariance_matrix_bao_lya_zbin_1p9_3p7.ecsv' %(bao_data_fd)
    bao_data_opfd = 'data/bao_data/desi_bao_dr3_mock'
elif which_bao_data == 'desi_dr3_lowz':
    bao_data_fd = 'data/bao_data/desi_bao_dr3_stuffs/'
    bao_data_fname = '%s/Tab7_data_vector_baorsd_zbin_0p0_2p1.ecsv' %(bao_data_fd)
    bao_data_cov_fname = '%s/Tab7_covariance_matrix_baorsd_zbin_0p0_2p1.ecsv' %(bao_data_fd)
    bao_data_opfd = 'data/bao_data/desi_bao_dr3_mock'
bao_data_cov_fname_op = bao_data_cov_fname.replace( bao_data_fd, bao_data_opfd )

def bao_model(param_dict, z_arr, observable_arr, cosmo = None, camb_results = None):
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

    if camb_or_astropy == 'camb' and camb_results is None:
        pars, camb_results = sne_cmb_fisher_tools.set_camb(param_dict, lmax = 10, WantTransfer = True, z_arr_for_matter_power = z_arr)

    #compute the models
    ombh2 = param_dict['ombh2']
    ommh2 = param_dict['omch2'] + param_dict['ombh2']
    neff = param_dict['neff']
    model_arr =[]
    for zcntr, (z, o) in enumerate( zip( z_arr, observable_arr ) ):
        ##print(z, o)

        if camb_or_astropy == 'astropy':
            curr_DM = cosmo.comoving_distance(z).value #Transverse comoving distance
            curr_DH = ( constants.c.to('km/s')/cosmo.H(z) ).value #Hubble distance
        elif camb_or_astropy == 'camb':
            curr_DA = camb_results.angular_diameter_distance(z) #Angular diameter distance
            curr_DM = curr_DA * (1+z) #Transverse comoving distance
            curr_DH = ( constants.c.to('km/s')/camb_results.hubble_parameter(z) ).value #Hubble distance
            curr_Hs = camb_results.hubble_parameter(z)#, units="km/s/Mpc")
            #from IPython import embed; embed()

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
        elif o == 'DA_over_rs':
            curr_model = curr_DA / curr_rd
        elif o == 'Hz_rs':
            curr_model = curr_Hs * curr_rd
        elif o == "f_sigma8":
            curr_model = camb_results.get_fsigma8()[zcntr]

        model_arr.append( curr_model )
    model_arr = np.asarray( model_arr )

    return model_arr



if not os.path.exists( bao_data_opfd ): os.system('mkdir -p %s' %(bao_data_opfd))
opfname = bao_data_fname.replace(bao_data_fd, bao_data_opfd)
opfname = opfname.replace('.txt', '_%s.txt' %(camb_or_astropy))

#create a new mock file for LCDM numbers, and push the original cov in there.

bao_rec = np.recfromtxt(bao_data_fname)

if which_bao_data == 'desi_dr2':
    opf = open( opfname, 'w' )
    opline = '# [z] [value at z] [quantity]'
    opf.writelines( '%s\n' %(opline) )
    #op_arr = []
    for curr_rec in bao_rec:
        curr_rec = list( curr_rec )
        curr_z, curr_val, curr_obs = curr_rec
        curr_obs = curr_obs.decode("utf-8")
        ##print(curr_z, curr_val, curr_obs); sys.exit()
        curr_model_val = bao_model(param_dict, [curr_z], [curr_obs], cosmo = cosmo, camb_results = camb_results)[0]
        print( curr_z, curr_val, curr_obs, curr_model_val )
        #base_classes.bao.BAO.theory_fun( curr_z, curr_obs )
        #theory_fun( curr_z, curr_obs )
        #op_arr.append( [curr_z, curr_model_val, curr_obs] )
        opline = '%s %s %s' %(curr_z, curr_model_val, curr_obs)
        opf.writelines( '%s\n' %(opline) )
elif which_bao_data in ['desi_dr3', 'desi_dr3_lowz', 'desi_dr3_highz']:
    opf = open( opfname, 'w' )
    opline = '# [z] [value at z] [quantity]'
    opf.writelines( '%s\n' %(opline) )
    if which_bao_data  == 'desi_dr3_highz':
        curr_obs_arr = ['DA_over_rs', 'Hz_rs']
    elif which_bao_data  == 'desi_dr3_lowz':
        curr_obs_arr = ['f_sigma8', 'DA_over_rs', 'Hz_rs']
    for recntr, curr_rec in enumerate( bao_rec ):
        curr_rec = list( curr_rec )
        if which_bao_data  == 'desi_dr3_highz':
            curr_z, curr_val1, curr_val2 = curr_rec
            curr_z = curr_z.decode("utf-8")
            curr_val1 = curr_val1.decode("utf-8")
            curr_val2 = curr_val2.decode("utf-8")
        elif which_bao_data  == 'desi_dr3_lowz':
            curr_z, curr_val1, curr_val2, curr_val3 = curr_rec
            curr_z = curr_z.decode("utf-8")
            curr_val1 = curr_val1.decode("utf-8")
            curr_val2 = curr_val2.decode("utf-8")
            curr_val3 = curr_val3.decode("utf-8")

        if which_bao_data  == 'desi_dr3_highz' and ignore_first_entry_for_dr3_highz and recntr == 1:
            print(curr_rec, 'hi')
            continue

        if curr_z == 'z': continue
        curr_z = float( curr_z )
        curr_val1 = float( curr_val1 )
        curr_val2 = float( curr_val2 )
        curr_val_arr = [curr_val1, curr_val2]
        if which_bao_data  == 'desi_dr3_lowz':
            curr_val3 = float( curr_val3 )
            curr_val_arr = [curr_val1, curr_val2, curr_val3]

        if which_bao_data  == 'desi_dr3_highz': #swap indices now.
            curr_obs_arr = curr_obs_arr[::-1]
            curr_val_arr = curr_val_arr[::-1]
        for (curr_obs, curr_val) in zip(curr_obs_arr, curr_val_arr):
            curr_model_val = bao_model(param_dict, [curr_z], [curr_obs], cosmo = cosmo, camb_results = camb_results)[0]
            opline = '%s %s %s' %(curr_z, curr_model_val, curr_obs)
            print( curr_z, curr_val, curr_obs, curr_model_val )
            opf.writelines( '%s\n' %(opline) )


#op_arr = np.asarray( op_arr )

###np.savetxt( opfname, bao_rec, fmt = '', header = "# [z] [value at z] [quantity]")
#push the cov now
if which_bao_data == 'desi_dr2':
    cmd = 'cp %s %s' %(bao_data_cov_fname, bao_data_cov_fname_op)
    os.system( cmd )
elif which_bao_data in ['desi_dr3', 'desi_dr3_lowz', 'desi_dr3_highz']:
    f = open(bao_data_cov_fname, 'r');
    cov_arr = []
    for lcntr, line in enumerate( f ):
        if line.find('#')>-1 or line.find('covariance')>-1: 
            prevlcntr = lcntr
            continue
        tmp = line.strip().replace('[','').replace(']','').split(',')
        print(tmp)
        cov_arr.append(tmp)

    cov_arr = np.asarray( cov_arr )
    cov_arr = cov_arr.astype(np.float)

    '''
    if which_bao_data  == 'desi_dr3_highz': #swap indices now.
        cov_diag = np.diag(cov_arr)
        cov_diag_rolled = np.roll(cov_diag, 1)
        #sys.exit()
        odd_inds = np.arange(1,len(cov_arr), 2)
        even_inds = np.arange(0,len(cov_arr), 2)
        cov_arr_odd_inds = cov_diag[odd_inds]
        cov_arr_even_inds = cov_diag[even_inds]
        cov_diag_mod = []
        for (o,e) in zip(cov_arr_odd_inds, cov_arr_even_inds):
            cov_diag_mod.append( o )
            cov_diag_mod.append( e )
        cov_diag_mod = np.asarray( cov_diag_mod )
        
        cov_off_diag_low = np.tril(cov_arr, k=-1)       
        cov_off_diag_high = np.triu(cov_arr, k=1)
        cov_arr_mod = np.eye( len(cov_arr) ) * cov_diag_mod
        cov_arr_mod = cov_arr_mod + cov_off_diag_low + cov_off_diag_high
        cov_arr = np.copy( cov_arr_mod )

        """
        cov_arr_odd_inds = cov_arr[odd_inds]
        cov_arr_even_inds = cov_arr[even_inds]
        cov_arr_mod = []
        for (o,e) in zip(cov_arr_odd_inds, cov_arr_even_inds):
            cov_arr_mod.append( o )
            cov_arr_mod.append( e )
        cov_arr_mod = np.asarray( cov_arr_mod )
        cov_arr = cov_arr_mod
        """
    '''


    if (0):##which_bao_data  == 'desi_dr3_highz' and ignore_first_entry_for_dr3_highz:
        cov_arr = cov_arr[2:, 2:]
        #print(cov_arr.shape)

    np.savetxt(bao_data_cov_fname_op, cov_arr, fmt = '%g')  

print(bao_data_cov_fname_op)
print('Done')
