import numpy as np, os, sys
sys.path.append('modules/')
import sne_cmb_fisher_tools, misc

zmin, zmax = -1, -1
sample_based_on_z_from = None
if (1):
    sne_exp = 'lsst_unbinned'
    sim_no_arr = [1]##, 2, 3, 4, 5]
    if (0): #removing high-z
        zmax = 1.
    if (1): #sample based on DES
        sample_based_on_z_from = 'des'

        if sample_based_on_z_from == 'des':
            des_sne_details_dic = sne_cmb_fisher_tools.get_sne_details('des', 0, unbinned_sne_sim_no = 0, obtain_covs = False)
            z_underlying = des_sne_details_dic['sne_zarr']
            rsval = 1

if (0): #LSST binned
    sne_exp = 'lsst_binned'
    sim_no_arr = [1]
if (0): #For Rick and Ayan
    #sne_exp = 'lsst_v2_unbinned'
    sne_exp = 'lsst_v2_binned'
    sim_no_arr = [1]
if (0):
    sne_exp = 'des'
    sim_no_arr = [0]

if sample_based_on_z_from is None:
    z_underlying = None
    rsval = -1

cobaya_fd_spt = '/home/sri/.local/lib/python3.9/site-packages/cobaya/likelihoods/sn/'
cobaya_data_fd_spt = '/home/sri/cobaya/desi/data/sn_data/'
if os.path.exists( cobaya_fd_spt ): #local
    cobaya_fd = cobaya_fd_spt
    cobaya_data_fd = cobaya_data_fd_spt
    machine_name = 'spt'
else:
    cobaya_fd = '/usr/local/lib/python3.9/site-packages/cobaya/likelihoods/sn/'
    cobaya_data_fd = '/Users/sraghunathan/Research/cobaya/desi/data/sn_data/'
    machine_name = 'local'


for sim_no in sim_no_arr:
    print('\n\nSim = %s' %(sim_no))
    add_stat_error = 0
    sne_details_dic = sne_cmb_fisher_tools.get_sne_details(sne_exp, add_stat_error, unbinned_sne_sim_no = sim_no, obtain_covs = False, zmin = zmin, zmax = zmax, underlying_zdist_for_sampling = z_underlying, rsval = rsval)
    print(sne_details_dic.keys())

    if sne_exp == 'lsst_unbinned':
        exp_dr_str = 'LSSTY3'
        op_fd = '%s/%s_sim%s/' %(cobaya_data_fd, exp_dr_str, sim_no)
        #opfname_suff = '%s_SN_sim%s.csv' %(exp_dr_str, sim_no)
    elif sne_exp == 'lsst_binned':
        exp_dr_str = 'LSSTY3_binned'
        op_fd = '%s/%s_sim%s/' %(cobaya_data_fd, exp_dr_str, sim_no)
        #opfname_suff = '%s_SN_sim%s.csv' %(exp_dr_str, sim_no)
    elif sne_exp == 'lsst_v2_unbinned':
        exp_dr_str = 'LSSTY3_v2'
        op_fd = '%s/%s_sim%s/' %(cobaya_data_fd, exp_dr_str, sim_no)
        #opfname_suff = '%s_SN_sim%s.csv' %(exp_dr_str, sim_no)
    elif sne_exp == 'lsst_v2_binned':
        exp_dr_str = 'LSSTY3_v2_binned'
        op_fd = '%s/%s_sim%s/' %(cobaya_data_fd, exp_dr_str, sim_no)
        #opfname_suff = '%s_SN_sim%s.csv' %(exp_dr_str, sim_no)
    elif sne_exp == 'des':
        exp_dr_str = 'DESY5'
        op_fd = '%s/%s_sim/' %(cobaya_data_fd, exp_dr_str)
        #opfname_suff = '%s_SN_sim.csv' %(exp_dr_str)


    op_fd = op_fd.strip('/')
    if zmin != -1:
        op_fd = '%s_zim%g' %(op_fd, zmin)
        exp_dr_str = '%s_zim%g' %(exp_dr_str, zmin)
    if zmax != -1:
        op_fd = '%s_zmax%g' %(op_fd, zmax)
        exp_dr_str = '%s_zmax%g' %(exp_dr_str, zmax)
    if sample_based_on_z_from is not None:
        op_fd = '%s_samplebasedonzfrom%s_rsval%s' %(op_fd, sample_based_on_z_from, rsval)
        exp_dr_str = '%s_samplebasedonzfrom%s_rsval%s' %(exp_dr_str, sample_based_on_z_from, rsval)

    if sne_exp == 'des':
        opfname_suff = '%s_SN_sim.csv' %(exp_dr_str)
    else:
        opfname_suff = '%s_SN_sim%s.csv' %(exp_dr_str, sim_no)

    if not os.path.exists(op_fd): os.system('mkdir -p %s' %(op_fd))

    opfname = '%s/%s' %(op_fd, opfname_suff)
    print(opfname, exp_dr_str); ###sys.exit()
    header = 'CID,IDSURVEY,zCMB,zHD,zHEL,MU,MUERR_FINAL'
    total_sne = len(sne_details_dic['sne_zarr'])
    sne_id_arr = np.arange( total_sne )
    #note that sne_muerr_stat is actually stat + sys
    op_arr = np.column_stack( (sne_id_arr, sne_id_arr, sne_details_dic['sne_zarr'], sne_details_dic['sne_zarr'], sne_details_dic['sne_zarr'], sne_details_dic['sne_muarr'], sne_details_dic['sne_muerr_stat']) ) 
    np.savetxt( opfname, op_arr, header = header, delimiter = ',', fmt='%d,%d,%g,%g,%g,%g,%g')
    print(opfname)

    #------------------------------
    #now write config.dataset
    config_opfname = '%s/config.dataset' %(op_fd)
    print(config_opfname)
    ##sys.exit()
    line_arr = ['name = %s_sim%s' %(exp_dr_str, sim_no), 
                          'data_file = %s_sim%s/%s_SN_sim%s.csv' %(exp_dr_str, sim_no, exp_dr_str, sim_no), 
                          'mag_covmat_file = %s_sim%s/covsys_000.txt' %(exp_dr_str, sim_no)]

    opf = open( config_opfname, 'w' )
    for line in line_arr:
        opf.writelines('%s\n' %(line))
    opf.close()
    #------------------------------

    #------------------------------
    #now write cov
    cov_opfname = '%s/covsys_000.txt' %(op_fd)
    print(cov_opfname); ##sys.exit()
    if (1):###not os.path.exists(cov_opfname):
        sne_details_dic = sne_cmb_fisher_tools.get_sne_details(sne_exp, add_stat_error, unbinned_sne_sim_no = sim_no, obtain_covs = True, reqd_cov_tags = [0], zmin = zmin, zmax = zmax, underlying_zdist_for_sampling = z_underlying, rsval = rsval)
        sne_cov = sne_details_dic['sne_tot_cov_dic'][0]
        print(sne_details_dic['sne_zarr'].shape, sne_cov.shape); sys.exit()
        op_arr = np.concatenate( ([total_sne], sne_cov.ravel()))
        np.savetxt( cov_opfname, op_arr, fmt = '%g' )
    #------------------------------

    #------------------------------
    #xxx.yaml in /usr/local/lib/python3.9/site-packages/cobaya/likelihoods/sn
    line_arr = ['# Settings for the LSSTY3 SN Ia analysis.', 
                '# Path to the data: where the sn_data has been cloned',
                'path: null', 
                '# .dataset file with settings', 
                'dataset_file: %s_sim%s/config.dataset' %(exp_dr_str, sim_no),
                '# Overriding of .dataset parameters',
                'dataset_params:',
                '# field: value',
                '# Aliases for automatic covariance matrix',
                'aliases: [%s_sim%s]' %(exp_dr_str, sim_no),
                '# Use absolute magnitude',
                'use_abs_mag: False',
                '# Speed in evaluations/second',
                'speed: 100',
                ]
    yaml_opfname = '%s/%s_sim%s.yaml' %(cobaya_fd, exp_dr_str.lower(), sim_no)
    print(yaml_opfname)
    opf = open( yaml_opfname, 'w' )
    for line in line_arr:
        opf.writelines('%s\n' %(line))
    opf.close()
    #------------------------------

    #------------------------------
    #xxx.py in /usr/local/lib/python3.9/site-packages/cobaya/likelihoods/sn
    line_arr = ['from .pantheonplus import PantheonPlus',
                'class %s_sim%s(PantheonPlus):' %(exp_dr_str, sim_no),
                '\t"""',
                '\tLikelihood for %s type Ia supernovae sample.' %(exp_dr_str),
                '\tReference',
                '\t---------',
                '\thttps://arxiv.org/abs/2401.02929',
                '\t"""',
                '\tdef configure(self):',
                '\t\tself.pre_vars = self.mag_err ** 2',

                '\tdef _read_data_file(self, data_file):',
                '\t\tfile_cols = ["zhd", "zhel", "mu", "muerr_final"]',
                '\t\tself.cols = ["zcmb", "zhel", "mag", "mag_err"]',
                '\t\tself._read_cols(data_file, file_cols, sep=",")'
                ]
    py_opfname = '%s/%s_sim%s.py' %(cobaya_fd, exp_dr_str.lower(), sim_no)
    print(py_opfname)
    opf = open( py_opfname, 'w' )
    for line in line_arr:
        opf.writelines('%s\n' %(line))
    opf.close()
    #------------------------------

    #------------------------------
    if machine_name == 'local': #transfer to spt
        #data folder
        #cmd = 'rsync -trvz %s spt:%s' %(op_fd.strip('/'), cobaya_data_fd_spt)
        cmd = 'rsync -trvz %s spt:%s' %(op_fd, cobaya_data_fd_spt)
        print(cmd); ###sys.exit()
        os.system(cmd)

        #yaml and py files
        cmd = 'rsync -trvz %s spt:%s/' %(yaml_opfname, cobaya_fd_spt)
        print(cmd)
        os.system(cmd)

        cmd = 'rsync -trvz %s spt:%s/' %(py_opfname, cobaya_fd_spt)
        print(cmd)
        os.system(cmd)

    #------------------------------
    #diagonal cov
    if (0):##sne_exp == 'lsst_unbinned' and sim_no == 1: #make the LSST cov diagonal

        op_fd_mod = op_fd.replace('LSSTY3_sim%s' %(sim_no), 'LSSTY3_sim%s_diagcov' %(sim_no))
        cmd = 'rsync -trvz %s %s' %(op_fd, op_fd_mod)
        print( cmd )
        os.system( cmd )

        cov_fname = '%s/covsys_000.txt' %(op_fd_mod)
        sne_cov = np.loadtxt(cov_fname, skiprows = 1).reshape(total_sne, total_sne)
        sne_cov_mod = np.diag( np.diag( sne_cov ) )
        op_arr = np.concatenate( ([total_sne], sne_cov_mod.ravel()))
        np.savetxt( cov_fname, op_arr, fmt = '%g' )

print('Done')
sys.exit()
