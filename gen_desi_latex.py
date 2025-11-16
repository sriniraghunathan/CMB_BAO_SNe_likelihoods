def corr_from_cov(covmat):
    diags = np.sqrt(np.diag(covmat))
    corrmat = np.zeros_like(covmat)
    for i in range(covmat.shape[0]):
        for j in range(covmat.shape[0]):
            corrmat[i, j] = covmat[i, j] / (diags[i] *  diags[j])
    return corrmat

import numpy as np, os, sys

data_vector_fname = 'data/bao_data/desi_bao_dr3_mock/Tab7_data_vector_bao_lya_zbin_1p9_3p7.ecsv'
cov_fname = 'data/bao_data/desi_bao_dr3_mock/Tab7_covariance_matrix_bao_lya_zbin_1p9_3p7.ecsv'

cov = np.loadtxt( cov_fname )
corr = corr_from_cov( cov )
#print(corr)
z_all, data_vector_all = np.loadtxt( data_vector_fname, usecols = [0, 1], unpack = True )
obs_all = np.loadtxt( data_vector_fname, usecols = [2], dtype = 'str' )
print(z_all.shape, data_vector_all.shape, cov.shape)

data_vector_dic = {}
ind_dic = {}
for ind, (z, d, o) in enumerate( zip(z_all, data_vector_all, obs_all) ):
    if z not in data_vector_dic:
        data_vector_dic[z] = []
        ind_dic[z] = []
    ind_dic[z].append( ind )
    data_vector_dic[z].append( d )

corr_dic = {}
for z in ind_dic:
    curr_inds = np.asarray( ind_dic[z] )
    curr_corr = corr[curr_inds[:, None], curr_inds[None, :]]
    corr_dic[z] = [curr_corr[0,1]]
    #print(corr_dic[z]); #sys.exit()

    opline = '%.2f' %(z) + ' & ' + '%.2f & %.2f' %( tuple(data_vector_dic[z]) ) + ' & ' + '%.3f' %( tuple(corr_dic[z]) )
    opline = '%s \\\\\hline' %(opline)
    print(opline)


print('\n\n\n')

data_vector_fname = 'data/bao_data/desi_bao_dr3_mock/Tab7_data_vector_baorsd_zbin_0p0_2p1.ecsv'
cov_fname = 'data/bao_data/desi_bao_dr3_mock/Tab7_covariance_matrix_baorsd_zbin_0p0_2p1.ecsv'

cov = np.loadtxt( cov_fname )
corr = corr_from_cov( cov )
#print(corr)
z_all, data_vector_all = np.loadtxt( data_vector_fname, usecols = [0, 1], unpack = True )
obs_all = np.loadtxt( data_vector_fname, usecols = [2], dtype = 'str' )
print(z_all.shape, data_vector_all.shape, cov.shape)

data_vector_dic = {}
ind_dic = {}
for ind, (z, d, o) in enumerate( zip(z_all, data_vector_all, obs_all) ):
    if z not in data_vector_dic:
        data_vector_dic[z] = []
        ind_dic[z] = []
    ind_dic[z].append( ind )
    data_vector_dic[z].append( d )

corr_dic = {}
for z in ind_dic:
    curr_inds = np.asarray( ind_dic[z] )
    curr_corr = corr[curr_inds[:, None], curr_inds[None, :]]
    corr_dic[z] = [curr_corr[0,1], curr_corr[0,2], curr_corr[1,2]]
    #print(corr_dic[z]); #sys.exit()

    opline = '%.2f' %(z) + ' & ' + '%.2f & %.2f & %.2f' %( tuple(data_vector_dic[z]) ) + ' & ' + '%.3f & %.3f & %.3f' %( tuple(corr_dic[z]) )
    opline = '%s \\\\\hline' %(opline)
    print(opline)
sys.exit()


