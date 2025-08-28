import numpy as np
from pylab import *

fd = 'data/bao_data/desi_bao_dr3_mock/'
cov_lowz = np.loadtxt('%s/Tab7_covariance_matrix_baorsd_zbin_0p0_2p1.ecsv' %(fd))
cov_highz = np.loadtxt('%s/Tab7_covariance_matrix_bao_lya_zbin_1p9_3p7.ecsv' %(fd))

lowz_fields = ['f_sigma8', 'DA_over_rs', 'Hz_rs']
highz_fields = ['DA_over_rs', 'Hz_rs']

errors_lowz = np.diag(cov_lowz)**0.5
errors_highz = np.diag(cov_highz)**0.5

reclen_lowz = len(errors_lowz)
reclen_highz = len(errors_highz)

total_fields_lowz = len( lowz_fields )
total_fields_highz = len( highz_fields )


lowz_da_over_rs_inds = np.arange(0, reclen_lowz, total_fields_lowz) + 1
highz_da_over_rs_inds = np.arange(0, reclen_highz, total_fields_highz)

lowz_hz_rs_inds = np.arange(0, reclen_lowz, total_fields_lowz) + 2
highz_hz_rs_inds = np.arange(0, reclen_highz, total_fields_highz) + 1


#print( errors_lowz[lowz_da_over_rs_inds]**2 )
#print( errors_highz[highz_da_over_rs_inds]**2 )

print( errors_lowz[lowz_hz_rs_inds]**2 )
print( errors_highz[highz_hz_rs_inds]**2 )