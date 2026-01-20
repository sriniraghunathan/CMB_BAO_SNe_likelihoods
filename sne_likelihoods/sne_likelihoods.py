#from typing import Optional, Sequence
#from cobaya.conventions import packages_path_input
#from cobaya.likelihoods.base_classes import InstallableLikelihood
from cobaya.likelihoods.sn.pantheonplus import PantheonPlus

class LSSTY3_mock(PantheonPlus):#, InstallableLikelihood):
    """
    Likelihood for LSSTY3 type Ia supernovae sample.
    Reference
    ---------
    https://arxiv.org/abs/2401.02929
    """

    """    
    def initialize(self):
        pass
    """    
    def configure(self):
        self.pre_vars = self.mag_err ** 2
    
    def _read_data_file(self, data_file):
        file_cols = ["zhd", "zhel", "mu", "muerr_final"]
        self.cols = ["zcmb", "zhel", "mag", "mag_err"]
        self._read_cols(data_file, file_cols, sep=",")

    """
    def _read_data_file(self):
        file_cols = ["zhd", "zhel", "mu", "muerr_final"]
        self.cols = ["zcmb", "zhel", "mag", "mag_err"]
        self._read_cols(self.data_file, file_cols, sep=",")
    """

class lssty3_sne_mock(LSSTY3_mock):
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock.
    """

class lssty3_sne_mock_binned(LSSTY3_mock):
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_zmax1_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_samplebasedonzfromdes_rsval1_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_samplebasedonzfromdes_rsval2_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_samplebasedonzfromdes_rsval3_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_samplebasedonzfromdes_rsval4_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_samplebasedonzfromdes_rsval5_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """    

class lssty3_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_moddatavector_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_binned_moddatavector_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_moddatavector_zmax1_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_moddatavector_samplebasedonzfromdes_rsval1_sim1(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_sim2(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_sim3(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_sim4(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class lssty3_sim5(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """        

class desy5_sim0(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """

class desy5_moddatavector_sim0(LSSTY3_mock):    
    """
    Likelihood for LSSTY3 type Ia supernovae sample mock binned.
    """        