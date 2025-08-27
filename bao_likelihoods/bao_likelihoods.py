from cobaya.likelihoods.base_classes import BAO

class desi_2024_base(BAO):
    _is_abstract = True
    bibtex_file = 'desi_2024_bao.bibtex'

class desidr2bao_mock(desi_2024_base):
    r"""
    DESI BAO likelihood for all tracers.
    """

class desidr3bao_mock(desi_2024_base):
    r"""
    DESI BAO likelihood for all tracers.
    """