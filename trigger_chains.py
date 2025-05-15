import numpy as np, os, sys

cmb_experiments = []
likelihood_name_dic = {'lssty3_sne_mock': 'sne_likelihoods', 
                      'desidr2bao_mock': ''}

if (1): #CMB, DESI-BAO, and LSST-SNe combinations
    experiment_arr = ['so_baseline']
    dataset_arr = ['TTEETE', 'TTEETEPP', 'TTEETEPP+desidr2bao_mock', 'TTEETEPP+lssty3_sne_mock', 'TTEETEPP+desidr2bao_mock+lssty3_sne_mock']

if (0): #BAO and SNe combinations.
    experiment_arr = ['dummy']
    dataset_arr = ['desidr2bao_mock', 'lssty3_sne_mock', 'desidr2bao_mock+lssty3_sne_mock']

comso_arr = ['lcdm', 'w0walcdm']
cmb_experiments_bands_dic = {'spt3g': [90, 150, 220], 
                 'spt3g_winter': [90, 150, 220], 
                 'spt3g_summer': [90, 150, 220], 
                 'spt3g_wide': [90, 150, 220], 
                 'so_baseline': [30, 40, 90, 150, 220, 280], 
                 'so_goal': [30, 40, 90, 150, 220, 280], 
                 's4_wide': [30, 40, 90, 150, 220, 280], 
                 }
spaceval = '  '
for experiment in experiment_arr:
    print('\nExperiment = %s' %(experiment))
    for dataset in dataset_arr:
        print('\tDataset = %s' %(dataset))
        for cosmo in comso_arr:
            print('\t\tCosmo = %s' %(cosmo))

            #----
            #gather yaml template for this cosmo
            yaml_cosmo_template_fname = 'yamls/templates/template_for_%ssampler.yaml' %(cosmo)
            #----

            #----
            #gather yaml template for calibration
            yaml_cal_template_fname = 'yamls/templates/template_for_calfactors.yaml'
            #----

            #--------
            #--------
            #create yaml
            yaml_fd = 'yamls/for_paper/'
            chains_fd = 'chains/for_paper/'
            if experiment not in cmb_experiments_bands_dic: #not a CMB experiment.
                exp_data_cosmo_str = '%s-%s' %(dataset, cosmo)                
            else:
                exp_data_cosmo_str = '%s-%s-%s' %(experiment, dataset, cosmo)                

            yaml_fname = '%s/%s.yaml' %(yaml_fd, exp_data_cosmo_str)
            outputfolder = '%s/%s/' %(chains_fd, exp_data_cosmo_str)

            if not os.path.exists( yaml_fd ): os.system('mkdir -p %s' %(yaml_fd))
            if not os.path.exists( outputfolder ): os.system('mkdir -p %s' %(outputfolder))

            #---fill details
            yaml_cosmo_template = open(yaml_cosmo_template_fname, 'r')
            yaml_f = open(yaml_fname, 'w')
            for line in yaml_cosmo_template:

                opline = line#.strip('\n')

                if opline.find('placeholder_for_likelihood')>-1: #likelihood name
                    dataset_split = dataset.split('+')
                    likelihood_str = ''
                    for ddd in dataset_split:
                        if ddd == 'lssty3_sne_mock':
                            likelihood_str = '%s\n%ssne_likelihoods.lssty3_sne_mock: null' %(likelihood_str, spaceval)
                        elif ddd == 'desidr2bao_mock':
                            likelihood_str = '%s\n%sbao.generic:' %(likelihood_str, spaceval)
                            likelihood_str = '%s\n%s%smeasurements_file: bao_data/desi_bao_dr2/desi_bao_dr2_mock/desi_gaussian_bao_ALL_GCcomb_mean_camb.txt' %(likelihood_str, spaceval, spaceval)
                            likelihood_str = '%s\n%s%scov_file: bao_data/desi_bao_dr2/desi_bao_dr2_mock/desi_gaussian_bao_ALL_GCcomb_cov.txt' %(likelihood_str, spaceval, spaceval)
                        else:
                            likelihood_str = '%s\n%scmb_likelihoods.%s_%s: null' %(likelihood_str, spaceval, experiment, dataset)

                    likelihood_str = likelihood_str.strip('\n')
                    opline = opline.replace('placeholder_for_likelihood', likelihood_str)

                if opline.find('placeholder_for_outputfolder')>-1: #chain output folder
                    opline = opline.replace('placeholder_for_outputfolder', outputfolder)

                yaml_f.writelines('%s' %(opline))
            yaml_cosmo_template.close()
            yaml_f.writelines('\n')

            #---fill in the calibration factors now
            if experiment in cmb_experiments_bands_dic:
                exp_bands = cmb_experiments_bands_dic[experiment]
                total_bands = len( exp_bands )
                for TP in ['T', 'P']: #T and P cals for each band now.
                    for bandcntr, curr_band in enumerate( exp_bands ):
                        yaml_cal_template = open(yaml_cal_template_fname, 'r')
                        for line in yaml_cal_template:
                            opline = line#.strip('\n')                            
                            if opline.find('placeholder_for_calparam')>-1: #calibration parameter name
                                calparam_name = '\n%smap%sCal%s' %(spaceval, TP, bandcntr+1)
                                #print(calparam_name)
                                opline = opline.replace('placeholder_for_calparam', calparam_name)
                            
                            if opline.find('placeholder_for_latexcalparam')>-1: #calibration parameter latex name
                                calparam_latex_name = r'%s_{\mathrm cal^\mathrm{%s}}' %(TP, curr_band)
                                #print(calparam_latex_name)
                                opline = opline.replace('placeholder_for_latexcalparam', calparam_latex_name)

                            yaml_f.writelines('%s' %(opline))
                        yaml_cal_template.close()
            yaml_f.writelines('\n')

            #---done.
            yaml_f.close()

            #--------
            #--------

            #--------
            #submit chains now.
            jobname = exp_data_cosmo_str
            num_nodes, num_cpus = 4, 20
            cmd = 'sbatch -J %s submit_chains.sh %s %s %s' %(jobname, yaml_fname, num_nodes, num_cpus)
            print(cmd, '\n')
    print('\n')

