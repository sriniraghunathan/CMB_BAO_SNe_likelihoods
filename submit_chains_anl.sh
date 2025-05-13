#!/bin/bash
#SBATCH --partition=SPT3G
#SBATCH --output=batch_jobs/job.o%j
#SBATCH --cpus-per-task=50
###SBATCH --exclude=ccc0240,ccc0242
#SBATCH --nodes=1
#SBATCH --time=24:0:00
#SBATCH --mem=20G
export SHELL=bash

#example
#coabya-run yamls/so_baseline_sampler_plus_desidr2boa.yaml
cobaya-run $1
