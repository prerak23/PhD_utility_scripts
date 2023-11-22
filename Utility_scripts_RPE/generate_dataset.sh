#!/bin/bash
#OAR -q production
#OAR -p cluster='grue'
#OAR -l walltime=20:00:00
#OAR -O OUT/oar_job.%jobid%.output
#OAR -E OUT/oar_job.%jobid%.error

source /home/psrivastava/base-env_2/bin/activate
python /home/psrivastava/baseline/scripts/pre_processing/mlh_baseline.py



