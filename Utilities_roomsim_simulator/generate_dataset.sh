#!/bin/bash
#OAR -q production
#OAR -p cluster='grvingt'
#OAR -l walltime=30:00:00
#OAR -O OUT/oar_job.%jobid%.output
#OAR -E OUT/oar_job.%jobid%.error

source /home/psrivastava/base-env/bin/activate
python /home/psrivastava/baseline/sofamyroom_2/test.py



