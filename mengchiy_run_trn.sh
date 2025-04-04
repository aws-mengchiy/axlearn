#!/bin/bash
#SBATCH --output=slurm-%x-%j.out
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --nodelist=ip-10-0-2-119


srun  --kill-on-bad-exit=1  run_step_time_multinodes_test.sh fuji-1B-v2-flash