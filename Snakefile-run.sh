#!/bin/bash
#SBATCH -J Snakefile
#SBATCH -o Snakefile.out
#SBATCH -e Snakefile.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G

date +"%Y-%m-%d %H:%M:%S"

source .venv/bin/activate
snakemake --executor slurm --jobs 20 --latency-wait 1 --cores 1 --scheduler ilp --scheduler-ilp-solver PULP_CBC_CMD #--forcerun train_psp
