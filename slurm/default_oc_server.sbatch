#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute03

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
optimizer="nsga2"
experiment="runs/moo_sc/population/pop_comparison.py"
dataset="protein_structure"
study_name="PopComp nsga2 c:128"
config="128"

srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p $dataset -j $SLURM_JOB_ID -o $optimizer -c $config"
