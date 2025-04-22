#!/usr/bin/env bash
#SBATCH --time=96:00:00
#SBATCH --partition=cpu
#SBATCH --output=output/output-%j.txt
#SBATCH --cpus-per-task=5
#SBATCH --mem=20G
#SBATCH --nodelist=oc-compute03

JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
export PYTHONPATH=${PYTHONPATH}:${JOB_DIR}
experiment="runs/moo_sc_comparison_tests/nsgaii_tuning.py"
dataset="airfoil_self_noise"
optimizer="NSGA-II"
study_name="NSGA-II Tuning - $dataset"

srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p airfoil_self_noise -j $SLURM_JOB_ID -o $optimizer"

srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p concrete_strength -j $SLURM_JOB_ID -o $optimizer"

srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p combined_cycle_power_plant -j $SLURM_JOB_ID -o $optimizer"

srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p protein_structure -j $SLURM_JOB_ID -o $optimizer"

srun nix-shell $JOB_DIR/slurm/default.nix --command "PYTHONPATH=$JOB_DIR/$PYTHONPATH \
python $JOB_DIR/$experiment -p parkinson_total -j $SLURM_JOB_ID -o $optimizer"