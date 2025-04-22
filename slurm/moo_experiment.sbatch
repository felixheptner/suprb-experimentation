#!/usr/bin/env bash
###############################################################################
# Slurm directives that apply to every array task
###############################################################################
#SBATCH --time=96:00:00                 # 96 h wall‑clock limit
#SBATCH --partition=cpu                # queue / partition
#SBATCH --nodelist=oc-compute03        # stay on this node
#SBATCH --cpus-per-task=5              # 5 CPUs per array task
#SBATCH --mem=20G                      # 20 GiB RAM per array task
#SBATCH --output=output/%A_%a.out      # stdout file  (%A=array‑id, %a=task‑id)
#SBATCH --error=output/%A_%a.err       # stderr file
#SBATCH --array=0-4                    # ----  five array tasks  ----
#
# Submit: sbatch nsgaii_array.sh
# Check : squeue -j <array‑id>      or  sacct -j <array‑id>
###############################################################################

# ---------------------------------------------------------------------------
# 1. Per‑job environment
# ---------------------------------------------------------------------------
JOB_DIR=/data/oc-compute03/$USER/suprb-experimentation
export PYTHONPATH="${PYTHONPATH}:${JOB_DIR}"
EXPERIMENT="runs/moo_sc_comparison_tests/nsgaii_tuning.py"
OPTIMIZER="NSGA-II"

# ---------------------------------------------------------------------------
# 2. Map array‑task ID → dataset name
# ---------------------------------------------------------------------------
datasets=(
  airfoil_self_noise
  concrete_strength
  combined_cycle_power_plant
  protein_structure
  parkinson_total
)

dataset="${datasets[$SLURM_ARRAY_TASK_ID]}"
study_name="NSGA-II Tuning - ${dataset}"

# ---------------------------------------------------------------------------
# 3. Launch the experiment
# ---------------------------------------------------------------------------
echo "[$(date)] Starting dataset '${dataset}' (array task ${SLURM_ARRAY_TASK_ID})"

srun nix-shell "${JOB_DIR}/slurm/default.nix" --command "
  python ${JOB_DIR}/${EXPERIMENT} \
    -p ${dataset} \
    -j ${SLURM_ARRAY_JOB_ID} \
    -o ${OPTIMIZER} \
    --study-name \"${study_name}\"
"