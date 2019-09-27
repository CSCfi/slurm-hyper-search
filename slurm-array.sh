#!/bin/bash
#SBATCH --job-name=ft-train
#SBATCH --account=project_2001825
#SBATCH --partition=small
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=1G

FASTTEXT=~/projappl/hpd/mvsjober/fastText/fasttext
TRAIN_DATA=~/scratch/hpd/mvsjober/fasttext/yso-cicero-fasttext-train-10k.txt
TEST_DATA=~/scratch/hpd/mvsjober/fasttext/yso-cicero-fasttext-valid.txt

PARAMS_FILE="$1/${SLURM_ARRAY_TASK_ID}/params"
RESULTS_FILE="$1/${SLURM_ARRAY_TASK_ID}/results"
SLURM_ID_FILE="$1/${SLURM_ARRAY_TASK_ID}/slurm_id_${SLURM_JOB_ID}"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: could not find parameter file $PARAMS_FILE"
    exit 1
fi
touch $SLURM_ID_FILE

MODEL=$(mktemp)
PARAMS=$(cat $PARAMS_FILE)

set -xv
srun $FASTTEXT supervised -verbose 1 -input $TRAIN_DATA -output $MODEL $PARAMS
srun $FASTTEXT test ${MODEL}.bin $TEST_DATA 5 > $RESULTS_FILE
