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

PARAMS="$1/${SLURM_ARRAY_TASK_ID}/params"
if [ ! -f "$PARAMS" ]; then
    echo "ERROR: could not find parameter file $PARAMS"
    exit 1
fi

MODEL=$(mktemp)

set -xv
srun $FASTTEXT supervised -input $TRAIN_DATA -output $MODEL $(cat $PARAMS)
srun $FASTTEXT test ${MODEL}.bin $TEST_DATA 5
