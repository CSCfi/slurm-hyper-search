#!/bin/bash
#SBATCH --job-name=ft-train
#SBATCH --account=project_2001825
#SBATCH --partition=small
#SBATCH --time=1-0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

OUTPUT_DIR="$1/${SLURM_ARRAY_TASK_ID}"
PARAMS_FILE="${OUTPUT_DIR}/params"
RESULTS_FILE="${OUTPUT_DIR}/results"
MODEL_FILE="${OUTPUT_DIR}/${SLURM_ARRAY_JOB_ID}"

touch "${OUTPUT_DIR}/slurm_id_${SLURM_ARRAY_JOB_ID}"

if [ ! -f "$PARAMS_FILE" ]; then
    echo "ERROR: could not find parameter file $PARAMS_FILE"
    exit 1
fi

PARAMS=$(cat $PARAMS_FILE)
FASTTEXT=/projappl/project_2001825/mvsjober/fastText/fasttext
TRAIN_DATA=/scratch/project_2001825/data/txt/yso-cicero-finna-fi-train-lc-voikko.txt
TEST_DATA=$(ls /scratch/project_2001825/data/txt/*{validate,test}*txt)

echo "*** TRAIN ***"
(set -x
 srun $FASTTEXT supervised -verbose 1 -label "http://www.yso.fi/" -input $TRAIN_DATA -output $MODEL_FILE -loss hs $PARAMS
)

test $? -ne 0 && exit 1

echo "*** TEST ***"
for TD in $TEST_DATA
do
    TD_RESULTS_FILE="${RESULTS_FILE}.$(basename $TD .txt)"
    (set -x
     srun $FASTTEXT test ${MODEL_FILE}.bin $TD 5 > $TD_RESULTS_FILE
    )
done
rm ${MODEL_FILE}.bin
rm ${MODEL_FILE}.vec
