#!/bin/bash
#SBATCH --job-name=ft-train
#SBATCH --account=project_2001825
#SBATCH --partition=small
#SBATCH --time=0-1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

DB_FILE=$1
PARAMS_OFFSET=$2

if [ ! -f "$DB_FILE" ]
then
    echo "Usage: $0 DB_FILE [PARAMS_OFFSET]"
    exit 1
fi

if [ -z "$PARAMS_OFFSET" ]
then
    PARAMS_OFFSET=0
fi

PARAMS_ID=$(( $SLURM_ARRAY_TASK_ID + $PARAMS_OFFSET ))
JOB_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

MODEL_FILE=$(mktemp)
PARAMS=$(python3 get_params.py $DB_FILE $PARAMS_ID)
FASTTEXT=/projappl/project_2001825/mvsjober/fastText/fasttext

#TRAIN_DATA=/scratch/project_2001825/data/txt/yso-cicero-finna-fi-train-lc-voikko.txt
TRAIN_DATA=/scratch/project_2001825/data/txt/yso-cicero-finna-sv.tsv-stem.txt

# TEST_DATA=$(ls /scratch/project_2001825/data/txt/*{validate,test}*txt)
# TEST_DATA=/scratch/project_2001825/data/txt/combined-fi-validate-lc-voikko.txt
TEST_DATA=$(ls /scratch/project_2001825/data/txt/{jyu-theses-swe,kirjaesittelyt-yso-swe-all,varia-corpus-sv-all,swe-combined-valid,swe-combined-test}-stem.txt)

echo "*** TRAIN ***"
(set -x
 srun $FASTTEXT supervised -verbose 1 -label "http://www.yso.fi/" -input $TRAIN_DATA -output $MODEL_FILE -loss hs $PARAMS
)

test $? -ne 0 && exit 1

echo "*** TEST ***"
for TD in $TEST_DATA
do
    BN=$(basename $TD .txt)
    (set -x
     srun $FASTTEXT test ${MODEL_FILE}.bin $TD 5 | python3 store_results.py $DB_FILE $PARAMS_ID --slurm_id $JOB_NAME --result_name $BN
    )
done
rm ${MODEL_FILE}.bin
rm ${MODEL_FILE}.vec
