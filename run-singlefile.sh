#!/bin/bash
#SBATCH --job-name=ft-train
#SBATCH --account=project_2001825
#SBATCH --partition=small
#SBATCH --time=0-1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

TMPDIR=/scratch/project_2001825/tmp

PARAMS_FILE=$1
RESULTS_FILE=$2
PARAMS_OFFSET=$3

if [ ! -f "$PARAMS_FILE" -o -z "$RESULTS_FILE" ]
then
    echo "Usage: $0 PARAMS_FILE RESULTS_FILE [PARAMS_OFFSET]"
    exit 1
fi

if [ -z "$PARAMS_OFFSET" ]
then
    PARAMS_OFFSET=0
fi

PARAMS_ID=$(( $SLURM_ARRAY_TASK_ID + $PARAMS_OFFSET ))
JOB_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

RUNLOG_FILE="${RESULTS_FILE}.runlog"
echo "$PARAMS_ID|$JOB_NAME|$SLURM_SUBMIT_DIR" >> $RUNLOG_FILE

MODEL_FILE=$(mktemp)
PARAMS=$(tail -n +${PARAMS_ID} scratch/yso-2019-swe.params | head -n 1)
FASTTEXT=/projappl/project_2001825/mvsjober/fastText/fasttext

#TRAIN_DATA=/scratch/project_2001825/data/txt/yso-cicero-finna-fi-train-lc-voikko.txt
TRAIN_DATA=/scratch/project_2001825/data/txt/yso-cicero-finna-sv.tsv.txt

# TEST_DATA=$(ls /scratch/project_2001825/data/txt/*{validate,test}*txt)
# TEST_DATA=/scratch/project_2001825/data/txt/combined-fi-validate-lc-voikko.txt
TEST_DATA=$(ls /scratch/project_2001825/data/txt/{jyu-theses-swe,kirjaesittelyt-yso-swe-all,varia-corpus-sv-all,swe-combined-valid,swe-combined-test}.txt)

echo "*** TRAIN ***"
(set -x
 srun $FASTTEXT supervised -verbose 1 -label "http://www.yso.fi/" -input $TRAIN_DATA -output $MODEL_FILE -loss hs $PARAMS
)

test $? -ne 0 && exit 1

echo "*** TEST ***"
for TD in $TEST_DATA
do
    BN=$(basename $TD .txt)
    TMPFILE=${MODEL_FILE}.$BN.results
    echo -n "$PARAMS_ID|$PARAMS|$JOB_NAME|$BN|" > $TMPFILE
    (set -x
     srun $FASTTEXT test ${MODEL_FILE}.bin $TD 5 | tr '\n\t' '| ' >> $TMPFILE
    )
    echo >> $TMPFILE
    cat $TMPFILE >> $RESULTS_FILE
    rm $TMPFILE
done
rm ${MODEL_FILE}.bin
rm ${MODEL_FILE}.vec
