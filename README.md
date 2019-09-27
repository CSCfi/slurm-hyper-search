# Simple hyper-parameter generation for optimizing on Slurm-based HPC clusters

First, you need to modify the parameter generation in [`generate_params.py`](generate_params.py), and the slurm script in [`slurm-array.sh`](slurm-array.sh) for your particular case.

## Usage

To generate a run called `test-1` with 10 random parameter sets:

    ./generate_params.py test-1 10
    
It will create files like `test-1/N/params` where `N` is the parameter set index from 0 to 9.  If you run the same command again it will detect the existing files and continue from 10 onward.

To submit the 10 first runs to slurm:

    sbatch -a 0-9 slurm-array.sh test-1
    
At any time you can check what is the best run so far (it will also report missing and empty results files, which might indicate problems):

    ./check_results.py test-1 P@5

The second argument is the measure you want to optimize.  The script assumes that the format of the results file is one measure per line, and the measure name and value are whitespace-separated.
