# Simple hyper-parameter generation for optimizing on Slurm-based HPC clusters

First, you need to modify the parameter generation in [`generate_params.py`](generate_params.py), and the slurm script in [`slurm-array.sh`](slurm-array.sh) for your particular case.

## Usage

To generate a run called `test-1` with 10 random parameter sets:

    ./generate_params.py test-1 10
    
It will create files like `test-1/N/params` where `N` is the parameter set index from 0 to 9.  If you run the same command again it will detect the existing files and continue from 10 onward.

To submit the 10 first runs to slurm:

    sbatch -a 0-9 slurm-array.sh test-1
    
