# Simple hyper-parameter search for Slurm-based HPC clusters

<img src="https://www.csc.fi/documents/10180/524564/en_horizontal_cef_logo_2-800px.png/cda2f571-dfca-cda3-40fb-a8a06656d46b?t=1602756015080&imagePreview=1" />

This repository aims to provide a very light-weight and implementation agnostic template for running hyper-parameter optimization runs on a Slurm-based HPC cluster.  The main ideas:

- **Random search for hyper-parameter optimization**: easy to implement and has been deemed as more efficient than grid search. See [*Random Search for Hyper-Parameter Optimization*, J. Bengstra and Y. Bengio, JMLR 13 (2012)](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf).

- **Implementation agnostic**: the training and evaluation programs are launched as shell commands and can thus be implemented in any programming language, the only requirement is that they take the parameters as command line arguments.

- **HPC-friendly**: works by appending to a few big files, not producing thousands of small files which works inefficiently with distributed file systems like [Lustre](https://en.wikipedia.org/wiki/Lustre_(file_system)).

- **Unix-philosophy**: based on human readable text files instead of binary files.

The scripts provided in this repository happens to implement a particular scenario for text classification using fasttext and evaluating on several testsets.  You can use these as examples of how to implement your own scenario.


## Usage

### Generate parameters

First, you need to create a file `space_conf.py` specifying the parameter space, you can use [`space_conf.py.example`](space_conf.py.example) as a staring point.  Below are some examples of what you can do:

```python
space = {
    'n: [1, 2, 3, 4, 5],                   # simply specify the possible parameters in a list
    'dim': np.arange(50, 1001, step=10),   # use a numpy linear range
    'lr': np.geomspace(0.01, 5, num=20),   # ... or log scale
    'gamma': scipy.stats.expon(scale=.1),  # or specify a probability distribution with scipy
}
```

To generate a set of runs called `test-1` with 100 random samples from the parameter space, run:

    ./generate_params.py test-1 100
    
This will create a file `test-1/params` with 100 rows.  Each row corresponds to one random sample from the parameter space (specified in `generate_params.py`).  If you run the same command again it will concatenate the parameter file with 100 more random runs.

For a small parameter space it might make sense to generate all combinations (this will naturally not work if you are using probability distributions):

    ./generate_params.py test-1 all

By default the arguments are generated in GNU long format, e.g. `--dim=50 --lr=0.01`.  You can modify the formatting with the `--format` argument, e.g.:

    ./generate_params.py test-1 100 --format='-{name} {value}'

### Submit runs

First, you need to edit the [`run-example.sh`](run-example.sh) script for your particular case.  Typically it will run a training script with the given parameters, and then evaluate the results producing some evaluation measures.  In the following, we assume that you call the edited file `run.sh`.

To submit the 10 first runs to slurm you can run:

    sbatch -a 1-10 run.sh test-1
    
The array parameter refers to the line number in the parameter file, so `-a 1-10` means running the 10 runs corresponding to the ten first lines from the parameter file.


The `run.sh` script produces output to two files:

- Before starting the run a line is appended to `test-1/runlog` with the following contents:

      LINE_NUMBER|SLURM_ID|SLURM_SUBMIT_DIR
  
  This information is useful when later analysing failed runs.

- At the end of the run one or more lines will be appended to `test-1/results` with content like this:

      LINE_NUMBER|PARAMETERS|SLURM_ID|TESTSET_NAME|MEASURE_1|MEASURE_2|...|
    
The `run.sh` needs to take care of reading the output of the commands and format them into this single-line format.  For long running jobs it's best to store the output into a temporary file and append the final contents to `results` only at the very end.  This is to reduce the risk of several parallel jobs having opened the file for appending at the same time.

**Note:** this design assumes that appending to a file (for example the `results` file) works concurrently, i.e., that you can append from multiple programs without any corruption of the file.  This [*should* be the case](https://nullprogram.com/blog/2016/08/03/) if the filesystem is [POSIX-compatible](https://pubs.opengroup.org/onlinepubs/9699919799/functions/write.html), and in particular should be the case for Lustre.


### Analyze results and possible errors

At any time you can check what is the best run so far according to a specific measure:

    ./analyze_results.py test-1 --measure P@5

To check for errors and missing runs you can try:

    ./check_status.py test-1 --log_dir logs/
    
This command will also try to analyze the slurm logs (you might need to change the code here for your own case).  The command will also print a list of line numbers for the failed runs, which you can then copy and paste as array parameters to resubmit, e.g.:

    sbatch -a 4,5,9 run.sh test-1
