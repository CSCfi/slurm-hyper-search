#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import re
import sys

from glob import glob

def get_logfile(ndir, ni):
    run_ids = [int(x.split('_')[-1]) for x in
               glob(os.path.join(ndir, 'slurm_id_*'))]
    if len(run_ids) > 0:
        last_run_id = sorted(run_ids)[-1]
        run_log = 'slurm-{}_{}.out'.format(last_run_id, ni)
        if os.path.isfile(run_log):
            return run_log
    return None


def get_logerror(run_log):
    if run_log is None:
        return None
    
    with open(run_log, 'r') as fp:
        for line in fp:
            m = re.search("terminate called after throwing an instance of '(.*)'", line)
            if m:
                return m.group(1)
    return None


def print_warnings(res, msg, T):
    if res:
        if args.verbose:
            print('WARNING: the following runs have {}:'.format(msg),
                  ','.join(sorted(str(x) for x in res)))
        else:
            n = len(res)
            print('WARNING: {}/{} = {:.2%} runs had {}. (Try --verbose to which runs.)'.format(n, T, n/T, msg))


def main(args):
    out_dir = args.dir

    if not os.path.isdir(out_dir):
        print("ERROR: Directory {} doesn't exist!".format(out_dir))
        return 1

    T = 0
    no_results = set()
    nan_results = set()
    empty_results = set()
    results = {}
    best_n = None
    succeeded = 0
    col_space = None

    for n in os.listdir(out_dir):
        if not n.isdigit():
            continue
        ni = int(n)
        T += 1
        ndir = os.path.join(out_dir, n)
        results_file = os.path.join(ndir, args.results)
        params_file = os.path.join(ndir, 'params')
        if not os.path.isfile(results_file):
            log_fname = get_logfile(ndir, ni)
            err = get_logerror(log_fname)
            if err is None:
                no_results.add(ni)
            elif err == 'fasttext::DenseMatrix::EncounteredNaNError':
                nan_results.add(ni)
            else:
                print('Unexpected error', err, log_fname)
                no_results.add(ni)
        elif os.path.getsize(results_file) == 0:
            empty_results.add(ni)
        else:
            params = None
            with open(params_file, 'r') as fp:
                params = fp.read().rstrip()
            with open(results_file, 'r') as fp:
                res = {}
                for line in fp:
                    m_name, m_value = line.split()
                    assert len(m_name) > 0
                    res[m_name] = float(m_value)
                if args.measure in res:
                    succeeded += 1
                    
                res['params'] = params
                
                if col_space is None or len(params) > col_space:
                    col_space = len(params)
                
                results[ni] = res
                    # if m_name == args.measure:
                    #     succeeded += 1
                    #     v = float(m_value)
                        # results[ni] = v
                        # if best_n is None:
                        #     best_n = ni
                        # else:
                        #     bv = results[best_n]
                        #     if args.opt == 'max' and v > bv:
                        #         best_n = ni
                        #     elif args.opt == 'min' and v < bv:
                        #         best_n = ni

    df = pd.DataFrame.from_dict(results, orient='index')
    if args.opt == 'max':
        dfs = df.nlargest(args.N, args.measure)
    else:
        dfs = df.nsmallest(args.N, args.measure)

    print('Best {} results so far according to {} {} (out of {} succeeded runs).'.format(
        args.N, args.opt, args.measure, succeeded))
    print(dfs)

    print()
    print_warnings(empty_results, 'empty results files', T)
    print_warnings(no_results, 'no results files', T)
    print_warnings(nan_results, 'NaN errors', T)


if __name__ == '__main__':
    pd.set_option("display.max_colwidth", 1000)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str,
                        help='directory with parameter and results files')
    parser.add_argument('--results', type=str, default='results', required=False,
                        help='results file name')
    parser.add_argument('--measure', type=str, default='P@5', required=False,
                        help='measure to optimize')
    parser.add_argument('-N', type=int, default=5, 
                        help='number of top results to show')
    parser.add_argument('--opt', type=str,
                        choices=['max', 'min'], default='max', required=False,
                        help='whether to minimize or maximize the measure')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    sys.exit(main(args))
