#!/usr/bin/env python3

import argparse
import os
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


def main(args):
    out_dir = args.dir

    if not os.path.isdir(out_dir):
        print("ERROR: Directory {} doesn't exist!".format(out_dir))
        return 1

    no_results = set()
    nan_results = set()
    empty_results = set()
    results = {}
    best_n = None
    succeeded = 0

    for n in os.listdir(out_dir):
        if not n.isdigit():
            continue
        ni = int(n)
        ndir = os.path.join(out_dir, n)
        results_file = os.path.join(ndir, 'results')
        if not os.path.isfile(results_file):
            err = get_logerror(get_logfile(ndir, ni))
            if err is None:
                no_results.add(ni)
            elif err == 'fasttext::DenseMatrix::EncounteredNaNError':
                nan_results.add(ni)
            else:
                print('Unexpected error', err)
                no_results.add(ni)
        elif os.path.getsize(results_file) == 0:
            empty_results.add(ni)
        else:
            with open(results_file, 'r') as fp:
                for line in fp:
                    m_name, m_value = line.split()
                    if m_name == args.measure:
                        succeeded += 1
                        v = float(m_value)
                        results[ni] = v
                        if best_n is None:
                            best_n = ni
                        else:
                            bv = results[best_n]
                            if args.opt == 'max' and v > bv:
                                best_n = ni
                            elif args.opt == 'min' and v < bv:
                                best_n = ni

    print('Best ({}) result so far: {} = {} in run {} (out of {} succeeded runs).'.format(
        args.opt, args.measure, results[best_n], best_n, succeeded))
    print('Best parameters:')
    print(' ', open(os.path.join(out_dir, str(best_n), 'params')).read())

    if empty_results:
        print('WARNING: the following runs have empty results files:',
              ','.join(sorted(str(x) for x in empty_results)))

    if no_results:
        print('WARNING: the following runs have no results files:',
              ','.join(sorted(str(x) for x in no_results)))

    if nan_results:
        print('WARNING: the following runs had NaN errors:',
              ','.join(sorted(str(x) for x in nan_results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str,
                        help='directory with parameter and results files')
    parser.add_argument('--measure', type=str, default='P@5', required=False,
                        help='measure to optimize')
    parser.add_argument('--opt', type=str,
                        choices=['max', 'min'], default='max', required=False,
                        help='whether to minimize or maximize the measure')

    args = parser.parse_args()

    sys.exit(main(args))
