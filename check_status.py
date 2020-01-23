#!/usr/bin/env python3

from collections import defaultdict
import argparse
import itertools
import os
import re


def load_results(fn):
    results = []
    with open(fn, 'r') as fp:
        for line in fp:
            parts = line.rstrip().split('|')
            r = {
                'param_id': int(parts[0]),
                'slurm_id': parts[2],
                'result_name': parts[3]
            }
            for p in parts[1].split('-'):
                if len(p) > 0:
                    n, v = p.rstrip().split()
                    v = int(v) if v.isdigit() else float(v)
                    r[n] = v
            for p in parts[4:]:
                if len(p) > 0:
                    n, v = p.split()
                    v = int(v) if v.isdigit() else float(v)
                    r[n] = v
            results.append(r)
    return results


def load_params(fn):
    param_ids = []
    i = 1
    with open(fn, 'r') as fp:
        for line in fp:
            param_ids.append(i)
            i += 1
    return param_ids


def load_runlog(fn):
    if not os.path.isfile(fn):
        print("WARNING: no runlog found at", fn)
        return None

    runlog = []
    with open(fn, 'r') as fp:
        for line in fp:
            parts = line.rstrip().split('|')
            runlog.append({
                'param_id': int(parts[0]),
                'slurm_id': parts[1],
                'submit_dir': parts[2]
            })
    return runlog


def get_logerror(fn):
    if not os.path.isfile(fn):
        print('WARNING: could not find log file {}!  Try specifying --log_dir,'
              ' or --skip_logs if you don\'t want to analyze logs.'.format(fn))
        return None

    with open(fn, 'r') as fp:
        for line in fp:
            m = re.search("terminate called after throwing an instance of "
                          "'(.*)'", line)
            if m:
                return m.group(1)
    return None


def print_warnings(res, msg, tot):
    if res:
        n = len(res)
        print('WARNING: the following param indices have {}, '
              '{}/{} = {:.2%}:'.format(msg, n, tot, n/tot))
        for k, g in itertools.groupby(sorted(res),
                                      key=lambda x: (x-1)//1000):
            offset = k*1000
            print('[OFFSET: {}]'.format(offset), end=' ')
            print(','.join(str(x-offset) for x in g))


def main(args):
    in_dir = args.input

    params_fn = os.path.join(in_dir, 'params')
    param_ids = load_params(params_fn)
    print('Read {} which contained {} sets of parameters.'.format(
        params_fn, len(param_ids)))
    N = len(param_ids)

    runlog_fn = os.path.join(in_dir, 'runlog')
    runlog = load_runlog(runlog_fn)
    print('Read {} which contained data on {} runs.'.format(
        runlog_fn, len(runlog)))

    results = []
    results_fn = os.path.join(in_dir, 'results')

    results = load_results(results_fn)
    print('Read {} which contained {} results.'.format(results_fn,
                                                       len(results)))

    params_results = defaultdict(list)  # dict: param_id -> list of results
    testsets = set()
    for r in results:
        params_results[r['param_id']].append(r)
        testsets.add(r['result_name'])

    # params_without_runs = set(param_ids.copy())
    params_runs = defaultdict(list)  # dict: param_id -> list of runs
    for r in runlog:
        params_runs[r['param_id']].append(r)

    pids_with_unknown_errors = []
    pids_with_nan_errors = []
    pids_without_runs = []
    for pid in param_ids:
        res = params_results[pid]
        if len(res) == 0:
            runs = params_runs[pid]
            if len(runs) == 0:
                pids_without_runs.append(pid)
            else:
                slurm_ids = [r['slurm_id'] for r in runs]
                if len(runs) > 1:
                    print('WARNING: paramset {} has more than one run: ',
                          ', '.join(slurm_ids))
                slurm_id = slurm_ids[-1]

                if args.skip_logs:
                    pids_with_unknown_errors.append(pid)
                else:
                    slurmlog_fn = os.path.join(args.log_dir, 'slurm-{}.out'
                                               .format(slurm_id))
                    err = get_logerror(slurmlog_fn)
                    if err == 'fasttext::DenseMatrix::EncounteredNaNError':
                        pids_with_nan_errors.append(pid)
                    else:
                        pids_with_unknown_errors.append(pid)
                        if err is not None:
                            print('WARNING: Unknown error', err, slurmlog_fn)
        else:  # len(res) > 0
            pid_testsets = [r['result_name'] for r in res]
            pid_testsets_set = set(pid_testsets)
            if pid_testsets_set != testsets:
                print('WARNING: paramset {} is missing some results:'.format(
                    pid), pid_testsets_set)
            elif len(pid_testsets) != len(testsets):
                print('WARNING: paramset {} has wrong number of results: {}'.
                      format(pid, len(pid_testsets)))

    print_warnings(pids_without_runs, 'no runs', N)
    print_warnings(pids_with_nan_errors, 'NaN errors', N)
    print_warnings(pids_with_unknown_errors, 'unknown errors', N)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='directory with the results file')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--skip_logs', action='store_true')
    args = parser.parse_args()

    main(args)
