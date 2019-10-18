#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import re
import tqdm
import subprocess
import sys

from glob import glob


def get_runid(ndir, ni):
    run_ids = [x.split('_', maxsplit=2)[-1] for x in
               glob(os.path.join(ndir, 'slurm_id_*'))]
    if len(run_ids) > 0:
        last_run_id = sorted(run_ids)[-1]
        if '_' not in last_run_id:
            ni %= 1000
            last_run_id = '{}_{}'.format(last_run_id, ni)
        return last_run_id
    return None


def get_logfile(run_id):
    if run_id is not None:
        run_log = 'slurm-{}.out'.format(run_id)
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
            print('WARNING: {}/{} = {:.2%} runs had {}. (Try --verbose.)'.
                  format(n, T, n/T, msg))


def add_slurm_info(run_id, res):
    cmd = 'sacct -P -n -a --format JobID,State,ElapsedRaw,MaxRSS -j {}'.format(run_id)
    out = subprocess.run(cmd.split(), capture_output=True, text=True)
    out = [r.split('|') for r in out.stdout.split('\n')]
    res['slurm:status'] = out[0][1]
    res['slurm:elapsed_sec'] = int(out[0][2])

    max_m = 0.0
    for r in out:
        if len(r) < 4:
            continue

        m = r[3]
        if len(m) == 0:
            continue

        if m.isdigit():
            m = float(m)
        elif m[-1] == 'K':
            m = float(m[:-1]) * 1024.0
        elif m[-1] == 'M':
            m = float(m[:-1]) * 1024.0 * 1024.0
        elif m[-1] == 'G':
            m = float(m[:-1]) * 1024.0 * 1024.0 * 1024.0
        else:
            assert False
        if m > max_m:
            max_m = m
    res['slurm:mem_gb'] = max_m / 1024.0 / 1024.0 / 1024.0


def main(args):
    T = 0
    no_results = set()
    nan_results = set()
    empty_results = set()
    results = {}
    succeeded = 0

    out_dir = args.dir
    if not os.path.isdir(out_dir):
        print("ERROR: Directory {} doesn't exist!".format(out_dir))
        return 1

    for n in tqdm.tqdm(os.listdir(out_dir), desc='Processing'):
        if not n.isdigit():
            continue
        ni = int(n)
        T += 1
        ndir = os.path.join(out_dir, n)
        results_file = os.path.join(ndir, args.results)
        params_file = os.path.join(ndir, 'params')
        run_id = get_runid(ndir, ni)
        if not os.path.isfile(results_file):
            log_fname = get_logfile(run_id)
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
                    res[m_name] = int(m_value) if m_value.isdigit() else float(m_value)
                if args.measure in res:
                    succeeded += 1

                pa = params.split()
                for k, v in zip(pa[::2], pa[1::2]):
                    if k[0] == '-':
                        k = k[1:]
                    res[k] = int(v) if v.isdigit() else float(v)

                if run_id is not None and args.slurm:
                    add_slurm_info(run_id, res)
                results[ni] = res

    df = pd.DataFrame.from_dict(results, orient='index')
    if args.output:
        df.to_csv(args.output)
        print('Wrote results to', args.output)
    if args.opt == 'max':
        dfs = df.nlargest(args.N, args.measure)
    else:
        dfs = df.nsmallest(args.N, args.measure)

    print('Best {} results so far according to {} {} (out of {} succeeded).'.
          format(args.N, args.opt, args.measure, succeeded))
    print(dfs)

    print()
    print_warnings(empty_results, 'empty results files', T)
    print_warnings(no_results, 'no results files', T)
    print_warnings(nan_results, 'NaN errors', T)


if __name__ == '__main__':
    pd.set_option("display.max_colwidth", 1000)

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str,
                        help='directory with parameters and results files')
    parser.add_argument('--results', type=str, default='results',
                        required=False, help='results file name')
    parser.add_argument('--measure', type=str, default='P@5', required=False,
                        help='measure to optimize')
    parser.add_argument('-N', type=int, default=5,
                        help='number of top results to show')
    parser.add_argument('--opt', type=str,
                        choices=['max', 'min'], default='max', required=False,
                        help='whether to minimize or maximize the measure')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', '-O', type=str)
    parser.add_argument('--slurm', action='store_true',
                        help='enable recording slurm info '
                        '(runtime, memory usage, etc.)')
    args = parser.parse_args()

    sys.exit(main(args))
