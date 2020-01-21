#!/usr/bin/env python3

import argparse
import itertools
import os
import pandas as pd
import re
import sys
from collections import defaultdict


def load_sqlite(fn):
    results = []

    import sqlite3
    conn = sqlite3.connect(fn)
    c = conn.cursor()
    c.execute('SELECT results.rowid, results.*, params.* FROM results '
              'LEFT JOIN params ON param_id=params.rowid')
    col_names = [re.sub(r'_(\d)', '@\\1', x[0]) for x in c.description]
    while True:
        row = c.fetchone()
        if row is None:
            break
        r = dict(zip(col_names, row))
        results.append(r)

    c.execute('SELECT rowid from params')
    param_ids = [x[0] for x in c.fetchall()]
    conn.close()

    return results, param_ids


def load_singletext(fn):
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
        return None

    runlog = {}
    with open(fn, 'r') as fp:
        for line in fp:
            parts = line.rstrip().split('|')
            param_id = int(parts[0])
            runlog[param_id] = {
                'slurm_id': parts[1],
                'submit_dir': parts[2]
            }
    return runlog


def get_logerror(fn):
    if os.path.isfile(fn):
        with open(fn, 'r') as fp:
            for line in fp:
                m = re.search("terminate called after throwing an instance of "
                              "'(.*)'", line)
                if m:
                    return m.group(1)
    return None


def print_warnings(res, msg, num_dirs):
    if res:
        if args.verbose:
            print('WARNING: the following runs have {}:'.format(msg))
            for k, g in itertools.groupby(sorted(res), key=lambda x: x//1000):
                offset = k*1000
                print('[OFFSET: {}]'.format(offset), end=' ')
                print(','.join(str(x-offset) for x in g))
        else:
            n = len(res)
            print('WARNING: {}/{} = {:.2%} runs had {}. (Try --verbose.)'.
                  format(n, num_dirs, n/num_dirs, msg))


def main(args):
    results = []

    for fn in args.input:
        if fn.endswith('.db') or args.sqlite:
            res, param_ids = load_sqlite(fn)
            if args.params:
                print('WARNING: no need to give --params with sqlite, '
                      'as the parameters are stored in the same database.')
        else:
            res = load_singletext(fn)
            if args.params:
                param_ids = load_params(args.params)
            elif args.check_errors:
                print('ERROR: if you wish to --check_errors with the '
                      'singletext format you also need to supply the '
                      'parameter file with --params.')
                sys.exit(1)

        print('Read {} which contained {} results.'.format(fn, len(res)))

        testset_counts = defaultdict(int)  # testset => number of results
        params_counts = defaultdict(int)   # param_id => number of results
        for r in res:
            testset_counts[r['result_name']] += 1
            params_counts[r['param_id']] += 1

        if args.check_errors:
            print('Results per testset:')
            for k, v in testset_counts.items():
                print(' ', k, v)

            runlog = load_runlog(fn + '.runlog')

            counts_lists = defaultdict(list)  # result cnt => list of param_ids
            for p in param_ids:
                counts_lists[params_counts[p]].append(p)

            good_count = max(counts_lists.keys())

            bad_runs = 0
            no_results = set()
            nan_results = set()
            runlog_missing = set()
            print('Runs with less than {} results:'.format(good_count))
            for k, v in counts_lists.items():
                if k != good_count:
                    bad_runs += len(v)
                    # print('  {}: {}'.format(k, ','.join([str(x) for x in v])))
                    if runlog is not None:
                        for p in v:
                            if p not in runlog:
                                runlog_missing.add(p)
                            else:
                                rp = runlog[p]
                                slurmlog_fn = os.path.join(
                                    rp['submit_dir'],
                                    'slurm-{}.out'.format(rp['slurm_id']))
                                err = get_logerror(slurmlog_fn)
                                if err is None:
                                    no_results.add(p)
                                elif err == 'fasttext::DenseMatrix::EncounteredNaNError':
                                    nan_results.add(p)
                                else:
                                    print('Unexpected error', err, slurmlog_fn)
                                    no_results.add(p)

            lp = len(param_ids)
            print('{} paramsets have missing runs out of {} ({:.2%})'.format(
                bad_runs, lp, bad_runs/lp))

            print_warnings(runlog_missing, 'missing entry in runlog', lp)
            print_warnings(no_results, 'no results files', lp)
            print_warnings(nan_results, 'NaN errors', lp)

            # log_fname = get_logfile(run_id)
            # err = get_logerror(log_fname)
            # if err is None:
            #     no_results.add(ni)
            # elif err == 'fasttext::DenseMatrix::EncounteredNaNError':
            #     nan_results.add(ni)
            # else:
            #     print('Unexpected error', err, log_fname)
            #     no_results.add(ni)

        testsets = testset_counts.keys()
        results.extend(res)

    df = pd.DataFrame.from_records(results)

    if args.output:
        df.to_csv(args.output)
        print('Wrote results to', args.output)

    if not args.check_errors:
        for testset in sorted(testsets):
            print('***', testset)

            print('Best {} results so far according to {} {}.'.format(
                args.N, args.opt, args.meas))
            if args.opt == 'max':
                dfs = df[df['result_name'] == testset].nlargest(args.N, args.meas)
            else:
                dfs = df[df['result_name'] == testset].nsmallest(args.N, args.meas)
            print(dfs.drop(columns='result_name'))
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+')
    parser.add_argument('--sqlite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output', '-O', type=str)
    parser.add_argument('--measure', type=str, default='P@5', required=False,
                        help='measure to optimize', dest='meas')
    parser.add_argument('-N', type=int, default=5,
                        help='number of top results to show')
    parser.add_argument('--opt', type=str,
                        choices=['max', 'min'], default='max', required=False,
                        help='whether to minimize or maximize the measure')
    parser.add_argument('--check_errors', action='store_true')
    parser.add_argument('--params', type=str)
    args = parser.parse_args()

    main(args)
