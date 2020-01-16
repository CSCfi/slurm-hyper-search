#!/usr/bin/env python3

import argparse
import pandas as pd
import re
from collections import defaultdict


def load_sqlite(fn):
    results = []

    import sqlite3
    conn = sqlite3.connect(fn)
    c = conn.cursor()
    c.execute('SELECT results.*, params.* FROM results '
              'LEFT JOIN params ON param_id=params.rowid')
    col_names = [re.sub(r'_(\d)', '@\\1', x[0]) for x in c.description]
    while True:
        row = c.fetchone()
        if row is None:
            break
        r = dict(zip(col_names, row))
        del r['param_id']
        results.append(r)
    conn.close()

    return results


def load_singletext(fn):
    results = []
    with open(fn, 'r') as fp:
        for line in fp:
            parts = line.rstrip().split('|')
            r = {
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


def main(args):
    results = []

    for fn in args.input:
        if fn.endswith('.db') or args.sqlite:
            res = load_sqlite(fn)
        else:
            res = load_singletext(fn)

        print('Read {} which contained {} results as follows:'.format(fn, len(res)))
        d = defaultdict(int)
        for r in res:
            d[r['result_name']] += 1
        for k, v in d.items():
            print(' ', k, v)
        testsets = d.keys()
        results.extend(res)

    df = pd.DataFrame.from_records(results)

    if args.output:
        df.to_csv(args.output)
        print('Wrote results to', args.output)
    # print(df)
    #    import ipdb; ipdb.set_trace()

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
    parser.add_argument('--output', '-O', type=str)
    parser.add_argument('--measure', type=str, default='P@5', required=False,
                        help='measure to optimize', dest='meas')
    parser.add_argument('-N', type=int, default=5,
                        help='number of top results to show')
    parser.add_argument('--opt', type=str,
                        choices=['max', 'min'], default='max', required=False,
                        help='whether to minimize or maximize the measure')
    args = parser.parse_args()

    main(args)
