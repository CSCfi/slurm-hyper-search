#!/usr/bin/env python3

import argparse
import os
import pandas as pd

from check_status import load_results
from functools import reduce


def main(args):
    in_dir = args.input

    results = []
    results_fn = os.path.join(in_dir, 'results')

    results = load_results(results_fn, args.measures)
    print('Read {} which contained {} results.'.format(
        results_fn, len(results)))

    df = pd.DataFrame.from_records(results)

    if args.output:
        df.to_csv(args.output)
        print('Wrote results to', args.output)

    best_per_testset = {}
    result_names = df.result_name.unique()
    for testset in sorted(result_names):
        idx = df['result_name'] == testset
        dft = df[idx]
        print()
        print('# {}, {} results'.format(testset, len(dft)))

        print('Best {} results so far according to {} {}.'.format(
            args.N, args.opt, args.meas))
        if args.opt == 'max':
            dfs = dft.nlargest(args.N, args.meas)
        else:
            dfs = dft.nsmallest(args.N, args.meas)
        best = dfs.iloc[0][args.meas]
        best_per_testset[testset] = best

        print(dfs.drop(columns='result_name'))

    if args.overall_score:
        def aggregate_results(x):
            xm = x.mean()
            v = {}
            for rn in result_names:
                v[rn] = 0.0
            for index, row in x.iterrows():
                rn = row['result_name']
                assert rn in result_names
                v[rn] = row[args.meas]/best_per_testset[rn]
                # if rn.startswith('combo'):
                #     v[rn] *= 1.2
            assert len(v) == len(result_names)
            xm['overall_score'] = reduce(lambda x, y: x * y, v.values())
            return xm.drop(columns='slurm_id')

        assert args.opt == 'max', 'min not implemented yet :-)'
        print()
        print('# Best overall score.')
        res = df.groupby(['param_id']).apply(aggregate_results)
        ress = res.nlargest(args.N, 'overall_score')
        print(ress)

        print()
        print('# Individual results for the five best ones')
        # best_param_id = ress.iloc[0]['param_id']
        # print(df[df['param_id'] == best_param_id])
        for i in range(5):
            best_param_id = ress.iloc[i]['param_id']
            print(df[df['param_id'] == best_param_id])

        print("\n# Best {} per testset".format(args.meas))
        for rn in result_names:
            print('{:30} {:.5}'.format(rn, best_per_testset[rn]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='directory with the results file')
    parser.add_argument('--output', '-O', type=str)
    parser.add_argument('--measure', type=str, default='P@5', required=False,
                        help='measure to optimize', dest='meas')
    parser.add_argument('-N', type=int, default=5,
                        help='number of top results to show')
    parser.add_argument('--opt', type=str,
                        choices=['max', 'min'], default='max', required=False,
                        help='whether to minimize or maximize the measure')
    parser.add_argument('--measures', type=str,
                        help='names of measures if missing from results, '
                        'e.g., --measures=P@1,P@3,P@5')
    parser.add_argument('--overall_score', action='store_true')
    args = parser.parse_args()

    main(args)
