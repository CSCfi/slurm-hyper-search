#!/usr/bin/env python3

import argparse
import os
import pandas as pd

from check_status import load_results


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

    for testset in sorted(df.result_name.unique()):
        dft = df[df['result_name'] == testset]
        print()
        print('# {}, {} results'.format(testset, len(dft)))

        print('Best {} results so far according to {} {}.'.format(
            args.N, args.opt, args.meas))
        if args.opt == 'max':
            dfs = dft.nlargest(args.N, args.meas)
        else:
            dfs = dft.nsmallest(args.N, args.meas)
        print(dfs.drop(columns='result_name'))


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
    args = parser.parse_args()

    main(args)
