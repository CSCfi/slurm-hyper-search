#!/usr/bin/env python3

import argparse
import os
import sys


def main(args):
    out_dir = args.dir

    if not os.path.isdir(out_dir):
        print("ERROR: Directory {} doesn't exist!".format(out_dir))
        return 1

    no_results = set()
    empty_results = set()
    results = {}
    best_n = None

    for n in os.listdir(out_dir):
        if not n.isdigit():
            break
        ni = int(n)
        results_file = os.path.join(out_dir, n, 'results')
        if not os.path.isfile(results_file):
            no_results.add(ni)
        elif os.path.getsize(results_file) == 0:
            empty_results.add(ni)
        else:
            with open(results_file, 'r') as fp:
                for line in fp:
                    m_name, m_value = line.split()
                    if m_name == args.measure:
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

    print('Best result so far: {} = {} in run {}.'.format(
        args.measure, results[best_n], best_n))
    print('Best parameters:')
    print(' ', open(os.path.join(out_dir, str(best_n), 'params')).read())

    if empty_results:
        print('WARNING: the following runs have empty results files:',
              ','.join(sorted(str(x) for x in empty_results)))

    if no_results:
        print('WARNING: the following runs have no results files:',
              ','.join(sorted(str(x) for x in no_results)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str,
                        help='directory with parameter and results files')
    parser.add_argument('measure', type=str, help='measure to optimize')
    parser.add_argument('--opt', type=str,
                        choices=['max', 'min'], default='max', required=False,
                        help='whether to minimize or maximize the measure')

    args = parser.parse_args()

    sys.exit(main(args))
