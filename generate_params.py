#!/usr/bin/env python3

import argparse
import os
import numpy as np
from sklearn.model_selection import ParameterSampler, ParameterGrid

from space_conf import space


def main(args):
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    fn = os.path.join(out_dir, 'params')

    if args.n.lower() == 'all':
        ps = ParameterGrid(space)
    else:
        n = int(args.n)
        rng = np.random.RandomState(args.seed)
        ps = ParameterSampler(space, n_iter=n, random_state=rng)

    with open(fn, 'a') as fp:
        for p in ps:
            p_str = ' '.join([args.format.format(name=k, value=v) for k, v in p.items()])
            if args.extra:
                p_str = args.extra + ' ' + p_str
            fp.write(p_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a set of hyper parameters')
    parser.add_argument('output', type=str,
                        help='output directory')
    parser.add_argument('n', type=str,
                        help='random search: number of hyper parameter sets '
                        'to sample, for grid search: set to "all"')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for deterministic runs')
    parser.add_argument('--format', type=str, default='--{name}={value}',
                        help='format for parameter arguments, default is '
                        '--{name}={value}')
    parser.add_argument('--extra', type=str, help='Extra arguments to add')

    args = parser.parse_args()
    main(args)
