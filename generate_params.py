#!/usr/bin/env python3

import argparse
import os
import numpy as np

from sklearn.model_selection import ParameterSampler

space = {
    'dim': np.arange(50, 1001, step=10),
    'lr': np.around(np.geomspace(0.01, 5, num=20), decimals=6),
    'epoch': np.arange(5, 121, step=5),
    'minCount': [1, 2, 3, 4, 5],
    'wordNgrams': [1, 2, 3],
    'minn': [2, 3, 4],
    'maxn': [5, 6, 7],
}

# Note: you can also specify probability distributions, e.g.,
# import scipy
#     'C': scipy.stats.expon(scale=100),
#     'gamma': scipy.stats.expon(scale=.1),


def main(args):
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    fn = os.path.join(out_dir, 'params')

    rng = np.random.RandomState(args.seed)
    ps = ParameterSampler(space, n_iter=args.n, random_state=rng)

    with open(fn, 'a') as fp:
        for p in ps:
            p_str = ' '.join(['-{} {}'.format(k, v) for k, v in p.items()])
            fp.write(p_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a set of random hyper parameters')
    parser.add_argument('output', type=str,
                        help='output directory')
    parser.add_argument('n', type=int,
                        help='number of hyper parameter sets to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for deterministic runs')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args)
