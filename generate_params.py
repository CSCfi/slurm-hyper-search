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
    'minn': [3, 4, 5],
    'maxn': [5, 6, 7],
}

# Note: you can also specify probability distributions, e.g.,
# import scipy
#     'C': scipy.stats.expon(scale=100),
#     'gamma': scipy.stats.expon(scale=.1),


def main(args):
    # directory structure:
    # args.dir/param-id/
    #                   params -- parameters in form: -dim $dim -lr $lr ...
    #                   result -- output value to optimize, e.g., loss

    out_dir = args.dir

    n = 0
    if os.path.isdir(out_dir):
        max_n = max(int(x) for x in os.listdir(out_dir) if x.isdigit())
        n = max_n + 1
        print('Found existing runs up to {}, starting from {}.'.
              format(max_n, n))

    rng = np.random.RandomState(args.seed)
    ps = ParameterSampler(space, n_iter=args.n, random_state=rng)

    for i, p in enumerate(ps):
        p_dir = os.path.join(out_dir, str(n + i))
        assert not os.path.exists(p_dir)
        os.makedirs(p_dir)

        p_fname = os.path.join(p_dir, 'params')
        with open(p_fname, 'w') as fp:
            p_str = ' '.join(['-{} {}'.format(k, v) for k, v in p.items()])
            fp.write(p_str)
        print('[{}]: {}'.format(p_dir, p_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a set of random hyper parameters')
    parser.add_argument('dir', type=str,
                        help='directory to keep parameter and results files')
    parser.add_argument('n', type=int,
                        help='number of hyper parameter sets to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for deterministic runs')
    args = parser.parse_args()

    main(args)
