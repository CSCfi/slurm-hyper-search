#!/usr/bin/env python3

import argparse
import json
import numpy as np

from sklearn.model_selection import ParameterSampler

space = {
    'dim': np.arange(50, 1001, step=10),
    'lr': np.around(np.geomspace(0.01, 5, num=20), decimals=6),
    'epoch': np.arange(5, 121, step=5),
    'minCount': [1, 2, 3, 4, 5],
    'ngrams': [1, 2, 3],
    'minn': [3, 4, 5],
    'maxn': [5, 6, 7],
}

# Note: you can also specify probability distributions, e.g.,
# import scipy
#     'C': scipy.stats.expon(scale=100),
#     'gamma': scipy.stats.expon(scale=.1),


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        else:
            return super(NpEncoder, self).default(obj)


def main(args):
    rng = np.random.RandomState(args.seed)
    ps = ParameterSampler(space, n_iter=args.n, random_state=rng)

    print(json.dumps(list(ps), indent=2, cls=NpEncoder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    main(args)
