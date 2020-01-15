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
    # directory structure:
    # args.dir/param-id/
    #                   params  -- parameters in form: -dim $dim -lr $lr ...
    #                   results -- results to optimize on

    out_dir = args.output

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
            fp.write(p_str + '\n')
        print('[{}]: {}'.format(p_dir, p_str))


def sql_type(x):
    if isinstance(x, (list, np.ndarray)):
        x = x[0]
    t = 'BLOB'
    if isinstance(x, (np.float, np.float64)):
        t = 'REAL'
    elif isinstance(x, (np.int, np.int64)):
        t = 'INTEGER'
    elif isinstance(x, str):
        t = 'TEXT'
    return t


def sql_format(x):
    if isinstance(x, str):
        return '"' + x + '"'
    else:
        return str(x)


def main_sqlite(args):
    import sqlite3

    conn = sqlite3.connect(args.output)
    c = conn.cursor()

    col_names = list(space.keys())
    cols = ["{} {}".format(n, sql_type(space[n])) for n in col_names]
    cmd = 'CREATE TABLE IF NOT EXISTS params ({})'.format(', '.join(cols))
    print('[SQL]', cmd)
    c.execute(cmd)
    conn.commit()

    cmd = 'SELECT max(rowid) FROM params'
    c.execute(cmd)
    next_id = 0
    max_id = c.fetchone()[0]

    if max_id is not None:
        next_id = max_id+1
        print('Found existing runs up to {}, starting from {}.'.
              format(max_id, next_id))

    rng = np.random.RandomState(args.seed)
    ps = ParameterSampler(space, n_iter=args.n, random_state=rng)

    for p in ps:
        values = [sql_format(p[n]) for n in col_names]
        cmd = 'INSERT INTO params (rowid,{}) VALUES ({},{})'.format(
            ','.join(col_names), str(next_id), ','.join(values))
        print('[SQL]', cmd)
        c.execute(cmd)
        next_id += 1
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a set of random hyper parameters')
    parser.add_argument('output', type=str,
                        help='directory or (sqlite) file to keep parameter and results files')
    parser.add_argument('n', type=int,
                        help='number of hyper parameter sets to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed for deterministic runs')
    parser.add_argument('--sqlite', action='store_true',
                        help='use sqlite')

    args = parser.parse_args()
    if args.sqlite:
        main_sqlite(args)
    else:
        main(args)
