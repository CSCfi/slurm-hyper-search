#!/usr/bin/env python3

import argparse
import numpy as np
import sqlite3


def main(args):
    conn = sqlite3.connect(args.database)
    c = conn.cursor()

    cmd = 'SELECT * FROM params WHERE rowid={}'.format(args.param_id)
    c.execute(cmd)
    col_names = [x[0] for x in c.description]
    params = c.fetchone()
    conn.close()

    print(' '.join(['-{} {}'.format(k, v) for k, v in zip(col_names, params)]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch previously generated hyper parameters')
    parser.add_argument('database', type=str, help='sqlite3 database file')
    parser.add_argument('param_id', type=int, help='rowid of parameter set')

    args = parser.parse_args()
    main(args)
