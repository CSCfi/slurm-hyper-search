#!/usr/bin/env python3

import argparse
import sqlite3
import sys


def sql_type(x):
    return 'REAL' if '.' in x else 'INTEGER'


def sql_format(x):
    return x.replace('@', '_')


def main(args):
    results = [line.rstrip().split() for line in sys.stdin]
    res_names, res_values = zip(*results)
    res_names = list(map(sql_format, res_names))

    conn = sqlite3.connect(args.database)
    c = conn.cursor()

    cols = ["{} {}".format(k, sql_type(v))
            for k, v in zip(res_names, res_values)]

    std_cols = ["param_id INTEGER", "result_name TEXT", "slurm_id TEXT"]

    cmd = 'CREATE TABLE IF NOT EXISTS results ({})'.format(
        ', '.join(std_cols + cols))
    print('[SQL]', cmd)
    c.execute(cmd)
    conn.commit()

    cmd = '''INSERT INTO results (param_id,result_name,slurm_id,{})
             VALUES ({},"{}","{}",{})'''.format(
                 ','.join(res_names),
                 args.param_id, args.result_name, args.slurm_id,
                 ','.join(res_values))
    print('[SQL]', cmd)
    c.execute(cmd)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fetch previously generated hyper parameters')
    parser.add_argument('database', type=str, help='sqlite3 database file')
    parser.add_argument('param_id', type=int, help='rowid of parameter set')
    parser.add_argument('--slurm_id', type=str, help='job name', default=None)
    parser.add_argument('--result_name', type=str, default=None,
                        help='identifier for the result, e.g. test set name')

    args = parser.parse_args()
    main(args)
