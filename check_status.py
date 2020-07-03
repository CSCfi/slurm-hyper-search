#!/usr/bin/env python3

from collections import defaultdict
import argparse
import datetime
import itertools
import numpy as np
import os
import re
import subprocess
import sys


def load_results(fn, measures=None, result_in_parts=False, safe_measure_names=False):
    results = []
    if measures:
        measures = measures.split(',')

    with open(fn, 'r') as fp:
        for i, line in enumerate(fp):
            line_ok = True
            m_idx = 0
            parts = line.rstrip().split('|')
            if not parts[0].isdigit():
                print('BAD LINE number {}:'.format(i), line)
                continue
            r_tags = {
                'param_id': int(parts[0]),
                'slurm_id': parts[2],
                'result_name': parts[3]
            }
            r_params = {}
            for p in parts[1].split('-'):
                if len(p) > 0:
                    pp = re.split(r'\s|=', p.rstrip())
                    if len(pp) == 1:
                        n = pp[0]
                        v = None
                    else:
                        n, v = pp
                        if v.isdigit():
                            v = int(v)
                        else:
                            try:
                                v = float(v)
                            except ValueError:
                                pass
                    r_params[n] = v
            r_measures = {}
            for p in parts[4:]:
                if len(p) > 0:
                    pp = p.split()
                    if len(pp) == 1:
                        if not measures:
                            print('ERROR: measure {} has no name in {}, '
                                  'please specify with --measures '
                                  'argument'.format(p, fn))
                            print('LINE number {}:'.format(i), line)
                            line_ok = False
                            break
                        n = measures[m_idx]
                        v = pp[0]
                        m_idx += 1
                    else:
                        n, v = pp

                    v = int(v) if v.isdigit() else float(v)
                    if safe_measure_names:
                        n = n.replace('@', '_at_')
                    r_measures[n] = v
            if line_ok:
                if result_in_parts:
                    r = (r_tags, r_params, r_measures)
                else:
                    r = {**r_tags, **r_params, **r_measures}
                results.append(r)
    return results


def load_params(fn):
    param_ids = []
    i = 1
    with open(fn, 'r') as fp:
        for line in fp:
            param_ids.append(i)
            i += 1
    return param_ids


def load_runlog(fn):
    if not os.path.isfile(fn):
        print("WARNING: no runlog found at", fn)
        return None

    runlog = []
    with open(fn, 'r') as fp:
        for line in fp:
            parts = line.rstrip().split('|')
            runlog.append({
                'param_id': int(parts[0]),
                'slurm_id': parts[1],
                'submit_dir': parts[2]
            })
    return runlog


def parse_mem(s):
    if not s:
        return 0
    if s.isnumeric():
        return float(s)
    val = float(s[:-1])
    ch = s[-1]
    if ch == 'K':
        return val*1024
    elif ch == 'M':
        return val*1024*1024
    elif ch == 'G':
        return val*1024*1024*1024


def load_slurm_data(slurm_ids):
    cmd = ['sacct', '-P', '-n', '--format', 'JobId,State,ElapsedRaw,MaxRSS',
           '-j', ','.join(si for si in slurm_ids if si[:6] != 'SERIAL')]

    res = subprocess.run(cmd, stdout=subprocess.PIPE)
    output = res.stdout.decode('utf-8')

    data = {}
    max_maxrss = 0
    max_elapsed = 0
    for line in output.split('\n'):
        if line == '':
            continue
        parts = line.split('|')

        slurm_id = parts[0].split('.', 1)[0]  # e.g. 850049_1.3 => 850049_1

        elapsed = int(parts[2]) if len(parts[2]) > 0 else 0
        maxrss = parse_mem(parts[3])

        if slurm_id not in data:
            data[slurm_id] = {
                'status': parts[1],
                'elapsed': elapsed,
                'maxrss': maxrss
            }
        else:
            if elapsed > data[slurm_id]['elapsed']:
                data[slurm_id]['elapsed'] = elapsed
            if maxrss > data[slurm_id]['maxrss']:
                data[slurm_id]['maxrss'] = maxrss

        if elapsed > max_elapsed:
            max_elapsed = elapsed
        if maxrss > max_maxrss:
            max_maxrss = maxrss

    return data, max_elapsed, max_maxrss


def get_logerror(fn):
    if not os.path.isfile(fn):
        print('WARNING: could not find log file {}!  Try specifying --log_dir,'
              ' or --skip_logs if you don\'t want to analyze logs.'.format(fn))
        return None

    with open(fn, 'r') as fp:
        for line in fp:
            m = re.search("terminate called after throwing an instance of "
                          "'(.*)'", line)
            if m:
                return m.group(1)
    return None


def indices_to_str(ids):
    ids = list(ids)
    to_print = []
    range_start = 0
    for i in range(1, len(ids)+1):
        if i == len(ids) or ids[i] != ids[i-1]+1:
            if range_start < i-1:
                to_print.append('{}-{}'.format(ids[range_start], ids[i-1]))
            else:
                to_print.append(str(ids[i-1]))
            range_start = i
    return ','.join(to_print)


def print_warnings(res, msg, tot):
    if res:
        n = len(res)
        print('WARNING: the following param indices {}, '
              '{}/{} = {:.2%}:'.format(msg, n, tot, n/tot))
        for k, g in itertools.groupby(sorted(res),
                                      key=lambda x: (x-1)//1000):
            offset = k*1000
            print('[OFFSET: {}]'.format(offset), end=' ')
            #print(','.join(str(x-offset) for x in g))
            print(indices_to_str(x-offset for x in g))


def format_mem(num):
    for unit in ['', 'K', 'M', 'G']:
        if num < 1024.0:
            return "{:.4} {}".format(num, unit)
        num /= 1024.0
    return "{:.4} T".format(num)


def format_time(secs):
    return datetime.timedelta(seconds=float(secs))


def main(args):
    in_dir = args.input

    params_fn = os.path.join(in_dir, 'params')
    param_ids = load_params(params_fn)
    print('Read {} which contained {} sets of parameters.'.format(
        params_fn, len(param_ids)))

    runlog_fn = os.path.join(in_dir, 'runlog')
    runlog = load_runlog(runlog_fn)
    print('Read {} which contained data on {} runs.'.format(
        runlog_fn, len(runlog)))

    results = []
    results_fn = os.path.join(in_dir, 'results')

    results = load_results(results_fn, args.measures)
    print('Read {} which contained {} results.'.format(results_fn,
                                                       len(results)))

    params_results = defaultdict(list)  # dict: param_id -> list of results
    testsets = set()
    slurm_main_ids = set()
    for r in results:
        params_results[r['param_id']].append(r)
        testsets.add(r['result_name'])
        slurm_main_ids.add(r['slurm_id'].split('_', 1)[0])

    error_statuses = {}  # dict: slurm_id -> error status
    if not args.skip_slurm:
        # FIXME: used slurm_data[slurm_id]['status'] for something...
        slurm_data, max_elapsed, max_maxrss = load_slurm_data(slurm_main_ids)
        times = []
        for slurm_id, data in slurm_data.items():
            s = data['status']
            if s != 'COMPLETED':
                error_statuses[slurm_id] = s
            times.append(data['elapsed'])

        if args.time_histogram:
            N = len(times)
            print('Elapsed times (histogram):')
            bins = list(range(0, int(max_elapsed), args.time_histogram*60))
            bins[-1] = max_elapsed
            hist, bin_edges = np.histogram(times, bins=bins)
            cum = 0.0
            for i, c in enumerate(hist):
                bl = int(c/N*100)
                cum += c
                print(' [{} - {}]: {:7.2%} ({:7.2%}) {}'.format(
                    format_time(bin_edges[i]),
                    format_time(bin_edges[i+1]),
                    c/N, cum/N, '*' * bl))
            print('Average elapsed time: {}'.format(
                format_time(np.mean(times))))

    params_runs = defaultdict(list)  # dict: param_id -> list of runs
    for r in runlog:
        params_runs[r['param_id']].append(r)

    pids_with_unknown_errors = []
    pids_with_nan_errors = []
    pids_without_runs = []
    pids_with_error_status = defaultdict(list)  # dict: error status -> pids
    for pid in param_ids:
        res = params_results[pid]
        if len(res) == 0:
            runs = params_runs[pid]
            if len(runs) == 0:
                pids_without_runs.append(pid)
            else:
                slurm_ids = [r['slurm_id'] for r in runs]
                if len(runs) > 1:
                    print('WARNING: paramset {} has more than one run: '.
                          format(pid), ', '.join(slurm_ids))
                slurm_id = slurm_ids[-1]

                if args.skip_logs:
                    pids_with_unknown_errors.append(pid)
                else:
                    slurmlog_fn = os.path.join(args.log_dir, 'slurm-{}.out'
                                               .format(slurm_id))
                    err = get_logerror(slurmlog_fn)
                    if err == 'fasttext::DenseMatrix::EncounteredNaNError':
                        pids_with_nan_errors.append(pid)
                    elif slurm_id in error_statuses:
                        pids_with_error_status[error_statuses[slurm_id]].append(pid)
                    else:
                        pids_with_unknown_errors.append(pid)
                        if err is not None:
                            print('WARNING: Unknown error', err, slurmlog_fn)
        else:  # len(res) > 0
            slurm_ids = set([r['slurm_id'] for r in res])
            pid_testsets = [r['result_name'] for r in res]
            pid_testsets_set = set(pid_testsets)
            if pid_testsets_set != testsets:
                print('WARNING: paramset {} ({}) is missing some results:'.
                      format(pid, ','.join(slurm_ids)),
                      ','.join(pid_testsets_set))
                for sid in slurm_ids:
                    if sid in error_statuses:
                        pids_with_error_status[error_statuses[sid]].append(pid)
            elif len(pid_testsets) != len(testsets):
                print('WARNING: paramset {} has wrong number of results: {}'.
                      format(pid, len(pid_testsets)))

    N = len(param_ids)
    print_warnings(pids_without_runs, 'no runs', N)
    print_warnings(pids_with_nan_errors, 'NaN errors', N)
    for error_status, pids in pids_with_error_status.items():
        print_warnings(pids, 'have {} status'.format(error_status), N)
    print_warnings(pids_with_unknown_errors, 'unknown errors', N)

    if not args.skip_slurm:
        print()
        print('Max RSS:', format_mem(max_maxrss))
        print('Max elapsed:', format_time(max_elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='directory with the results file')
    parser.add_argument('--log_dir', type=str, default='./')
    parser.add_argument('--skip_logs', action='store_true')
    parser.add_argument('--skip_slurm', action='store_true')
    parser.add_argument('--time_histogram', type=int)
    parser.add_argument('--measures', type=str,
                        help='names of measures if missing from results, '
                        'e.g., --measures=P@1,P@3,P@5')
    args = parser.parse_args()

    main(args)
