#!/usr/bin/env python3

import argparse
import os
import mlflow
from tqdm import tqdm

from check_status import load_results


def main(args):
    in_dir = args.input

    results = []
    results_fn = os.path.join(in_dir, 'results')
    results = load_results(results_fn, result_in_parts=True,
                           safe_measure_names=True)
    print('Read {} which contained {} results.'.format(
        results_fn, len(results)))

    # Use directory name as experiment name
    mlflow.set_experiment(in_dir)

    for (tags, params, measures) in tqdm(results):
        with mlflow.start_run():
            mlflow.set_tags(tags)
            mlflow.log_params(params)
            mlflow.log_metrics(measures)

    print('Stored MLflow data to', mlflow.get_tracking_uri())


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Log results to MLflow')
    parser.add_argument('input', type=str,
                        help='directory with the results file')
    args = parser.parse_args()

    main(args)
