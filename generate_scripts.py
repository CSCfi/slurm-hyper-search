#!/usr/bin/env python3

import argparse
import json
import sys

from string import Template


# Default delimiter $ gets confused with bash variables...
class MyTemplate(Template):
    delimiter = '%'


def main(args):
    template_fname = args.template
    template = open(template_fname, 'r').read()
    t = MyTemplate(template)

    psets = json.load(open(args.params_json, 'r'))

    counter = 0
    for p in psets:
        try:
            s = t.substitute(p)
            output_fname = 'slurm-{}.sh'.format(counter)
            counter += 1
            with open(output_fname, 'w') as fp:
                fp.write(s)
            print('Wrote', output_fname)
        except KeyError as key:
            print('ERROR: parameter {} in template file {} not specified in:\n'
                  '       {}'.format(key, template_fname, p), file=sys.stderr)
            sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('params_json', type=str)
    parser.add_argument('--template', type=str, default='slurm.sh.template')
    args = parser.parse_args()

    main(args)
