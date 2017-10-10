#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function

from collections import defaultdict

import numpy as np
import pandas as pd
from sympy.parsing.mathematica import mathematica

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda iterator: iterator



def parse_ctx(fname):
    """Reads the ctx files and returns a dict containing all the values.

    :param fname: Path to the file to be parsed
    :returns: Dict with the data from ctx file

    """
    with open(fname, 'r') as inf:
        split_lines = (line.strip().split(':', 1) for line in inf.readlines()
                       if ':' in line)
        res = dict()
        counter = defaultdict(lambda: -1)

        for key, val in split_lines:
            counter[key] = counter[key] + 1
            res[key + (str(counter[key]) if counter[key] > 0 else '')] = \
                eval(val.replace('null', 'None'))
        return res


def load_complex_array(fname):
    def convert(s):
        s = s.replace(b'(', b'').replace(b')', b'')
        s = s.replace(b'^', b'10^')
        s = s.replace(b'e', b'*10^')
        return complex(mathematica(s.decode('utf-8')))

    with open(fname, 'r') as buf:
        columns = len(buf.readline().strip().split())
    converter = {i: convert for i in range(columns)}
    return np.loadtxt(fname, converters=converter, dtype=complex)


def load_simdata(infile):
    columns = {'tmat_name': object,
               'dim': int,
               'measurements': int,
               'target': object,
               'recons': object,
               'approx_err': object}
    df = pd.DataFrame(columns=columns.keys())
    for col, dtype in columns.items():
        df[col] = df[col].astype(dtype)

    for dimgroup in tqdm(infile.values()):
        dim = dimgroup.attrs['DIM']
        tmats = {name: value.value
                 for name, value in dimgroup['TARGETS'].items()}
        recov_groups = (group for name, group in dimgroup.items()
                        if name.startswith('RECOV_'))

        for recov_group in recov_groups:
            nr_measurements = recov_group.attrs['NR_MEASUREMENTS']
            for tmat_name in tmats:
                try:
                    recov = recov_group[tmat_name].value
                    approx_err = recov_group[tmat_name].attrs['errs']
                except KeyError:
                    recov = np.nan
                    approx_err = np.nan
                df.loc[len(df)] = [tmat_name, dim, nr_measurements,
                                   tmats[tmat_name], recov, approx_err]

    return df
