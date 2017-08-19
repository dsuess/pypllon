#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import, division, print_function

import click

import h5py
import numpy as np
from os import path

SETUP_LABELS = ['GAUSS', 'RECR', 'RRECR']


def load_power_reference(h5id, key, datadir, fileformat):
    key = key.replace('Rad', 'rad')
    fname = path.join(datadir, fileformat(h5id, key))
    return np.asscalar(np.loadtxt(fname))


def add_power_reference(h5id, group, datadir):
    for key, count in group['RAWCOUNTS'].items():
        fileformat = lambda h5id, key: 'Ref_{}_{}.dat'.format(h5id, key)
        count.attrs['POWER_REF'] = load_power_reference(h5id, key, datadir, fileformat)
    for key, count in group['SINGLE_COUNTS'].items():
        fileformat = lambda h5id, key: 'Ref_{}_S5_0{}.dat'.format(h5id, int(key) + 1)
        count.attrs['POWER_REF'] = load_power_reference(h5id, key, datadir, fileformat)


@click.command()
@click.argument('h5file')
@click.argument('datadir')
def main(h5file, datadir):
    _, h5name = path.split(h5file)
    h5id, _ = path.splitext(h5name)

    with h5py.File(h5file, 'a') as buffile:
        for label in SETUP_LABELS:
            try:
                add_power_reference(h5id, buffile[label], datadir)
            except KeyError:
                print('Could not open label {} in {}'.format(label, h5file))


if __name__ == '__main__':
    main()
