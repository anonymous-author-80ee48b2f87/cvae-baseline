#!/usr/bin/env python2

from __future__ import print_function

import sys

# from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import molecule_vae

# 0. Constants
nb_smiles = 5000
chunk_size = 5000
encode_times = 10
decode_times = 100


# 1. load the test smiles
smiles_file = '../../Dropbox/data/250k_rndm_zinc_drugs_clean_kekulised.smi'
smiles = [line.strip() for index, line in zip(xrange(nb_smiles), open(smiles_file).xreadlines())]


def reconstruct_single(model, smiles):
    print('a chunk starts...')
    decode_result = []

    chunk = smiles
    chunk_result = [[] for _ in range(len(chunk))]
    for _encode in range(encode_times):
        z1 = model.encode(chunk)
        this_encode = []
        encode_id, encode_total = _encode + 1, encode_times
        for _decode in tqdm(list(range(decode_times)),
                'encode %d/%d decode' % (encode_id, encode_total)
            ):
            _result = model.decode(z1)
            for index, s in enumerate(_result):
                chunk_result[index].append(s)

    decode_result.extend(chunk_result)
    assert len(decode_result) == len(smiles)
    return decode_result

def reconstruct(model, smiles):
    '''
    chunk_result = Parallel(n_jobs=1)(
        delayed(reconstruct_single)(model, smiles[chunk_start: chunk_start + chunk_size])
        for chunk_start in range(0, len(smiles), chunk_size)
    )
    '''

    chunk_result = [reconstruct_single(model, smiles[chunk_start: chunk_start + chunk_size])
        for chunk_start in range(0, len(smiles), chunk_size)
    ]

    decode_result = [_1 for _0 in chunk_result for _1 in _0]
    assert len(decode_result) == len(smiles)
    return decode_result

def save_decode_result(decode_result, filename):
    with open(filename, 'w') as fout:
        for s, cand in zip(smiles, decode_result):
            print(','.join([s] + cand), file=fout)

def cal_accuracy(decode_result):
    accuracy = [sum([1 for c in cand if c == s]) * 1.0 / len(cand) for s, cand in zip(smiles, decode_result)]
    return (sum(accuracy) * 1.0 / len(accuracy))

def main():
    char_weights = '../../Dropbox/model/zinc_vae_str_kekulised_L56_E100_val.hdf5'
    accuracy_save_file = char_weights + '.reconstruct_accuracy.txt'
    decode_result_save_file = char_weights + '.reconstruct_decode_result.txt'
    model = molecule_vae.ZincCharacterModel(char_weights)

    decode_result = reconstruct(model, smiles)
    accuracy = cal_accuracy(decode_result)

    print('accuracy:', accuracy)

    save_result = True
    if save_result:
        with open(accuracy_save_file, 'w') as fout:
            print('accuracy:', accuracy, file=fout)

        save_decode_result(decode_result, decode_result_save_file)

import pdb, traceback, sys, code

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
