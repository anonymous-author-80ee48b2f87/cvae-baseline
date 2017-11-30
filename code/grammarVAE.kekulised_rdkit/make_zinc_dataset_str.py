import numpy as np
import pdb
from models.utils import many_one_hot
import h5py

f = open('../../Dropbox/data/250k_rndm_zinc_drugs_clean_kekulised_rdkit.smi','r')

L = []
chars = ['C', '(', ')', '1', '2', '=', 'O', 'N', '3', 'F', '[', '@', 'H', ']', '-', '#', 'S', 'l', '+', 'B', 'r', '/', '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']
DIM = len(chars)
for line in f:
    line = line.strip()
    L.append(line)
f.close()

count = 0
MAX_LEN = 120
OH = np.zeros((249456,MAX_LEN,DIM))
for chem in L:
    indices = []
    for c in chem:
        indices.append(chars.index(c))
    if len(indices) < MAX_LEN:
        indices.extend((MAX_LEN-len(indices))*[DIM-1])
    OH[count,:,:] = many_one_hot(np.array(indices), DIM)
    count = count + 1
f.close()
h5f = h5py.File('../../Dropbox/data/zinc_str_kekulised_rdkit_dataset.h5','w')
h5f.create_dataset('data', data=OH)
h5f.create_dataset('chr',  data=chars)
h5f.close()
