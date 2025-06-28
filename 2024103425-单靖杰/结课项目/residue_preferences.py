import torch
import time
import timeit

import sys
sys.path.append('./res/gvp-pytorch/')
import importlib
import gvpTools
importlib.reload(gvpTools)
import gvpTools as gt
from gvpTools import get_gvp_res_prefs




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import importlib
import timeit

sys.path.append('../src/')
import covesTools
importlib.reload(covesTools)
import covesTools as ct

### file paths
din_preferences = '../data/coves/preferences/'
dout_at_sample = '../data/coves/samples/at/'
dout_gfp_sample = '../data/coves/samples/gfp/'

dout_test_sample = '../data/coves/samples/at/test_to_delete/'


fin_gfp_exp = '../data/coves/scores/all_scores/df_all_gfp.csv'





device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = float(time.time())
print(device)

pref_dout = '../data/coves/preferences/' # output folder for the residue preferences
res_weight_fin = '../data/coves/res_weights/RES_1646945484.3030427_8.pt' # the trained weights for the GVP-RES model
pdb_dir = '../data/coves/pdbs/' # the directory of PDB files

wt_at = 'MANVEKMSVAVTPQQAAVMREAVEAGEYATASEIVREAVRDWLAKRELRHDDIRRLRQLWDEGKASGRPEPVDFDALRKEARQKLTEVPPNGR'
n_ave = 100

start = timeit.default_timer()
df_result = get_gvp_res_prefs(wt_seq=wt_at,
                                protein_name ='at',
                                chain_number='A',
                                pdb_din=pdb_dir+'at/ta/',
                                lmdb_dout=pdb_dir+'at/lmdb_at_all/',
                                model_weight_path = res_weight_fin,
                                dout = pref_dout,
                                n_ave = n_ave
                             )
end = timeit.default_timer()

print(f'Computing residue preferences for {len(wt_at)} amino acids took {end-start} seconds.')



