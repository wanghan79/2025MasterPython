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

df_gvp_pred_at = ct.read_res_pred(din_preferences +'gvp_100_m_at_1646945484.3030427_8_220711.csv')

mut_pos = ['L47', 'D51', 'I52', 'R54', 'L55', 'F73', 'R77', 'E79', 'A80', 'R81']
mut_pos_m1 = [m[0] + str(int(m[1:])+1) for m in mut_pos]
start = timeit.default_timer()

n_sample = 500 # number of variants to sample
t_range = [0.1, 0.5,0.7,1,1.5,2,2.25,2.5,2.75,3,4,5]
for t in t_range: # range of sampling temperatures
    sampled_mutkeys = ct.sample_coves(df_gvp_pred_at, mut_pos_m1, n_sample = n_sample, t=t)
    with open(dout_at_sample + f'/gvp_100_m_RES_1646945484_3030427_8_220711_samples_t{t}_n{n_sample}.csv', 'w') as fout:
        for m in sampled_mutkeys:
            fout.write(m + '\n')

end = timeit.default_timer()

print(f'It took {end-start} seconds to sample {n_sample*len(t_range)} samples from CoVES.')