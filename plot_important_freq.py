import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from scipy import stats
import argparse



parser = argparse.ArgumentParser('plot frequency of being important')
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--iterations',help='number of iterations for each run',type=int)
parser.add_argument('--runs',help='number of runs',type=int)
parser.add_argument('--vs_freq',help='the number of iterations between two variable selection steps',type=int)
#parser.add_argument('--T_max',type=int)
#parser.add_argument('--best_value',type=float,default=None)
#parser.add_argument('--wct',action='store_true')
#parser.add_argument('--cput',action='cput')
args = parser.parse_args()


object_dim = np.load(args.input_path+"/F_chosen_1.npy").shape[1]
chosen_stats = np.zeros(object_dim)

tot_freq = int(args.iterations/args.vs_freq)

for i in range(args.runs):
    F_chosen = np.load(args.input_path+"/F_chosen_"+str(i+1)+".npy")[:tot_freq]
    n = F_chosen.shape[0]
    for j in range(n):
        chosen_stats[F_chosen[j]]+=1

fig,ax = plt.subplots(figsize=(8, 6))
plt.bar(range(1,object_dim+1),chosen_stats)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
plt.xlabel("Variable index", fontsize=20)
plt.ylabel("Total frequency of being important", fontsize=14)
#plt.show()
plt.savefig(args.output_path+"/F_chosen_bar.png")
plt.close()
