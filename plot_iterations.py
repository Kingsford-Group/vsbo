import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from scipy import stats
import argparse


parser = argparse.ArgumentParser('plot versus iteration')
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--method',type=str)
parser.add_argument('--runs',type=int)
parser.add_argument('--iterations',type=int)
parser.add_argument('--best_value',type=float,default=None)
#parser.add_argument('--epochs',type=int,default=4000)
#parser.add_argument('--sgld_gamma',type=float,default=0.35)
#parser.add_argument('--folder_index',type=int)
args = parser.parse_args()




def get_Y_curve(input_path,Y_num,interval_num,if_max=1,init_sample=5):
    Y_tot = np.zeros((Y_num,interval_num))
    marker = interval_num+1
    #pdb.set_trace()
    for i in range(Y_num):
        if(os.path.exists(input_path+"Y_"+str(i+1)+".npy")):
            Y = np.load(input_path+"Y_"+str(i+1)+".npy")
            if not if_max:
                Y = -Y
            if(Y.shape[0]>=interval_num):
                for j in range(interval_num):
                    Y_tot[i,j] = np.max(Y[init_sample:j+init_sample+1])
            else:
                marker = min(marker,Y.shape[0])
                for j in range(interval_num):
                    Y_tot[i,j] = np.max(Y[init_sample:j+init_sample+1])
        else:
            break
    #Y_mean = np.mean(Y_tot[:i+1],axis=0)
    #Y_std = np.std(Y_tot[:i+1],axis=0)
    #return Y_mean,Y_std
    return Y_tot[:i]


if args.best_value is None:
    Y_tot = get_Y_curve(args.input_path,args.runs,args.iterations)
else:
    Y_tot = args.best_value-get_Y_curve(args.input_path,args.runs,args.iterations)
Y_err = stats.sem(Y_tot)
Y_mean = np.mean(Y_tot,axis=0)

fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
ax.set_prop_cycle(color=["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","lime","black"])
plt.errorbar(range(0,args.iterations),Y_mean,Y_err,label=args.method)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
plt.xlabel("Number of iterations", fontsize=20)
if args.best_value is None:
    plt.ylabel("Best value found", fontsize=20)
else:
    plt.ylabel("Regret", fontsize=20)

fig.savefig(args.output_path+"/curve_iterations.png")
plt.close()