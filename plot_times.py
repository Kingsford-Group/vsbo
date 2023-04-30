import numpy as np
import matplotlib.pyplot as plt
import os
import pdb
from scipy import stats
import argparse


parser = argparse.ArgumentParser('plot versus time')
parser.add_argument('--input_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--method',type=str)
parser.add_argument('--runs',help='number of runs',type=int)
#parser.add_argument('--iterations',help='number of iterations for each run',type=int)
parser.add_argument('--T_max',type=int)
parser.add_argument('--best_value',type=float,default=None)
parser.add_argument('--wct',action='store_true')
parser.add_argument('--not_max',action='store_true')
parser.add_argument('--not_accumuate',action='store_true')
#parser.add_argument('--cput',action='cput')
args = parser.parse_args()



def get_T_Y_data(input_path,Y_num,T_max,interval_num,T_type=0,if_max=1,if_accumuate=1):
    T_Y = np.zeros((Y_num,interval_num))
    for i in range(Y_num):
        if not os.path.exists(input_path+"Time_"+str(i+1)+".npy"):
            break
        if(T_type==0):
            T = np.load(input_path+"Time_"+str(i+1)+".npy")
        else:
            T = np.load(input_path+"Time_process_"+str(i+1)+".npy")
        '''
        if Dragonfly:
            #pdb.set_trace()
            T[6:] = T[6:] - T[6] + 1
        '''
        Y = np.load(input_path+"Y_"+str(i+1)+".npy")
        if not if_max:
            Y = -Y
        #assert T.shape[0]==Y.shape[0]-5
        x = T.shape[0]
        #print(x,Y.shape[0])
        Y_min = Y.min()
        pointer = 0
        for j in range(interval_num):
            while pointer < x:
                if(if_accumuate==1 and T[pointer]>int(T_max/interval_num)*(j+1)):
                    break
                elif(if_accumuate==0 and np.sum(T[:pointer+1])>int(T_max/interval_num)*(j+1)):
                    break
                pointer+=1
            if(pointer==0):
                T_Y[i,j] = Y_min
            else:
                T_Y[i,j] = np.max(Y[:Y.shape[0]-T.shape[0]+pointer-1])
    return T_Y[:i]


if args.wct:
    T_type = 0
else:
    T_type = 1

if args.not_max:
    if_max = 0
else:
    if_max = 1

if args.not_accumuate:
    if_accumuate = 0
else:
    if_accumuate = 1

interval = 100
if args.best_value is None:
    Y_tot = get_T_Y_data(args.input_path,args.runs,args.T_max,interval,T_type=T_type,if_max=if_max,if_accumuate=if_accumuate)
else:
    Y_tot = args.best_value-get_T_Y_data(args.input_path,args.runs,args.T_max,interval,T_type=T_type,if_max=if_max,if_accumuate=if_accumuate)
Y_err = stats.sem(Y_tot)
Y_mean = np.mean(Y_tot,axis=0)


fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(111)
ax.set_prop_cycle(color=["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","lime","black"])
plt.errorbar(range(0,args.T_max,int(args.T_max/interval)),Y_mean,Y_err,label=args.method)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
plt.ylim(0, 8)
if T_type==0:
    plt.xlabel("Wall clock time (seconds)", fontsize=20)
else:
    plt.xlabel("CPU time (seconds)", fontsize=20)
plt.legend(fontsize="18")
if args.best_value is None:
    plt.ylabel("Best value found", fontsize=20)
else:
    plt.ylabel("Regret", fontsize=20)

if T_type==0:
    fig.savefig(args.output_path+"/curve_wct.png")
else:
    fig.savefig(args.output_path+"/curve_cput.png")

plt.close()