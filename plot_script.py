### Draw performance curves by iterations

import numpy as np
import matplotlib.pyplot as plt
import os
import pdb


def get_Y_curve(input_path,Y_num,interval_num,init_sample=5,slow_bo=0):
    Y_tot = np.zeros((Y_num,interval_num))
    marker = interval_num+1
    for i in range(Y_num):
        Y = np.load(input_path+"Y_"+str(i+1)+".npy")
        if(Y.shape[0]>=interval_num):
            for j in range(interval_num):
                Y_tot[i,j] = np.max(Y[:j+init_sample+1])
        else:
            marker = min(marker,Y.shape[0])
            for j in range(interval_num):
                Y_tot[i,j] = np.max(Y[:j+init_sample+1])
    Y_mean = np.mean(Y_tot,axis=0)
    Y_std = np.std(Y_tot,axis=0)
    if(slow_bo):
        Y_mean[max(160,marker):] = np.nan
        Y_std[max(160,marker):] = np.nan
    else:
        Y_mean[max(210,marker):] = np.nan
        Y_std[max(210,marker):] = np.nan
    return Y_mean,Y_std


func_name = "Branin_2_2_2_50"
Y_num = 20
interv = 210
method_list = ["VSBO","HeSBO_4_dim","HeSBO_6_dim","REMBO_4_dim","REMBO_6_dim","ALEBO_4_dim","ALEBO_6_dim","Dragonfly","Botorch","REMBO_interleave_4_dim","REMBO_interleave_6_dim"]
method_name = ["VS-BO","HeSBO(4 dim)","HeSBO(6 dim)", "REMBO (4 dim)","REMBO (6 dim)","ALEBO (4 dim)","ALEBO (6 dim)","Dragonfly","vanilla BO","REMBO Interleave (4 dim)","REMBO Interleave (6 dim)"]


Y_mean_list = []
Y_std_list = []
for i in range(len(method_list)):
    if(method_list[i][:5]=="REMBO"):
        Y_mean,Y_std = get_Y_curve("./"+method_list[i]+"/",Y_num,interv,init_sample=0)
    #elif(method_list[i]=="Dragonfly"):
        #Y_mean,Y_std = get_Y_curve("./"+method_list[i]+"_ask_tell/",Y_num,interv)
    elif(method_list[i]=="Botorch" or method_list[i]=="ALEBO_6_dim"):
        Y_mean,Y_std = get_Y_curve("./"+method_list[i]+"/",Y_num,interv,slow_bo=1)
    else:
        Y_mean,Y_std = get_Y_curve("./"+method_list[i]+"/",Y_num,interv)
    Y_mean_list.append(Y_mean.copy())
    Y_std_list.append(Y_std.copy())

fig = plt.figure()
ax = plt.subplot(111)
ax.set_prop_cycle(color=["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","lime"])
for i in range(len(method_list)):
    ax.plot(range(0,interv),Y_mean_list[i],label=method_name[i])
    ax.fill_between(range(0,interv), (Y_mean_list[i]-Y_std_list[i]/4), (Y_mean_list[i]+Y_std_list[i]/4), alpha=.4)

ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=14)
plt.xlabel("Number of iterations", fontsize=16)
plt.ylabel("Best value found", fontsize=16)
fig.savefig(func_name+"_iter_compare_NIPS2021_nolegend.png")
plt.close()


### Draw the frequency plot of being chosen as important for each variable

import numpy as np
import matplotlib.pyplot as plt
object_dim = 50
chosen_stats = np.zeros(object_dim)
for i in range(20):
    F_chosen = np.load("./VSBO/F_chosen_"+str(i+1)+".npy")
    n = F_chosen.shape[0]
    for j in range(n):
        chosen_stats[F_chosen[j]]+=1


func_name = "Branin_2_2_2_50"
fig,ax = plt.subplots()
plt.bar(range(1,object_dim+1),chosen_stats)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
plt.xlabel("Variable index", fontsize=16)
plt.ylabel("Total frequency of being important", fontsize=12)
plt.savefig(func_name+"_F_chosen_bar.png")
plt.close()