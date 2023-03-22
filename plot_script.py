### Draw performance curves by iterations

import numpy as np
import matplotlib.pyplot as plt
import os
import pdb

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
    Y_mean = np.mean(Y_tot[:i+1],axis=0)
    Y_std = np.std(Y_tot[:i+1],axis=0)
    return Y_mean,Y_std


func_name = "Hartmann6_6_6_6_50"
Y_num = 20
interv = 200
std_band = 8
method_list = ["VSBO","HeSBO_4_dim","HeSBO_6_dim","REMBO_4_dim","REMBO_6_dim","ALEBO_4_dim","ALEBO_6_dim","saasbo","Botorch","REMBO_interleave_4_dim","REMBO_interleave_6_dim","LineBO"]
method_name = ["VS-BO","HeSBO(4 dim)","HeSBO(6 dim)", "REMBO (4 dim)","REMBO (6 dim)","ALEBO (4 dim)","ALEBO (6 dim)","SAASBO","vanilla BO","REMBO Interleave (4 dim)","REMBO Interleave (6 dim)","LineBO"]
#method_list = ["VSBO","HeSBO_6_dim","HeSBO_10_dim","REMBO_6_dim","REMBO_10_dim","ALEBO_6_dim","LineBO","saasbo","Botorch","REMBO_interleave_6_dim","REMBO_interleave_10_dim"]
#method_name = ["VS-BO","HeSBO(6 dim)","HeSBO(10 dim)", "REMBO (6 dim)","REMBO (10 dim)","ALEBO (6 dim)","LineBO","SAASBO","vanilla BO","REMBO Interleave (6 dim)","REMBO Interleave (10 dim)"]



### curves by iterations
Y_mean_list = []
Y_std_list = []
for i in range(len(method_list)):
    if(method_list[i][:5]=="REMBO"):
        Y_mean,Y_std = get_Y_curve("./"+method_list[i]+"/",Y_num,interv,init_sample=0)
    elif(method_list[i]=="saasbo"):
        Y_mean,Y_std = get_Y_curve("./"+method_list[i]+"/",10,interv,if_max=0)
    else:
        Y_mean,Y_std = get_Y_curve("./"+method_list[i]+"/",Y_num,interv)
    Y_mean_list.append(Y_mean.copy())
    Y_std_list.append(Y_std.copy())

fig = plt.figure()
ax = plt.subplot(111)
ax.set_prop_cycle(color=["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","lime","black"])
for i in range(len(method_list)):
    ax.plot(range(0,interv),Y_mean_list[i],label=method_name[i])
    ax.fill_between(range(0,interv), (Y_mean_list[i]-Y_std_list[i]/std_band), (Y_mean_list[i]+Y_std_list[i]/std_band), alpha=.4)

ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
#plt.ylim(-420, -260)
plt.xlabel("Number of iterations", fontsize=20)
plt.ylabel("Best value found", fontsize=20)
#fig.savefig(func_name+"_iter_compare_automl23_nolegend.png")
plt.show()
plt.close()




def get_T_Y_data(input_path,Y_num,T_max,interval_num,T_type=0,Dragonfly=0,if_max=1,if_accumuate=1):
    T_Y = np.zeros((Y_num,interval_num))
    for i in range(Y_num):
        if not os.path.exists(input_path+"Time_"+str(i+1)+".npy"):
            break
        if(T_type==0):
            T = np.load(input_path+"Time_"+str(i+1)+".npy")
        else:
            T = np.load(input_path+"Time_process_"+str(i+1)+".npy")
        if Dragonfly:
            #pdb.set_trace()
            T[6:] = T[6:] - T[6] + 1
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
    return np.mean(T_Y[:i],axis=0),np.std(T_Y[:i],axis=0)



method_list = ["VSBO","HeSBO_4_dim","HeSBO_6_dim","REMBO_4_dim","REMBO_6_dim","ALEBO_4_dim","ALEBO_6_dim","saasbo","Botorch","REMBO_interleave_4_dim","REMBO_interleave_6_dim","LineBO"]
method_name = ["VS-BO","HeSBO(4 dim)","HeSBO(6 dim)", "REMBO (4 dim)","REMBO (6 dim)","ALEBO (4 dim)","ALEBO (6 dim)","SAASBO","vanilla BO","REMBO Interleave (4 dim)","REMBO Interleave (6 dim)","LineBO"]
#method_list = ["VSBO","HeSBO_6_dim","HeSBO_10_dim","REMBO_6_dim","REMBO_10_dim","ALEBO_6_dim","LineBO","saasbo","Botorch","REMBO_interleave_6_dim","REMBO_interleave_10_dim"]
#method_name = ["VS-BO","HeSBO(6 dim)","HeSBO(10 dim)", "REMBO (6 dim)","REMBO (10 dim)","ALEBO (6 dim)","LineBO","SAASBO","vanilla BO","REMBO Interleave (6 dim)","REMBO Interleave (10 dim)"]

### curves by wall clock time and CPU time
Y_num = 20
interv = 100
T_type = 0
if T_type==0:
    T_max = 400
else:
    T_max = 6000
    

std_band = 8
T_Y_mean_list = []
T_Y_std_list = []
for i in range(len(method_list)):
    if(method_list[i]=="saasbo"):
        T_Y_mean,T_Y_std = get_T_Y_data("./"+method_list[i]+"/",Y_num,T_max,interv,T_type=T_type,if_max=0,if_accumuate=0)
    else:
        T_Y_mean,T_Y_std = get_T_Y_data("./"+method_list[i]+"/",Y_num,T_max,interv,T_type=T_type)
    T_Y_mean_list.append(T_Y_mean.copy())
    T_Y_std_list.append(T_Y_std.copy())


fig,ax = plt.subplots()
ax.set_prop_cycle(color=["blue","orange","green","red","purple","brown","pink","gray","olive","cyan","lime","black"])
for i in range(len(method_list)):
    ax.plot(range(0,T_max,int(T_max/interv)),T_Y_mean_list[i],label=method_name[i])
    ax.fill_between(range(0,T_max,int(T_max/interv)), (T_Y_mean_list[i]-T_Y_std_list[i]/std_band), (T_Y_mean_list[i]+T_Y_std_list[i]/std_band), alpha=.4)
    

ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
#plt.ylim(-420, -260)
plt.ylabel("Best value found", fontsize=20)
if T_type==0:
    plt.xlabel("Wall clock time (seconds)", fontsize=20)
    #fig.savefig(func_name+"_wct_compare_IJCAI2022_nolegend.png")
    #fig.savefig(func_name+"_wct_compare_IJCAI2022.png")
else:
    plt.xlabel("CPU time (seconds)", fontsize=20)
    #fig.savefig(func_name+"_cput_compare_IJCAI2022_nolegend.png")
    #fig.savefig(func_name+"_cput_compare_IJCAI2022.png")
plt.show()
plt.close()



### Draw the frequency plot of being chosen as important for each variable

import numpy as np
import matplotlib.pyplot as plt
object_dim = 50
chosen_stats = np.zeros(object_dim)
for i in range(20):
    F_chosen = np.load("./VSBO/F_chosen_"+str(i+1)+".npy")[:10]
    n = F_chosen.shape[0]
    for j in range(n):
        chosen_stats[F_chosen[j]]+=1


#func_name = "Branin_2_2_2_50"
fig,ax = plt.subplots()
plt.bar(range(1,object_dim+1),chosen_stats)
ax.tick_params(axis="x", labelsize=18)
ax.tick_params(axis="y", labelsize=18)
plt.xlabel("Variable index", fontsize=20)
#plt.ylabel("Total frequency of being important", fontsize=14)
plt.show()
#plt.savefig(func_name+"_F_chosen_bar.png")
plt.close()