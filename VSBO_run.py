from VSBO_class import *
import argparse


parser = argparse.ArgumentParser('VS-BO')
parser.add_argument('--obj_func', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--momentum',type=int,default=1)
parser.add_argument('--sampling',type=str,default='CMAES_posterior')
#parser.add_argument('--num_target',type=int,default=64)
#parser.add_argument('--epochs',type=int,default=4000)
#parser.add_argument('--sgld_gamma',type=float,default=0.35)
#parser.add_argument('--folder_index',type=int)
args = parser.parse_args()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


### some parameters
N_FS = 20
acq_optim_method = 'LBFGS'
### use CMAES to sample unimportant variables
less_important_sampling = args.sampling
init_samples = 5



if args.obj_func=="Branin":
    ### Branin test with D=50 and d_{e}=[2,2,2]
    object_dim = 50
    object_bounds = torch.cat([generate_branin_bounds(2),generate_branin_bounds(2),generate_branin_bounds(object_dim-4)],dim=1)
    object_func = Combine_func
    function_kwargs = {'func_set':[Branin_hd,Branin_hd,Branin_hd],'var_num_set':[2,2,2],'coeff_set':[1,0.1,0.01]}
    total_budget = 210
    total_time = 600
    total_cput = 9000
elif args.obj_func=="Hartmann6":
    ### Hartmann6 test with D=50 and d_{e}=[6,6,6]
    object_dim = 50
    object_bounds = generate_hartmann_bounds(object_dim)
    object_func = Combine_func
    function_kwargs = {'func_set':[Hartmann6,Hartmann6,Hartmann6],'var_num_set':[6,6,6],'coeff_set':[1,0.1,0.01]}
    total_budget = 210
    total_time = 400
    total_cput = 7000
elif args.obj_func=="StyblinskiTang4":
    ### StyblinskiTang4 test with D=50 and d_{e}=[4,4,4]
    object_dim = 50
    object_bounds = generate_StyblinskiTang_bounds(object_dim)
    object_func = Combine_func
    function_kwargs = {'func_set':[StyblinskiTang4_hd,StyblinskiTang4_hd,StyblinskiTang4_hd],'var_num_set':[4,4,4],'coeff_set':[1,0.1,0.01]}
    total_budget = 210
    total_time = 2000
    total_cput = 30000
elif args.obj_func=="rover":
    from rover_test_utils import *
    object_dim = 60
    object_bounds = generate_rover_bounds()
    object_func = rover_func
    total_budget = 210
    total_time = 2000
    total_cput = -1
    function_kwargs={}
elif args.obj_func=="mopta":
    from mopta_utils import *
    object_dim = 124
    object_bounds = generate_mopta_bounds()
    object_func = mopta_func
    total_budget = 210
    total_time = 3000
    total_cput = -1
    function_kwargs={}

output_path = "./"+args.obj_func+"/"+args.method+"/"
if args.method=="VSBO" and args.momentum==0:
    output_path = "./"+args.obj_func+"/"+args.method+"_nomom/"
if args.method=="VSBO" and args.momentum==0 and args.sampling!='CMAES_posterior':
    output_path = "./"+args.obj_func+"/"+args.method+"_nomom_"+args.sampling+"/"
makedirs(output_path)

if args.method =="VSBO":
    #print(total_cput)
    for test_id in range(1,21):
        BO_instance = VSBO(N_FS,object_dim,object_func,obj_func_kwargs=function_kwargs,bounds=object_bounds)
        BO_instance.data_initialize()
        if(less_important_sampling=='CMAES_posterior'):
            BO_instance.CMAES_initialize()
        Times = []
        T_process = []
        F_importance_val = []
        F_rank = []
        F_chosen = []
        t0 = time.time()
        t1 = time.process_time()
        iter_num = 0
        #while time.time() - t0 < total_time:
        while (time.time() -t0 < total_time or iter_num < total_budget or time.process_time()-t1 < total_cput):
        #for one_budget in range(total_budget):
            iter_num+=1
            try:
                ### GP fitting on important variables 
                BO_instance.GP_fitting_active(GP_Matern)
                BO_instance.BO_acq_optim_active(optim_method=acq_optim_method)
                ### sampling on unimportant variables
                BO_instance.data_update(method=less_important_sampling,n_sampling=20)
                Times.append(time.time()-t0)
                T_process.append(time.process_time()-t1)
            except ValueError as e:
                if(e.args[0]=='Too many cov mat singular!'):
                    BO_instance.erase_last_instance()
                    iter_num-=1
                    continue
                else:
                    raise ValueError(e.args[0])
            if(iter_num%BO_instance.N_FS==0):
                try:
                    BO_instance.GP_fitting(GP_Matern)
                    ### We provide three methods for variable seletion
                    ### KLrel: the Grad-IS method introduced in our manuscript
                    ### ard: Automatic Relevence Determination, use the correlation length scales in the kernel function
                    ### fANOVA: use the functional ANOVA (https://pypi.org/project/fanova/)
                    if args.momentum == 1:
                        BO_instance.variable_selection_2('KLrel')
                    elif args.momentum == 0:
                        BO_instance.variable_selection_nomom('KLrel')
                #BO_instance.variable_selection('KLrel')
                except ValueError as e:
                    if(e.args[0]=='Too many cov mat singular!'):
                        BO_instance.erase_last_instance()
                        iter_num-=1
                        continue
                    else:
                        raise ValueError(e.args[0])
                F_importance_val.append(BO_instance.FS_important_scores)
                F_rank.append(BO_instance.indices)
                F_chosen.append(BO_instance.active_f_list)
                if(less_important_sampling=='CMAES_posterior'):
                    BO_instance.CMAES_update()
                print(BO_instance.active_f_list)
                print(BO_instance.active_f_dims)
            if(iter_num%10==0):
                print(
                    f"Epoch {iter_num:>3} "
                    f"Best value: {torch.max(BO_instance.Y).item():>4.3f}"
                )
        np.save(output_path+"X_"+str(test_id)+".npy",BO_instance.X.numpy())
        np.save(output_path+"Y_"+str(test_id)+".npy",BO_instance.Y.numpy())
        np.save(output_path+"Time_"+str(test_id)+".npy",np.array(Times))
        np.save(output_path+"Time_process_"+str(test_id)+".npy",np.array(T_process))
        np.save(output_path+"F_importance_val_"+str(test_id)+".npy",torch.cat(F_importance_val).reshape(len(F_importance_val),object_dim).detach().numpy())
        np.save(output_path+"F_rank_"+str(test_id)+".npy",torch.cat(F_rank).reshape(len(F_rank),object_dim).numpy())
        np.save(output_path+"F_chosen_"+str(test_id)+".npy",torch.cat(F_chosen).reshape(len(F_chosen),object_dim).numpy())
