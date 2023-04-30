import argparse

import numpy as np
import numpyro
from numpyro.util import enable_x64

#from hartmann import hartmann6_50
from saasbo import run_saasbo
from VSBO_utils import *


parser = argparse.ArgumentParser('saasbo')
parser.add_argument('--obj_func', type=str)
#parser.add_argument('--method', type=str)
#parser.add_argument('--momentum',type=int,default=1)
#parser.add_argument('--sampling',type=str,default='CMAES_posterior')
#parser.add_argument('--num_target',type=int,default=64)
#parser.add_argument('--epochs',type=int,default=4000)
#parser.add_argument('--sgld_gamma',type=float,default=0.35)
#parser.add_argument('--folder_index',type=int)
args = parser.parse_args()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.obj_func=="Branin":
    ### Branin test with D=50 and d_{e}=[2,2,2]
    object_dim = 50
    object_bounds = torch.cat([generate_branin_bounds(2),generate_branin_bounds(2),generate_branin_bounds(object_dim-4)],dim=1)
    object_func = Combine_func
    function_kwargs = {'func_set':[Branin_hd,Branin_hd,Branin_hd],'var_num_set':[2,2,2],'coeff_set':[1,0.1,0.01]}
    total_budget = 200
    #total_time = 600
    #total_cput = 9000
elif args.obj_func=="Hartmann6":
    ### Hartmann6 test with D=50 and d_{e}=[6,6,6]
    object_dim = 50
    object_bounds = generate_hartmann_bounds(object_dim)
    object_func = Combine_func
    function_kwargs = {'func_set':[Hartmann6,Hartmann6,Hartmann6],'var_num_set':[6,6,6],'coeff_set':[1,0.1,0.01]}
    total_budget = 200
    #total_time = 400
    #total_cput = 7000
elif args.obj_func=="StyblinskiTang4":
    ### StyblinskiTang4 test with D=50 and d_{e}=[4,4,4]
    object_dim = 50
    object_bounds = generate_StyblinskiTang_bounds(object_dim)
    object_func = Combine_func
    function_kwargs = {'func_set':[StyblinskiTang4_hd,StyblinskiTang4_hd,StyblinskiTang4_hd],'var_num_set':[4,4,4],'coeff_set':[1,0.1,0.01]}
    total_budget = 200
    #total_time = 2000
    #total_cput = 30000
elif args.obj_func=="rover":
    from rover_test_utils import *
    object_dim = 60
    object_bounds = generate_rover_bounds()
    object_func = rover_func
    total_budget = 155
    #total_time = 2000
    #total_cput = -1
    function_kwargs={}
elif args.obj_func=="mopta":
    from mopta_utils import *
    object_dim = 124
    object_bounds = generate_mopta_bounds()
    object_func = mopta_func
    total_budget = 155
    #total_time = 3000
    #total_cput = -1
    function_kwargs={}

#object_dim = 50
#object_bounds = torch.cat([generate_branin_bounds(2),generate_branin_bounds(2),generate_branin_bounds(object_dim-4)],dim=1)
#object_bounds = generate_hartmann_bounds(object_dim)
#object_bounds = generate_StyblinskiTang_bounds(object_dim)
#function_kwargs = {'func_set':[Branin_hd,Branin_hd,Branin_hd],'var_num_set':[2,2,2],'coeff_set':[1,0.1,0.01]}
#function_kwargs = {'func_set':[Hartmann6,Hartmann6,Hartmann6],'var_num_set':[6,6,6],'coeff_set':[1,0.1,0.01]}
#function_kwargs = {'func_set':[StyblinskiTang4_hd,StyblinskiTang4_hd,StyblinskiTang4_hd],'var_num_set':[4,4,4],'coeff_set':[1,0.1,0.01]}
#object_func = Combine_func
#embed_dim = 6
#output_path = "./hyper_tuning/FS_BO_test/benchmark/Hartmann6/6_6_6_50/saasbo/"
#output_path = "./hyper_tuning/FS_BO_test/benchmark/Branin/2_2_2_50/saasbo/"
#output_path = "./hyper_tuning/FS_BO_test/benchmark/StyblinskiTang/4_4_4_50/saasbo/"


'''
object_dim = 60
object_bounds = generate_rover_bounds()
object_func = rover_func
function_kwargs = {}
output_path = "./hyper_tuning/FS_BO_test/benchmark/rover/saasbo/"


object_dim = 124
object_bounds = generate_mopta_bounds()
object_func = mopta_func
function_kwargs = {}
output_path = "./hyper_tuning/FS_BO_test/benchmark/mopta_08/saasbo/"


if not os.path.exists(output_path):
    os.makedirs(output_path)
'''

output_path = "./"+args.obj_func+"/saasbo/"
makedirs(output_path)


def func_wrapper(x):
    x_torch = torch.from_numpy(x).reshape((1,len(x)))
    y_torch = object_func(x_torch,**function_kwargs)
    return -y_torch.item()


numpyro.set_platform("cpu")
enable_x64()
#arctic
numpyro.set_host_device_count(40)
#alpine
#numpyro.set_host_device_count(80)
init_samples = 5

for test_id in range(1,11):
    seed = np.random.randint(int(1e6))
    X,Y,T,T_process = run_saasbo(
        func_wrapper,
        object_bounds[0].numpy(),
        object_bounds[1].numpy(),
        total_budget,
        init_samples,
        seed=seed,
        alpha=0.01,
        num_warmup=256,
        num_samples=256,
        thinning=32,
        device="cpu",
    )
    np.save(output_path+"X_"+str(test_id)+".npy",X)
    np.save(output_path+"Y_"+str(test_id)+".npy",Y)
    np.save(output_path+"Time_"+str(test_id)+".npy",T)
    np.save(output_path+"Time_process_"+str(test_id)+".npy",T_process)
