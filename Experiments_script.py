from VSBO_class import *

### some parameters
N_FS = 20
acq_optim_method = 'LBFGS'
### use CMAES to sample unimportant variables
less_important_sampling = 'CMAES_posterior'
init_samples = 5


### Branin test with D=50 and d_{e}=[2,2,2]
object_dim = 50
object_bounds = torch.cat([generate_branin_bounds(2),generate_branin_bounds(2),generate_branin_bounds(object_dim-4)],dim=1)
object_func = Combine_func
function_kwargs = {'func_set':[Branin_hd,Branin_hd,Branin_hd],'var_num_set':[2,2,2],'coeff_set':[1,0.1,0.01]}
total_budget = 210
total_time = 600
total_cput = 10000

### Hartmann6 test with D=50 and d_{e}=[6,6,6]
object_dim = 50
object_bounds = generate_hartmann_bounds(object_dim)
object_func = Combine_func
function_kwargs = {'func_set':[Hartmann6,Hartmann6,Hartmann6],'var_num_set':[6,6,6],'coeff_set':[1,0.1,0.01]}
total_budget = 250
total_time = 400
total_cput = 7000

### StyblinskiTang4 test with D=50 and d_{e}=[4,4,4]
object_dim = 50
object_bounds = generate_StyblinskiTang_bounds(object_dim)
object_func = Combine_func
function_kwargs = {'func_set':[StyblinskiTang4_hd,StyblinskiTang4_hd,StyblinskiTang4_hd],'var_num_set':[4,4,4],'coeff_set':[1,0.1,0.01]}
total_budget = 210
total_time = 2000
total_cput = 30000

### Rotation Hartmann6 test with D=100 and d_{e}=6
object_dim = 100
real_dim = 6
object_bounds = torch.stack([-torch.ones(object_dim, dtype=dtype, device=device),torch.ones(object_dim, dtype=dtype, device=device)])
function_kwargs = {}
total_budget = 220
total_time = 400
total_cput = 7000
np.random.seed(1000)
random_basis = torch.tensor(special_ortho_group.rvs(object_dim)[:real_dim, :],dtype=dtype, device=device)

def rotation_wrapper(X):
    #pdb.set_trace()
    kwargs = {'func_set':[Hartmann6],'var_num_set':[6],'coeff_set':[1]}
    X_new = torch.matmul(random_basis,X.transpose(0,1)).transpose(0,1)
    return Combine_func((X_new+1.0)/2.0,**kwargs)

object_func = rotation_wrapper

### Rover trajectory test D=60
from rover_test_utils import *
object_dim = 60
object_bounds = generate_rover_bounds()
object_func = rover_func
total_time = 3600
function_kwargs={}

### MOPTA08 test D=124
from mopta_utils import *
object_dim = 124
object_bounds = generate_mopta_bounds()
object_func = mopta_func
total_time = 4800
function_kwargs={}


### VS-BO pipeline

output_path = "./benchmark/Branin/2_2_2_50/VSBO/"

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
                BO_instance.variable_selection_2('KLrel')
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



### vanilla BO pipeline

output_path = "./benchmark/Branin/2_2_2_50/Botorch/"

for test_id in range(1,21):
    BO_instance = BOtorch(object_dim,object_func,obj_func_kwargs=function_kwargs,bounds=object_bounds)
    BO_instance.data_initialize()
    Times = []
    T_process = []
    t0 = time.time()
    t1 = time.process_time()
    iter_num = 0
    while (time.time() -t0 < total_time or iter_num < total_budget or time.process_time()-t1 < total_cput):
        iter_num+=1
        try:
            BO_instance.GP_fitting(GP_Matern)
            BO_instance.BO_acq_optim(optim_method='LBFGS')
            BO_instance.data_update()
            Times.append(time.time()-t0)
            T_process.append(time.process_time()-t1)
        except ValueError as e:
            if(e.args[0]=='Too many cov mat singular!'):
                BO_instance.erase_last_instance()
                iter_num-=1
                continue
            else:
                raise ValueError(e.args[0])
        if(iter_num%10==0):
            print(
                f"Epoch {iter_num:>3} "
                f"Best value: {torch.max(BO_instance.Y).item():>4.3f}"
            )
    np.save(output_path+"X_"+str(test_id)+".npy",BO_instance.X.numpy())
    np.save(output_path+"Y_"+str(test_id)+".npy",BO_instance.Y.numpy())
    np.save(output_path+"Time_"+str(test_id)+".npy",np.array(Times))
    np.save(output_path+"Time_process_"+str(test_id)+".npy",np.array(T_process)) 



from rembo_utils import *

### rembo pipeline

embed_dim = 6
output_path = "./benchmark/Branin/2_2_2_50/REMBO_6_dim/"
for test_id in range(1,21):
    opt = REMBOOptimizer(GP_model=GP_Matern,initial_random_samples=init_samples,n_dims=object_dim,n_embedding_dims=embed_dim)
    opt.model_initialise(object_bounds.T.numpy())
    opt.y_ = object_func(opt.X_,**function_kwargs)
    _,_ = opt.fit_model()
    Times = []
    T_process = []
    t0 = time.time()
    t1 = time.process_time()
    iter_num = 0
    #while time.time() - t0 < total_time:
    while (time.time() -t0 < total_time or iter_num < total_budget or time.process_time()-t1 < total_cput):
        iter_num+=1
        x_new = opt.select_query_point()
        #x_new[:,integer_f_list] = torch.round(x_new[:,integer_f_list])
        y_new = object_func(x_new,**function_kwargs)
        _,_ = opt.update(x_new,y_new)
        Times.append(time.time()-t0)
        T_process.append(time.process_time()-t1)
        if(iter_num%10==0):
            print(
                f"Epoch {iter_num:>3} "
                f"Best value: {opt.best_value().item():>4.3f}"
            )
    np.save(output_path + "X_"+str(test_id)+".npy",opt.X_.numpy())
    np.save(output_path + "Y_"+str(test_id)+".npy",opt.y_.numpy())
    np.save(output_path + "Time_"+str(test_id)+".npy",np.array(Times))
    np.save(output_path + "Time_process_"+str(test_id)+".npy",np.array(T_process))


### rembo interleave pipeline

embed_dim = 6
interleaved_cycle = 4
output_path = "./benchmark/Branin/2_2_2_50/REMBO_interleave_6_dim/"

for test_id in range(1,21):
    opt = InterleavedREMBOOptimizer(interleaved_runs=interleaved_cycle,random_state=test_id,GP_model=GP_Matern,n_dims=object_dim,n_embedding_dims=embed_dim)
    for i in range(interleaved_cycle):
        opt.rembos[i].model_initialise(object_bounds.T.numpy())
        opt.rembos[i].y_ = object_func(opt.rembos[i].X_,**function_kwargs)
        _,_ = opt.rembos[i].fit_model()
    #opt.model_initialise(object_bounds.T.numpy())
    #opt.y_ = object_func(opt.X_)
    opt.get_cycle()
    #_,_ = opt.fit_model()
    Times = []
    T_process = []
    t0 = time.time()
    t1 = time.process_time()
    iter_num = 0
    while (time.time() -t0 < total_time or iter_num < total_budget or time.process_time()-t1 < total_cput):
        iter_num+=1
        x_new = opt.select_query_point()
        #x_new[:,integer_f_list] = torch.round(x_new[:,integer_f_list])
        y_new = object_func(x_new,**function_kwargs)
        opt.update(x_new,y_new)
        Times.append(time.time()-t0)
        T_process.append(time.process_time()-t1)
        if(iter_num%10==0):
            print(
                f"Epoch {iter_num:>3} "
                f"Best value: {opt.best_value().item():>4.3f}"
            )
    np.save(output_path + "X_"+str(test_id)+".npy",opt.X_.numpy())
    np.save(output_path + "Y_"+str(test_id)+".npy",opt.y_.numpy())
    np.save(output_path + "Time_"+str(test_id)+".npy",np.array(Times))
    np.save(output_path + "Time_process_"+str(test_id)+".npy",np.array(T_process))



### Alebo Hesbo pipeline
from alebo_hesbo_utils import *
Times = []
T_process = []
wct = 0
cput = 0
y_max = -100
embed_dim = 6
parameters = convert_botorch_bounds_to_ax(object_bounds)


def object_func_ax_wrapper(parameterization):
    global Times
    global T_process
    global y_max
    #global function_kwargs
    #pdb.set_trace()
    Times.append(time.time()-wct)
    T_process.append(time.process_time()-cput)
    dim = len(parameterization)
    x = torch.tensor([parameterization["x"+str(i)] for i in range(dim)],dtype=dtype, device=device).reshape((1,dim))
    y = float(object_func(x,**function_kwargs))
    y_max = max(y,y_max)
    print(y,y_max)
    return {"objective": (y, 0.0)}


### Alebo pipeline
output_path = "./benchmark/Branin/2_2_2_50/ALEBO_6_dim/"

for i in range(1,21):
    alebo_strategy = ALEBOStrategy(D=object_dim, d=embed_dim, init_size=init_samples)
    wct = time.time()
    cput = time.process_time()
    Times = []
    T_process = []
    y_max = -100
    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        experiment_name="rover",
        objective_name="objective",
        evaluation_function=object_func_ax_wrapper,
        minimize=False,
        total_trials=total_budget,
        generation_strategy=alebo_strategy,
    )
    np.save(output_path + "Time_"+str(i)+".npy",np.array(Times))
    np.save(output_path + "Time_process_"+str(i)+".npy",np.array(T_process))
    Y = np.array([trial.objective_mean for trial in experiment.trials.values()])
    X = convert_para_to_X(experiment,Y.shape[0],object_dim)
    np.save(output_path + "X_"+str(i)+".npy",X)
    np.save(output_path + "Y_"+str(i)+".npy",Y)



### Hesbo pipeline
output_path = "./benchmark/Branin/2_2_2_50/HeSBO_6_dim/"

for i in range(1,21):
    hesbo_strategy = HeSBOStrategy(D=object_dim, d=embed_dim, init_per_proj=init_samples)
    wct = time.time()
    cput = time.process_time()
    Times = []
    T_process = []
    y_max = -100
    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        experiment_name="HeSBO_6",
        objective_name="objective",
        evaluation_function=object_func_ax_wrapper,
        minimize=False,
        total_trials=450,
        generation_strategy=hesbo_strategy,
    )
    np.save(output_path + "Time_"+str(i)+".npy",np.array(Times))
    np.save(output_path + "Time_process_"+str(i)+".npy",np.array(T_process))
    Y = np.array([trial.objective_mean for trial in experiment.trials.values()])
    X = convert_para_to_X(experiment,Y.shape[0],object_dim)
    np.save(output_path + "X_"+str(i)+".npy",X)
    np.save(output_path + "Y_"+str(i)+".npy",Y)