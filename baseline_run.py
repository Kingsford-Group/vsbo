from VSBO_class import *
import argparse


parser = argparse.ArgumentParser('VS-BO')
parser.add_argument('--obj_func', type=str)
parser.add_argument('--method', type=str)
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
makedirs(output_path)
init_samples = 5

if args.method == "vanillaBO":
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
elif args.method == "rembo":
    from rembo_utils import *
    embed_dim = 6
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
elif args.method=="rembo_interleave":
    from rembo_utils import *
    embed_dim = 6
    interleaved_cycle = 4
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
#Times = []
#T_process = []
#wct = 0
#cput = 0
#y_max = -100
#embed_dim = 6
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


if args.method=="alebo":
    embed_dim = 6
    for i in range(1,21):
        alebo_strategy = ALEBOStrategy(D=object_dim, d=embed_dim, init_size=init_samples)
        wct = time.time()
        cput = time.process_time()
        Times = []
        T_process = []
        y_max = -100
        best_parameters, values, experiment, model = optimize(
            parameters=parameters,
            experiment_name="ALEBO",
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
elif args.method=="hesbo":
    embed_dim = 6
    for i in range(1,21):
        hesbo_strategy = HeSBOStrategy(D=object_dim, d=embed_dim, init_per_proj=init_samples)
        wct = time.time()
        cput = time.process_time()
        Times = []
        T_process = []
        y_max = -100
        best_parameters, values, experiment, model = optimize(
            parameters=parameters,
            experiment_name="HeSBO",
            objective_name="objective",
            evaluation_function=object_func_ax_wrapper,
            minimize=False,
            total_trials=total_budget,
            generation_strategy=hesbo_strategy,
        )
        np.save(output_path + "Time_"+str(i)+".npy",np.array(Times))
        np.save(output_path + "Time_process_"+str(i)+".npy",np.array(T_process))
        Y = np.array([trial.objective_mean for trial in experiment.trials.values()])
        X = convert_para_to_X(experiment,Y.shape[0],object_dim)
        np.save(output_path + "X_"+str(i)+".npy",X)
        np.save(output_path + "Y_"+str(i)+".npy",Y)