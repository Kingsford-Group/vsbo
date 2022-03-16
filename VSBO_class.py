from VSBO_utils import *
tmvtnorm = importr('tmvtnorm')
rpy2.robjects.numpy2ri.activate()


class BOtorch(object):
    def __init__(self,input_dim,obj_func,obj_func_kwargs={},bounds=[],*args,**kwargs):
        #self.X = X
        self.X_dim = input_dim
        #self.Y = Y
        self.bound = bounds
        self.obj_info = {}
        self.obj_info['function'] = obj_func
        self.obj_info['function_kwarg'] = obj_func_kwargs
    def data_initialize(self,initial_random_samples=5):
        self.X = unnormalize(torch.rand(initial_random_samples, self.X_dim, device=device, dtype=dtype), bounds=self.bound)
        self.Y = self.obj_info['function'](self.X,**self.obj_info['function_kwarg'])
    @catch_error
    def fit_model(self,X,bounds,model_class,rand_init=0,**kwargs):
        X_normalize = normalize(X,bounds=bounds)
        self.Y_standard = standardize(self.Y)
        GP_model = model_class(X_normalize,self.Y_standard,bounds=bounds,ard_num_dims=X_normalize.shape[-1],**kwargs)
        if(rand_init>0 and rand_init<10):
            GP_model.covar_module.base_kernel.lengthscale = torch.rand(GP_model.covar_module.base_kernel.lengthscale.size(), device=device, dtype=dtype)
            GP_model.covar_module.outputscale = torch.rand(GP_model.covar_module.outputscale.size(),device=device, dtype=dtype)
            GP_model.likelihood.noise = rand_init+1
        elif rand_init>=10:
            raise ValueError('Too many cov mat singular!')
        mll = ExactMarginalLogLikelihood(likelihood=GP_model.likelihood, model=GP_model)
        mll = mll.to(X_normalize)
        init_loss = self.get_loss(GP_model,X_normalize,mll)
        fit_gpytorch_model(mll)
        final_loss = self.get_loss(GP_model,X_normalize,mll)
        return X_normalize,GP_model,mll,init_loss,final_loss
    def GP_fitting(self,model_class,**kwargs):
        self.model_class = model_class
        self.X_normalize,self.model,self.mll,self.init_loss,self.final_loss = self.fit_model(self.X,self.bound,model_class,**kwargs)
    def get_loss(self,model,X_normalize,mll):
        model.train()
        output = model(X_normalize)
        return - mll(output,model.train_targets)
    def acq_optimize(self,model,dim,acq_func,optim_method='LBFGS',**kwargs):
        #pdb.set_trace()
        acq = acq_func(model,**kwargs)
        '''
        if(acq_func=='EI'):
            acq = ExpectedImprovement(model,best_f=self.Y_standard.max().item())
        else:
            print('No implementation on this acquisition function!')
        '''
        if(optim_method=='LBFGS'):
            candidates,_ = optimize_acqf(
                acq_function = acq,
                bounds = torch.stack([
                    torch.zeros(dim, dtype=dtype, device=device),
                    torch.ones(dim, dtype=dtype, device=device)
                ]),
                q=1,
                num_restarts = 10,
                raw_samples = 20,
            )
            new_x_normalize = candidates.detach() 
        elif(optim_method=='CMAES'):
            es = cma.CMAEvolutionStrategy(
                x0 = np.random.rand(dim),
                sigma0=0.2,
                inopts={'bounds': [0, 1], "popsize": 50},
            )
            with torch.no_grad():
                while not es.stop():
                    xs = es.ask()
                    XS = torch.tensor(xs, device=device, dtype=dtype)
                    YS = -acq(XS.unsqueeze(-2))
                    ys = YS.view(-1).double().numpy()
                    es.tell(xs, ys)
            new_x_normalize = torch.from_numpy(es.best.x).to(XS).reshape((1,dim))
            #pdb.set_trace()
        else:
            print('No implementation on this optimization!')
        return acq,new_x_normalize
    def BO_acq_optim(self,optim_method='LBFGS'):
        self.optim_method = optim_method
        self.acq,self.new_x_normalize = self.acq_optimize(self.model,self.X_dim,ExpectedImprovement,optim_method=optim_method,best_f=self.Y_standard.max().item())
    def data_update(self):
        new_x = unnormalize(self.new_x_normalize,bounds=self.bound)
        new_y = self.obj_info['function'](new_x,**self.obj_info['function_kwarg'])
        self.X, self.Y = torch.cat((self.X,new_x)),torch.cat((self.Y,new_y))
    def erase_last_instance(self):
        self.X = self.X[:-1]
        self.Y = self.Y[:-1]


class VSBO(BOtorch):
    def __init__(self,N_FS,*args, **kwargs):
        super(VSBO, self).__init__(*args, **kwargs)
        self.N_FS = N_FS
        self.active_f_dims = self.X_dim
        self.active_f_list = torch.tensor([1 for i in range(self.X_dim)],dtype=torch.bool,device=device)
    def CMAES_initialize(self):
        self.es = cma.CMAEvolutionStrategy(
            x0 = np.random.rand(self.X_dim),
            sigma0=0.2,
            inopts={'bounds': [0, 1], "popsize": self.N_FS},
        )
    def CMAES_update(self):
        #pdb.set_trace()
        _ = self.es.ask()
        X_normalize = normalize(self.X,bounds=self.bound)
        X_normalize_np = X_normalize.numpy()
        Y_np = self.Y.numpy()
        self.es.tell([X_normalize_np[j] for j in range(-self.N_FS,0,1)],[-Y_np[j] for j in range(-self.N_FS,0,1)])
        #[mu1,mu2,cov11,cov12,cov21,cov22,cov22_inv,cond_cov]
        self.conditiona_normal_list = get_conditional_normal(self.es.mean,(self.es.sigma**2)*self.es.C,~self.active_f_list)
        self.cond_cov_cholesky = np.linalg.cholesky(self.conditiona_normal_list[-1])
    def GP_fitting_active(self,model_class,**kwargs):
        self.model_class = model_class
        self.X_normalize_active,self.model_active,self.mll_active,self.init_loss_active,self.final_loss_active = self.fit_model(self.X[:,self.active_f_list],self.bound[:,self.active_f_list],model_class,**kwargs)
    def BO_acq_optim_active(self,optim_method='LBFGS'):
        self.optim_method = optim_method
        self.acq_active,self.new_x_normalize_active = self.acq_optimize(self.model_active,self.active_f_dims,ExpectedImprovement,optim_method=optim_method,best_f=self.Y_standard.max().item())
    def calc_important_score(self,model,method='KLrel',*args, **kwargs):
        if(method=='ard'):
            return FS_ARD(model)
        elif(method=='KLrel'):
            return FS_KLrel(model,*args, **kwargs)
        elif(method=='fANOVA'):
            return FS_fANOVA(*args, **kwargs)
            #pdb.set_trace()
            #f = fANOVA(kwargs['X'],kwargs['Y'])
        else:
            print("This immportant score calculation method has not been implemented!")
    def variable_selection_2(self,FS_score_method,*args, **kwargs):
        #pdb.set_trace()
        self.FS_score_method = FS_score_method
        #self.GP_fitting(self.model_class,**kwargs)
        self.FS_important_scores = self.calc_important_score(self.model,self.FS_score_method,dim=self.X_dim,active_f_list=torch.tensor([1 for i in range(self.X_dim)],dtype=torch.bool,device=device),*args, **kwargs)
        kwargs_old = kwargs.copy()
        _,self.indices = torch.sort(self.FS_important_scores,descending=True)
        print(self.indices)
        if(self.X_dim==self.active_f_dims):
            self.stepwise_forward_2(0,torch.tensor([0 for k in range(self.X_dim)],dtype=torch.bool,device=device),**kwargs)
        else:
            delta_Y = torch.max(self.Y)-torch.max(self.Y[:-self.N_FS])
            if(delta_Y<=0):
                self.stepwise_forward_2(0,self.active_f_list,**kwargs)
            else:
                mark = 0
                prev_active_dim_remain = self.active_f_dims
                prev_active_index = torch.where(self.active_f_list==1)[0]
                start_point = 0
                if('variable_type' in kwargs.keys()):
                    kwargs = kwargs_old.copy()
                    kwargs['variable_type'] = kwargs_old['variable_type'][self.active_f_list]
                FS_important_scores_active = self.calc_important_score(self.model_active,self.FS_score_method,dim=self.active_f_dims,active_f_list=self.active_f_list,*args, **kwargs)
                _,indices_active = torch.sort(FS_important_scores_active,descending=True)
                #RFE
                prev_loss = self.final_loss_active
                get_loss_interval = 0
                for k in range(self.active_f_dims-1,0,-1):
                    try:
                        if('variable_type' in kwargs.keys()):
                            kwargs = kwargs_old.copy()
                            kwargs['variable_type'] = kwargs_old['variable_type'][prev_active_index[indices_active[:k]]]
                        _,_,_,_,sub_final_loss = self.fit_model(self.X[:,prev_active_index[indices_active[:k]]],self.bound[:,prev_active_index[indices_active[:k]]],self.model_class,**kwargs)
                    except ValueError as e:
                        if(e.args[0]=='Too many cov mat singular!'):
                            break
                        else:
                            raise ValueError(e.args[0])
                    if(sub_final_loss<=prev_loss):
                        prev_loss = sub_final_loss
                        prev_active_dim_remain-=1
                    else:
                        loss_interv = sub_final_loss - prev_loss
                        get_loss_interval+=1
                        break
                new_indices = prev_active_index[indices_active[:prev_active_dim_remain]]
                for j in range(self.X_dim):
                    if(self.indices[j] in new_indices):
                        continue
                    try:
                        if('variable_type' in kwargs.keys()):
                            kwargs = kwargs_old.copy()
                            kwargs['variable_type'] = kwargs_old['variable_type'][torch.cat([new_indices,torch.tensor([self.indices[j]])])]
                        _,_,_,_,sub_final_loss = self.fit_model(self.X[:,torch.cat([new_indices,torch.tensor([self.indices[j]])])],self.bound[:,torch.cat([new_indices,torch.tensor([self.indices[j]])])],self.model_class,**kwargs)
                    except ValueError as e:
                        if(e.args[0]=='Too many cov mat singular!'):
                            new_indices = torch.cat([new_indices,torch.tensor([self.indices[j]])])
                            continue
                        else:
                            raise ValueError(e.args[0])
                    if(get_loss_interval==0):
                        loss_interv = prev_loss - sub_final_loss
                        prev_loss = sub_final_loss
                        get_loss_interval+=1
                        new_indices = torch.cat([new_indices,torch.tensor([self.indices[j]])])
                        continue
                    if(prev_loss - sub_final_loss<loss_interv/10.0):
                        break
                    else:
                        loss_interv = prev_loss - sub_final_loss
                        prev_loss = sub_final_loss
                        new_indices = torch.cat([new_indices,torch.tensor([self.indices[j]])])
                self.active_f_dims = len(new_indices)
                self.active_f_list = torch.tensor([0 for k in range(self.X_dim)],dtype=torch.bool,device=device)
                self.active_f_list[new_indices] = 1
    def variable_selection_nomom(self,FS_score_method,*args, **kwargs):
        self.FS_score_method = FS_score_method
        self.GP_fitting(self.model_class,**kwargs)
        self.FS_important_scores = self.calc_important_score(self.model,self.FS_score_method,dim=self.X_dim,active_f_list=torch.tensor([1 for i in range(self.X_dim)],dtype=torch.bool,device=device),*args, **kwargs)
        kwargs_old = kwargs.copy()
        _,self.indices = torch.sort(self.FS_important_scores,descending=True)
        print(self.indices)
        self.stepwise_forward(0,torch.tensor([0 for k in range(self.X_dim)],dtype=torch.bool,device=device),**kwargs)
    def stepwise_forward_2(self,start_point,important_variables,**kwargs):
        #pdb.set_trace()
        get_loss_interval = -1
        if_fs = 0
        kwargs_new = kwargs.copy()
        for j in range(start_point,self.X_dim):
            if(important_variables[self.indices[j]]==1 and get_loss_interval==-1):
                continue
            try:
                if('variable_type' in kwargs.keys()):
                    kwargs_new['variable_type'] = kwargs['variable_type'][self.indices[:j+1]]
                _,_,_,_,sub_final_loss = self.fit_model(self.X[:,self.indices[:j+1]],self.bound[:,self.indices[:j+1]],self.model_class,**kwargs_new)
            except ValueError as e:
                if(e.args[0]=='Too many cov mat singular!'):
                    continue
                else:
                    raise ValueError(e.args[0])
            if(get_loss_interval==-1):
                prev_loss = sub_final_loss
                get_loss_interval+=1
            elif(get_loss_interval==0):
                loss_interv = prev_loss - sub_final_loss
                prev_loss = sub_final_loss
                get_loss_interval+=1
            else:
                if(loss_interv<=0 or prev_loss - sub_final_loss<loss_interv/10.0):
                    if_fs = 1
                    self.active_f_list = torch.tensor([0 for k in range(self.X_dim)],dtype=torch.bool,device=device)
                    self.active_f_list[self.indices[:j]] = 1
                    self.active_f_dims = j
                    break
                else:
                    loss_interv = prev_loss - sub_final_loss
                    prev_loss = sub_final_loss
        if not if_fs:
            self.active_f_dims = self.X_dim
            self.active_f_list = torch.tensor([1 for k in range(self.X_dim)],dtype=torch.bool,device=device)
    #use rtmvnorm in R to sample truncated multivariate normal distribution, use rpy2 to embed R code in python
    def truncated_multivariate_normal_sampling(self,mu,cov_mat,n_samp):
        #pdb.set_trace()
        mu = FloatVector(mu)
        x_dim,_ = cov_mat.shape
        cov = ro.r.matrix(cov_mat,nrow=x_dim,ncol=x_dim)
        lb = FloatVector(np.zeros(x_dim))
        ub = FloatVector(np.ones(x_dim))
        return np.array(tmvtnorm.rtmvnorm(n=n_samp,mean=mu,sigma=cov,lower=lb,upper=ub,algorithm='gibbs',burn=100, thinning = 5))
    def data_update(self,method='CMAES_posterior',n_sampling=1):
        #self.lessiv_n_sampling = n_sampling
        #pdb.set_trace()
        new_x = torch.tensor([0 for i in range(self.X_dim)],dtype=dtype,device=device).reshape((1,self.X_dim))
        new_x[:,self.active_f_list] = self.new_x_normalize_active
        if(self.X_dim>self.active_f_dims):
            #pdb.set_trace()
            if(method=='rand'):
                new_x[:,~self.active_f_list] = torch.rand(1, self.X_dim-self.active_f_dims, device=device, dtype=dtype)
            if(method=='mix'):
                #pdb.set_trace()
                rand_s = np.random.uniform(0,1)
                if(rand_s<=0.5):
                    new_x[:,~self.active_f_list] = torch.rand(1, self.X_dim-self.active_f_dims, device=device, dtype=dtype)
                else:
                    new_x[:,~self.active_f_list] = normalize(self.X,bounds=self.bound)[self.Y.argmax(),~self.active_f_list].reshape((1, self.X_dim-self.active_f_dims))
            elif(method=='CMAES_posterior'):
                CMA_cond_mean = self.conditiona_normal_list[0] + np.dot(np.dot(self.conditiona_normal_list[3],self.conditiona_normal_list[6]),(self.new_x_normalize_active.numpy().reshape((self.active_f_dims,))-self.conditiona_normal_list[1]))
                CMA_cond_cov = self.conditiona_normal_list[-1]
                self.new_x_normalize_inactive = self.truncated_multivariate_normal_sampling(CMA_cond_mean,CMA_cond_cov,n_sampling)
                #self.new_x_normalize_inactive = np.random.multivariate_normal(CMA_cond_mean,CMA_cond_cov,n_sampling)
                #x_arr,y_arr = np.where(self.new_x_normalize_inactive>1)
                #self.new_x_normalize_inactive[x_arr,y_arr] = 1
                #x_arr,y_arr = np.where(self.new_x_normalize_inactive<0)
                #self.new_x_normalize_inactive[x_arr,y_arr] = 0
                new_x_multi = new_x.repeat(n_sampling,1)
                new_x_multi[:,~self.active_f_list] = torch.tensor(self.new_x_normalize_inactive,device=device, dtype=dtype).reshape((n_sampling,self.X_dim-self.active_f_dims))
                if(n_sampling>1):
                    post = self.model.posterior(new_x_multi)
                    new_x = new_x_multi[post.mean.argmax()].reshape((1,self.X_dim))
                else:
                    new_x = new_x_multi
            else:
                print("The method to get the value of less important variables has not been implemented!")
        new_x = unnormalize(new_x,bounds=self.bound)
        new_y = self.obj_info['function'](new_x,**self.obj_info['function_kwarg'])
        self.X, self.Y = torch.cat((self.X,new_x)),torch.cat((self.Y,new_y))