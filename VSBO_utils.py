import math
import torch
import gpytorch
from abc import ABC
from copy import deepcopy
from typing import Dict, Optional, Tuple, Union
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, Kernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from torch.optim import SGD, LBFGS
import warnings
import pdb,io
import sys,os
from botorch.fit import fit_gpytorch_model
from botorch.test_functions import Branin, Hartmann, Bukin, Cosine8, EggHolder, Griewank, Powell, Rosenbrock, Levy, Ackley, StyblinskiTang
from botorch.utils.transforms import standardize, normalize, unnormalize
from botorch.optim import optimize_acqf
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective
from botorch.exceptions import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.posteriors.posterior import Posterior
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import convert_to_target_pre_hook, t_batch_mode_transform
from torch import Tensor
from torch.distributions import Normal
from gpytorch.constraints import Positive
from gpytorch.lazy import MatmulLazyTensor, RootLazyTensor, delazify
import numpy as np
from botorch.acquisition import UpperConfidenceBound, PosteriorMean, ExpectedImprovement
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.utils.sampling import draw_sobol_normal_samples
import random
import matplotlib.pyplot as plt
from functools import cmp_to_key
from multiprocessing import Manager
import multiprocessing
import subprocess
import botorch
from botorch.utils.sampling import draw_sobol_samples
from torch.autograd import Variable
import cma
import time
from functools import wraps
import signal,psutil
from scipy.stats import special_ortho_group

import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
temp_model = None


class GP_Matern(ExactGP,GPyTorchModel):
    _num_outputs = 1
    def __init__(self,train_X,train_Y,if_noise=False,**kwargs):
        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=MaternKernel(ard_num_dims=train_X.shape[-1]),)
        if if_noise:
            self.covar_module = NoiseKernel(self.covar_module)
        self.to(train_X)
    def forward(self,X):
        mean_X = self.mean_module(X)
        covar_X = self.covar_module(X)
        return MultivariateNormal(mean_X, covar_X) 


def catch_error(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        pid = os.getpid()
        while True:
            try:
                return func(*args,**kwargs)
            except RuntimeError as e:
                #pdb.set_trace()
                print("get runtimeerror: cov mat singular!")
                #torch.save(temp_model.state_dict(),"./"+str(pid)+"_bad_model")
                if 'rand_init' in kwargs.keys():
                    kwargs['rand_init'] = kwargs['rand_init'] + 1
                else:
                    kwargs['rand_init'] = 1
    return decorated


def get_conditional_normal(mu,cov,variable):
    #pdb.set_trace()
    u1 = mu[variable]
    u2 = mu[~variable]
    cov11 = cov[variable][:,variable]
    cov12 = cov[variable][:,~variable]
    cov21 = cov[~variable][:,variable]
    cov22 = cov[~variable][:,~variable]
    cov22_inv = np.linalg.inv(cov22)
    #cond_mu = u1 + np.dot(np.dot(cov12,cov22_inv),(cond_vec-u2))
    cond_cov = cov11 - np.dot(np.dot(cov12,cov22_inv),cov21)
    return [u1,u2,cov11,cov12,cov21,cov22,cov22_inv,cond_cov]

def Branin_hd(X):
    neg_branin = Branin(negate=True)
    return neg_branin(X[:,:2])

def generate_branin_bounds(d):
    if(d<2):
        print("Dimension Error: Branin!")
        #sys.exit(1)
    bounds = torch.tensor([[-5,0],[10,15]],dtype=dtype, device=device)
    if(d==2):
        return bounds
    irrelavant_vari = torch.stack([-torch.ones(d-2, dtype=dtype, device=device),torch.ones(d-2, dtype=dtype, device=device)])
    return torch.cat([bounds,irrelavant_vari],dim=1)

def StyblinskiTang4_hd(X):
    neg_st = StyblinskiTang(dim=4,negate=True)
    return neg_st(X[:,:4])

def generate_StyblinskiTang_bounds(d):
    if(d<2):
        print("Dimension Error: StyblinskiTang!")
    return torch.stack([-5*torch.ones(d, dtype=dtype, device=device),5*torch.ones(d, dtype=dtype, device=device)])


def Hartmann6(X):
    neg_hartmann6 = Hartmann(dim=6,negate=True)
    return neg_hartmann6(X[:,:6])

def generate_hartmann_bounds(d):
    if(d<6):
        print("Dimension Error: hartmann!")
    return torch.stack([torch.zeros(d, dtype=dtype, device=device),torch.ones(d, dtype=dtype, device=device)])


def Combine_func(X,func_set,var_num_set,coeff_set):
    l = len(func_set)
    if(len(var_num_set)!=l or len(coeff_set)!=l or l==0):
        print("Combine functions Error!")
        return None
    func_val = 0
    for i in range(l):
        func_val += coeff_set[i]*func_set[i](X[:,sum(var_num_set[:i]):sum(var_num_set[:i+1])])
    return func_val

def FS_ARD(model):
    #pdb.set_trace()
    ARD = 1/model.covar_module.base_kernel.lengthscale[0]
    return (ARD/torch.max(ARD))

def FS_KLrel(model,dim,sampling_num=10000,if_round=[],**kwargs):
    #pdb.set_trace()
    #generate random data
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    X_rand = draw_sobol_samples(bounds,sampling_num,1).reshape(sampling_num,dim).type(dtype).to(device)
    #pdb.set_trace()
    if(len(if_round)!=0):
        cur_bound,inter_f_info = if_round
        X_rand = unnormalize(X_rand,cur_bound)
        X_rand[:,inter_f_info] = torch.round(X_rand[:,inter_f_info])
        X_rand = normalize(X_rand,bounds=cur_bound)
    X_rand = Variable(X_rand,requires_grad=True)
    KLrel_vec = KLrel_new(X_rand,model)
    #print(KLrel_vec)
    return KLrel_vec/torch.max(KLrel_vec)

def KLrel_new(X,model):
    #pdb.set_trace()
    n = X.shape[0]
    p = X.shape[1]
    post = model.posterior(X)
    X_grad = torch.abs(torch.autograd.grad(post.mean.sum(),X,retain_graph=True)[0])
    post_var_denominator = post.variance.reshape((n,1)).repeat(1,p)
    X_grad = X_grad/torch.sqrt(post_var_denominator)
    return torch.mean(X_grad,axis=0)