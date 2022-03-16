from VSBO_utils import *
from ax.utils.measurement.synthetic_functions import branin
from ax.modelbridge.strategies.alebo import ALEBOStrategy
from ax.modelbridge.strategies.rembo import HeSBOStrategy

from ax.service.managed_loop import optimize

def convert_botorch_bounds_to_ax(botorch_bounds):
    domain_bounds = []
    dim = botorch_bounds.size(1)
    for  i in range(dim):
        domain_bounds.append({"name":"x"+str(i), "type": "range", "bounds": [float(botorch_bounds[0,i]), float(botorch_bounds[1,i])], "value_type": "float"})
    return domain_bounds


def convert_para_to_X(experiment_info,trial_dim,obj_dim):
    X = np.zeros((trial_dim,obj_dim))
    i=0
    for trial in experiment.trials.values():
        for j in range(obj_dim):
            X[i,j] = trial.arm.parameters["x"+str(j)]
        i = i+1
    return X

