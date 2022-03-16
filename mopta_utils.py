from VSBO_utils import *
from moptafunc import func_and_constr


def mopta_func(X):
    n = X.size(0)
    Y = torch.zeros(n,device=device, dtype=dtype)
    for i in range(n):
        c = np.zeros( 68 )
        f = func_and_constr(X[i].detach().numpy(), c)
        Y[i] = -(f + 10*np.sum(np.maximum(c,np.zeros(68))))
    return Y

def generate_mopta_bounds():
    return torch.stack([torch.zeros(124, dtype=dtype, device=device),torch.ones(124, dtype=dtype, device=device)])