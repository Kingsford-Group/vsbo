# Codes are adapted from the author: Jan Hendrik Metzen <janmetzen@mailbox.org>

from itertools import cycle

from sklearn.utils import check_random_state

#from .utils.optimization import global_optimization
from VSBO_utils import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#dtype = torch.double


class BayesianOptimizer(object):
    """Bayesian optimization for global black-box optimization

    Bayesian optimization models the landscape of the function to be optimized
    internally by a surrogate model (typically a Gaussian process) and
    evaluates always those parameters which are considered as global optimum
    of an acquisition function defined over this surrogate model. Different
    acquisition functions and optimizers can be used internally.

    Bayesian optimization aims at reducing the number of evaluations of the
    actual function, which is assumed to be costly. To achieve this, a large
    computational budget is allocated at modelling the true function and finding
    potentially optimal positions based on this model.

    .. seealso:: Brochu, Cora, de Freitas
                 "A tutorial on Bayesian optimization of expensive cost
                  functions, with application to active user modelling and
                  hierarchical reinforcement learning"

    Parameters
    ----------
    model : surrogate model object
        The surrogate model which is used to model the objective function. It
        needs to provide a methods fit(X, y) for training the model and
        predictive_distribution(X) for determining the predictive distribution
        (mean, std-dev) at query point X.

    acquisition_function : acquisition function object
        When called, this function returns the acquisitability of a query point
        i.e., how favourable it is to perform an evaluation at the query point.
        For this, internally the trade-off between exploration and exploitation
        is handled.

    optimizer: string, default: "direct"
        The optimizer used to identify the maximum of the acquisition function.
        The optimizer is specified by a string which may be any of "direct",
        "direct+lbfgs", "random", "random+lbfgs", "cmaes", or "cmaes+lbfgs".

    maxf: int, default: 1000
        The maximum number of evaluations of the acquisition function by the
        optimizer.

    initial_random_samples: int, default: 5
        The number of initial sample, in which random query points are selected
        without using the acquisition function. Setting this to values larger
        than 0 might be required if the surrogate model needs to be trained
        on datapoints before evaluating it.

    random_state : RandomState or int (default: None)
        Seed for the random number generator.
    """
    def __init__(self, GP_model,
                 maxf=1000, initial_random_samples=5, random_state=0,
                 *args, **kwargs):
        self.GP_model = GP_model
        #self.acquisition_function = acquisition_function
        self.maxf = maxf
        self.initial_random_samples = initial_random_samples

        self.rng = check_random_state(random_state)

        self.X_ = []
        self.y_ = []

    '''
    def select_query_point(self, boundaries,
                           incumbent_fct=lambda y: np.max(y)):
        """ Select the next query point in boundaries based on acq. function.

        Parameters
        ----------
        boundaries : ndarray-like, shape: [n_dims, 2]
            Box constraint on allowed query points. First axis corresponds
            to dimensions of the search space and second axis to minimum and
            maximum allowed value in the respective dimensions.

        incumbent_fct: function, default: returns maximum observed value
            A function which is used to determine the incumbent for the
            acquisition function. Defaults to the maximum observed value.
        """
        boundaries = np.asarray(boundaries)

        if len(self.X_) < self.initial_random_samples:
            X_query = self.rng.uniform(size=boundaries.shape[0]) \
                * (boundaries[:, 1] - boundaries[:, 0]) + boundaries[:, 0]
        else:
            self.acquisition_function.set_boundaries(boundaries)

            def objective_function(x):
                # Check boundaries
                if not np.all(np.logical_and(x >= boundaries[:, 0],
                                             x <= boundaries[:, 1])):
                    return -np.inf

                incumbent = incumbent_fct(self.y_)
                return self.acquisition_function(x, incumbent=incumbent)

            X_query = global_optimization(
                objective_function, boundaries=boundaries,
                optimizer=self.optimizer, maxf=self.maxf, random=self.rng)

        # Clip to hard boundaries
        return np.clip(X_query, boundaries[:, 0], boundaries[:, 1])
    '''

    '''
    def update(self, X, y):
        """ Update internal model for observed (X, y) from true function. """
        self.X_.append(X)
        self.y_.append(y)
        self.model.fit(self.X_, self.y_)
    '''

    def best_params(self):
        """ Returns the best parameters found so far."""
        return self.X_[torch.argmax(self.y_)]

    def best_value(self):
        """ Returns the optimal value found so far."""
        return torch.max(self.y_)


class REMBOOptimizer(BayesianOptimizer):
    """ Random EMbedding Bayesian Optimization (REMBO).

    This extension of Bayesian Optimizer (BO) is better suited for
    high-dimensional problems with a low effective dimensionality than BO.
    This is achieved by restricting the optimization to a low-dimensional
    linear manifold embedded in the higher dimensional space. Theoretical
    results suggest that even if the manifold is chosen randomly, the
    optimum on this manifold equals the global optimum if the function is
    indeed of the same intrinsic dimensionality as the manifold.

    .. seealso:: Wang, Zoghi, Hutter, Matheson, de Freitas
                 "Bayesian Optimization in High Dimensions via Random
                 Embeddings", International Joint Conferences on Artificial
                 Intelligence (IJCAI), 2013

    Parameters
    ----------
    n_dims : int
        The dimensionality of the actual search space

    n_embedding_dims : int, default: 2
        The dimensionality of the randomly chosen linear manifold on which the
        optimization is performed

    data_space: array-like, shape=[n_dims, 2], default: None
        The boundaries of the data-space. This is used for scaling the mapping
        from embedded space to data space, which is useful if dimensions of the
        data space have different ranges or are not centred around 0.

    n_keep_dims : int, default: 0
        The number of dimensions which are not embedded in the manifold but are
        kept 1-to-1 in the representation. This can be useful if some
        dimensions are known to be relevant. Note that it is expected that
        those dimensions come first in the data representation, i.e., the first
        n_keep_dims dimensions are maintained.

    Further parameters are the same as in BayesianOptimizer
    """

    def __init__(self, n_dims, n_embedding_dims=2, data_space=None,
                 n_keep_dims=0, *args, **kwargs):
        super(REMBOOptimizer, self).__init__(*args, **kwargs)

        self.n_dims = n_dims
        self.n_embedding_dims = n_embedding_dims
        self.data_space = data_space
        self.n_keep_dims = n_keep_dims
        if self.data_space is not None:
            self.data_space = np.asarray(self.data_space)
            if self.data_space.shape[0] != self.n_dims - n_keep_dims:
                raise Exception("Data space must be specified for all input "
                                "dimensions which are not kept.")

        # Determine random embedding matrix
        self.A = self.rng.normal(size=(self.n_dims - self.n_keep_dims,
                                       self.n_embedding_dims))
        #self.A /= np.linalg.norm(self.A, axis=1)[:, np.newaxis]  # XXX

        self.X_embedded_ = []
        #self.y_standard = []
        self.boundaries_cache = {}

    def model_initialise(self,boundaries):
        #pdb.set_trace()
        self.boundaries = np.asarray(boundaries)
        if not boundaries.shape[0] == self.n_dims:
            raise Exception("Dimensionality of boundaries should be %d" % self.n_dims)
        
        self.boundaries_embedded = self._compute_boundaries_embedding(boundaries)

        self.X_embedded_ = unnormalize(torch.rand(self.initial_random_samples, self.n_embedding_dims, device=device, dtype=dtype), bounds=torch.tensor(self.boundaries_embedded.T,device=device, dtype=dtype))
        self.X_ = torch.zeros((self.initial_random_samples,self.n_dims),device=device, dtype=dtype)
        for i in range(self.initial_random_samples):
            X_query_embedded = self.X_embedded_[i].numpy()
            X_query = np.clip(self._map_to_dataspace(X_query_embedded),self.boundaries[:, 0], self.boundaries[:, 1])
            self.X_[i] = torch.tensor(X_query,device=device, dtype=dtype)

    def select_query_point(self):
        """ Select the next query point in boundaries based on acq. function.

        Parameters
        ----------
        boundaries : ndarray-like, shape: [n_dims, 2]
            Box constraint on allowed query points. First axis corresponds
            to dimensions of the search space and second axis to minimum and
            maximum allowed value in the respective dimensions.

        incumbent_fct: function, default: returns maximum observed value
            A function which is used to determine the incumbent for the
            acquisition function. Defaults to the maximum observed value.
        """

        # Compute boundaries on embedded space
        #boundaries_embedded = self._compute_boundaries_embedding(boundaries)

        # Select query point by finding optimum of acquisition function
        # within boundaries
        #pdb.set_trace()

        y_standard = standardize(self.y_)
        self.acquisition_function = ExpectedImprovement(self.model,best_f=y_standard.max().item())

        X_query_embedded,_ = optimize_acqf(
            acq_function = self.acquisition_function,
            bounds = torch.stack([
                torch.zeros(self.n_embedding_dims, dtype=dtype, device=device),
                torch.ones(self.n_embedding_dims, dtype=dtype, device=device),
            ]),
            q=1,
            num_restarts = 10,
            raw_samples = 200,
        )

        X_query_embedded = unnormalize(X_query_embedded.detach(),bounds=torch.tensor(self.boundaries_embedded.T,device=device, dtype=dtype))

        self.X_embedded_ = torch.cat((self.X_embedded_,X_query_embedded))

        #self.X_embedded_.append(X_query_embedded)

        # Map to higher dimensional space and clip to hard boundaries
        X_query = np.clip(self._map_to_dataspace(X_query_embedded[0].numpy()),
                          self.boundaries[:, 0], self.boundaries[:, 1])
        return torch.tensor(X_query,device=device, dtype=dtype).reshape((1,self.n_dims))

    def erase_last_instance_embedded(self):
        self.X_embedded_ = self.X_embedded_[:-1]

    def update(self, X, y):
        """ Update internal model for observed (X, y) from true function. """
        self.X_ = torch.cat((self.X_,X))
        self.y_ = torch.cat((self.y_,y))
        return self.fit_model()
    
    def get_loss(self,model,likelihood_mode,X):
        model.train()
        output = model(X)
        return - likelihood_mode(output, model.train_targets)
    
    '''
    def add_data(self,X_new,Y_new):
        X_new_embedded = np.clip(X_new.numpy().dot(self.A),self.boundaries_embedded[:,0],self.boundaries_embedded[:,1])
        self.X_embedded_ = torch.cat((self.X_embedded_,torch.tensor(X_new_embedded,device=device, dtype=dtype)))
        self.X_ = torch.cat((self.X_,X_new))
        self.y_ = torch.cat((self.y_,Y_new))
    '''

    @catch_error
    def fit_model(self,rand_init=False):
        #pdb.set_trace()
        X_embedded_normalize = normalize(self.X_embedded_,bounds=torch.tensor(self.boundaries_embedded.T,device=device, dtype=dtype))
        y_standard = standardize(self.y_)
        model = self.GP_model(X_embedded_normalize,y_standard)
        if rand_init:
            model.covar_module.base_kernel.lengthscale = torch.rand(model.covar_module.base_kernel.lengthscale.size(), device=device, dtype=dtype)
            model.covar_module.outputscale = torch.rand(model.covar_module.outputscale.size(),device=device, dtype=dtype)
            model.likelihood.noise = 1
        mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        mll = mll.to(X_embedded_normalize)
        init_loss = self.get_loss(model,mll,X_embedded_normalize)
        fit_gpytorch_model(mll)
        final_loss = self.get_loss(model,mll,X_embedded_normalize)
        self.model = model
        return init_loss,final_loss

    def _map_to_dataspace(self, X_embedded):
        """ Map data from manifold to original data space. """
        X_query_kd = self.A.dot(X_embedded[self.n_keep_dims:])
        if self.data_space is not None:
            X_query_kd = (X_query_kd + 1) / 2 \
                * (self.data_space[:, 1] - self.data_space[:, 0]) \
                + self.data_space[:, 0]
        X_query = np.hstack((X_embedded[:self.n_keep_dims], X_query_kd))

        return X_query

    def _compute_boundaries_embedding(self, boundaries):
        """ Approximate box constraint boundaries on low-dimensional manifold"""
        # Check if boundaries have been determined before
        boundaries_hash = hash(boundaries[self.n_keep_dims:].tostring())
        if boundaries_hash in self.boundaries_cache:
            boundaries_embedded = \
                np.array(self.boundaries_cache[boundaries_hash])
            boundaries_embedded[:self.n_keep_dims] = \
                boundaries[:self.n_keep_dims]  # Overwrite keep-dim's boundaries
            return boundaries_embedded

        # Determine boundaries on embedded space
        boundaries_embedded = \
            np.empty((self.n_keep_dims + self.n_embedding_dims, 2))
        boundaries_embedded[:self.n_keep_dims] = boundaries[:self.n_keep_dims]
        for dim in range(self.n_keep_dims,
                         self.n_keep_dims + self.n_embedding_dims):
            x_embedded = np.zeros(self.n_keep_dims + self.n_embedding_dims)
            while True:
                x = self._map_to_dataspace(x_embedded)
                if np.sum(np.logical_or(
                    x[self.n_keep_dims:] < boundaries[self.n_keep_dims:, 0],
                    x[self.n_keep_dims:] > boundaries[self.n_keep_dims:, 1])) \
                   > (self.n_dims - self.n_keep_dims) / 2:
                    break
                x_embedded[dim] -= 0.01
            boundaries_embedded[dim, 0] = x_embedded[dim]

            x_embedded = np.zeros(self.n_keep_dims + self.n_embedding_dims)
            while True:
                x = self._map_to_dataspace(x_embedded)
                if np.sum(np.logical_or(
                    x[self.n_keep_dims:] < boundaries[self.n_keep_dims:, 0],
                    x[self.n_keep_dims:] > boundaries[self.n_keep_dims:, 1])) \
                   > (self.n_dims - self.n_keep_dims) / 2:
                    break
                x_embedded[dim] += 0.01
            boundaries_embedded[dim, 1] = x_embedded[dim]

        self.boundaries_cache[boundaries_hash] = boundaries_embedded

        return boundaries_embedded


class InterleavedREMBOOptimizer(BayesianOptimizer):
    """ Interleaved Random EMbedding Bayesian Optimization (REMBO).

    In this extension of REMBO, several different random embeddings are chosen
    and the optimization is performed on all embeddings interleaved (in a
    round-robin fashion). This way, the specific choice of one random embedding
    becomes less relevant. On the other hand, less evaluations on each
    particular embedding can be performed.

    .. seealso:: Wang, Zoghi, Hutter, Matheson, de Freitas
                 "Bayesian Optimization in High Dimensions via Random
                 Embeddings", International Joint Conferences on Artificial
                 Intelligence (IJCAI), 2013

    Parameters
    ----------
    interleaved_runs : int
        The number of interleaved runs (each on a different random embedding).
        This parameter is denoted as k by Wang et al.

    Further parameters are the same as in REMBOOptimizer
    """

    def __init__(self, interleaved_runs=2, *args, **kwargs):
        random_state = kwargs.pop("random_state", 0)

        self.rembos = [REMBOOptimizer(random_state=random_state + 100 + run,
                                      *args, **kwargs)
                       for run in range(interleaved_runs)]
        #self.rembos_cycle = cycle(self.rembos)
        #self.current_rembo = next(self.rembos_cycle)

        self.X_ = []
        self.y_ = []

    def get_cycle(self):
        self.rembos_cycle = cycle(self.rembos)
        self.current_rembo = next(self.rembos_cycle)

    def select_query_point(self):
        """ Select the next query point in boundaries based on acq. function.

        Parameters
        ----------
        boundaries : ndarray-like, shape: [n_dims, 2]
            Box constraint on allowed query points. First axis corresponds
            to dimensions of the search space and second axis to minimum and
            maximum allowed value in the respective dimensions.

        incumbent_fct: function, default: returns maximum observed value
            A function which is used to determine the incumbent for the
            acquisition function. Defaults to the maximum observed value.
        """
        return self.current_rembo.select_query_point()

    def update(self, X, y):
        """ Update internal REMBO responsible for observed (X, y). """
        #self.X_.append(X)
        #self.y_.append(y)
        #pdb.set_trace()
        if(self.X_ == []):
            self.X_ = X
            self.y_ = y
        else:
            self.X_ = torch.cat((self.X_,X))
            self.y_ = torch.cat((self.y_,y))

        _,_ = self.current_rembo.update(X, y)
        self.current_rembo = next(self.rembos_cycle)