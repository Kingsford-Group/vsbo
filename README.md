Cods for the manuscript "Computationally Efficient High-Dimensional Bayesian Optimization via Variable Selection"

We use python==3.7 to run all the experiments. 

To install the dependencies as:

`pip install -r requirements.txt`

The detailed description of ALEBO can be found in `https://github.com/facebookresearch/alebo`. The dependencies required for ALEBO are already in the file `requirements.txt`. 

The detailed description of ALEBO can be found in `https://github.com/aminnayebi/HesBO`. The dependencies required for HeSBO are already in the file `requirements.txt`. 

To run SAASBO, please download the source codes: `git clone https://github.com/martinjankowiak/saasbo.git`. The dependencies required for SAASBO are already in the file `requirements.txt`. 

To run MOPTA08 function, first decompress the file `mopta_libs.zip`. There are two libraries, `libmopta.so` and `moptafunc.so`, in this `.zip` file, please put them into the same folder of the running script. For detailed instruction on compiling these two `.so` files.


The file "VSBO_class.py" implements the VS-BO algorithm, the file "Experiments_script.py" shows how to run VS-BO as well as other baselines except for SAASBO, the codes for running SAASBO is in saasbo_script.py, and the file "plot_script.py" contains codes for drawing the figures in the manuscript. 


VS-BO has seven output files:

* `X_*.npy`: The query obtained for each iteration
* `Y_*.npy`: The output value obtained for each iteration
* `Time_*.npy`: The accumulated wall clock time for each iteration
* `Time_process_*.npy`: The accumulated CPU time for each iteration
* `F_importance_val_*.npy`: The importance scores of variables for each variable selection step
* `F_rank_*.npy`: Sorted variables for each variable selection step based on their importance scores
* `F_chosen_*.npy`: the chosen variables for each variable selection step. 


Every other baseline has four output files: 

* `X_*.npy`
* `Y_*.npy`
* `Time_*.npy`
* `Time_process_*.npy`
