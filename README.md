Cods for the manuscript "Computationally Efficient High-Dimensional Bayesian Optimization via Variable Selection"

The following python packages should be installed before using VS-BO: 

BoTorch: https://botorch.org
pycma: https://github.com/CMA-ES/pycma
rpy2: https://rpy2.github.io (since the R package tmvtnorm is used)

To run ALEBO and HeSBO, Ax should be installed: https://ax.dev

The fortran codes of MOPTA08 problem are included in this folder (mopta08_part1.zip, mopta08_part2.zip), please refer to https://gist.github.com/denis-bz/c951e3e59fb4d70fd1a52c41c3675187 for compiling it into python. 

The file "VSBO_class.py" implements the VS-BO algorithm, the file "Experiments_script.py" shows how to run VS-BO as well as other baselines except for SAASBO, the codes for running SAASBO is in saasbo_script.py, and the file "plot_script.py" contains codes for drawing the figures in the manuscript. 


VS-BO has seven output files:

X_*.npy: The query obtained for each iteration
Y_*.npy: The output value obtained for each iteration
Time_*.npy: The accumulated wall clock time for each iteration
Time_process_*.npy: The accumulated CPU time for each iteration
F_importance_val_*.npy: The importance scores of variables for each variable selection step
F_rank_*.npy: Sorted variables for each variable selection step based on their importance scores
F_chosen_*.npy: the chosen variables for each variable selection step. 


Every other baseline has four output files: 

X_*.npy
Y_*.npy
Time_*.npy
Time_process_*.npy