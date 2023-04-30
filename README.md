Cods for the manuscript "Computationally Efficient High-Dimensional Bayesian Optimization via Variable Selection"

We use python==3.7 to run all the experiments. 

To install the dependencies as:

`pip install -r requirements.txt`

The detailed description of ALEBO can be found in `https://github.com/facebookresearch/alebo`. The dependencies required for ALEBO are already in the file `requirements.txt`. 

The detailed description of HeSBO can be found in `https://github.com/aminnayebi/HesBO`. The dependencies required for HeSBO are already in the file `requirements.txt`. 

To run SAASBO, please download the source codes: `git clone https://github.com/martinjankowiak/saasbo.git`. The dependencies required for SAASBO are already in the file `requirements.txt`. 

To run MOPTA08 function, first decompress the file `mopta_libs.zip`. There are two libraries, `libmopta.so` and `moptafunc.so`, in this `.zip` file, please put them into the same folder of the running script. For detailed instruction on compiling these two `.so` files, please see `https://gist.github.com/denis-bz/c951e3e59fb4d70fd1a52c41c3675187`.

To run VS-BO, use the following command: `python VSBO_run.py --obj_func Branin --method VSBO`, it will automatically create an output path `./Branin/VSBO/`. VS-BO has seven output files, and all these output files will be in the output path:

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

To plot the regret curve versus iterations, use the following command: `python plot_iterations.py --input_path ./Branin/VSBO/ --output_path ./ --method VSBO --runs 20 --iterations 200 --best_value -0.44165457`, it will generate a figure called `curve_iterations.png`. 

To plot the regret curve versus wall clock times, use the following command: `python plot_times.py --input_path ./Branin/VSBO/ --output_path ./ --method VSBO --runs 20 --T_max 600 --best_value -0.44165457 --wct`, it will generate a figure called `curve_wct.png`. 

To plot the regret curve versus CPU times, use the following command: `python plot_times.py --input_path ./Branin/VSBO/ --output_path ./ --method VSBO --runs 20 --T_max 8000 --best_value -0.44165457`, it will generate a figure called `curve_cput.png`. 

Please choose T_max to be an integer value such that it can be divided by 100. 
