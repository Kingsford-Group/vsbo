The source codes of SAASBO are downloaded from `https://github.com/martinjankowiak/saasbo`. We revised the codes so that it can record time usages. 

To run SAASBO, use the command `python saasbo_run.py --obj_func Branin`. It will automatically create an output path `./Branin/saasbo`. 

To plot the regret curve versus iterations, use the following command: `python plot_iterations.py --input_path ./Branin/saasbo/ --output_path ./ --method saasbo --runs 10 --iterations 200 --best_value -0.44165457 --not_max`, it will generate a figure called `curve_iterations.png`. 

To plot the regret curve versus wall clock times, use the following command: `python plot_times.py --input_path ./Branin/saasbo/ --output_path ./ --method saasbo --runs 10 --T_max 600 --best_value -0.44165457 --wct --not_max --not_accumuate`, it will generate a figure called `curve_wct.png`. 

To plot the regret curve versus CPU times, use the following command: `python plot_times.py --input_path ./Branin/saasbo/ --output_path ./ --method saasbo --runs 10 --T_max 8000 --best_value -0.44165457 --not_max --not_accumuate`, it will generate a figure called `curve_cput.png`. 
