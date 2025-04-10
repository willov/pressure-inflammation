import copy
import os
import json 
from multiprocessing import Pool
import random 

import psutil
import numpy as np
from scipy.stats import chi2
import sund 
from tqdm import tqdm

from functions.utils import load_and_process_data, setup_simulations, get_dof, load_best_parameters, NumpyPandasArrayEncoder
from functions.optimization import f_cost, f_cost_parameter_identifiability_log, optimize, optimize_parameter_identifiability
from functions.plotting import plot_agreement, print_parameter_table

# %% Settings
DO_OPTIMIZATION = False
DO_PI = False
RUN_IN_PARALLEL = True
NUM_CORES  =  psutil.cpu_count(logical=False)
NUM_OPTIMIZATIONS = 500
NUM_PI_OPTIMIZATIONS = 1000

# %% Load the data
data = load_and_process_data("Data/data.json")
data.pop("control", None) # Remove the data for the control experiment, since it is not used for model training/evaluation

all_important_cytokines = ["VEGF-A", "CSF1", "IL-2Ra", "CHI3L1", "IL-9", "CX3CL1", "IL-1RA", "IL-1α", "IL-1β", "CCL3", "IL-27","IFN-α2a", "IL-23", "IL-7", "CSF3", "IL-15", "IL-25"]
keys_to_keep = ["meta", "info", "input", "extra", "all_times"] + all_important_cytokines
for key in list(data["mask application"].keys()):
    if key not in keys_to_keep:
        data["mask application"].pop(key, None)
        
# Count the number of non-inf SEM values in the data
dof = get_dof(data)

# %% Setup model and simulation
model_name = "M6_8"
sund.install_model(f"Models/{model_name}.txt")
model = sund.load_model(model_name)

sims, θ0, parameter_names = setup_simulations(model)

fixed_parameters = [] # List of parameter names to fix (i.e. "ignore" in the optimization)

θ0 = load_best_parameters(f"./Results/{model.name}", param_key="θopt", cost_key='cost', model=model) # The best parameters found so far with respect to the agreement to data

cost,_ = f_cost(θ0, sims, data, False)
print(f"Cost of initial guess: {cost}")

# %% Optimize the parameters

if DO_OPTIMIZATION:
    print("Starting optimization")
    if RUN_IN_PARALLEL:
        with Pool(processes=min(NUM_CORES, psutil.cpu_count(logical=False))) as pool:
            with tqdm(total=NUM_OPTIMIZATIONS) as pbar:
                for i in range(NUM_OPTIMIZATIONS):
                    pool.apply_async(optimize, args=(model, data, None, fixed_parameters), callback=lambda _: pbar.update())
            pool.close()
            pool.join()
    else:
        for i in range(NUM_OPTIMIZATIONS):
            print(f"Starting optimization nr {i+1}")

            θopt, cost_opt = optimize(model, data, θ0, fixed_parameters=fixed_parameters)

            θ0 = θopt.copy()

elif DO_PI:
        # If the current model has not been tested, save an initial guess for all parameters
    if not os.path.exists(f"./Results_PI/{model.name}"):
        θ_best = load_best_parameters(f"./Results/{model.name}", param_key="θopt", cost_key='cost', model=model) # The best parameters found so far with respect to the agreement to data
        os.makedirs(f"./Results_PI/{model.name}")
        for p_idx, p_name in enumerate(model.parameternames):
            cost = θ_best[p_idx]
            try:
                with open(f"./Results_PI/{model.name}/{model.name}-{p_name}-initial-({cost:e}).json", 'w') as file:
                    out = {"cost": cost, "θopt": θ_best}
                    json.dump(out, file, cls=NumpyPandasArrayEncoder)
            except PermissionError:
                print(f"Permission denied for file './Results_PI/{model.name}/{model.name}-{p_name}-initial-({cost:e}).json'. Not saving current solution.")

    fixed_parameters = "all"
    if RUN_IN_PARALLEL:
        with Pool(processes=max(NUM_CORES, psutil.cpu_count(logical=False))) as pool:
            with tqdm(total=NUM_PI_OPTIMIZATIONS) as pbar:
                for i in range(NUM_PI_OPTIMIZATIONS):
                    direction = random.choice([1, -1]) # v = direction*θ[i], so -1 = maximize, 1 = minimize
                    pool.apply_async(optimize_parameter_identifiability, (model, data, fixed_parameters, direction), callback=lambda _: pbar.update())
            pool.close()
            pool.join()
    else:
        for i in range(NUM_PI_OPTIMIZATIONS):
            print(f"Starting optimization nr {i+1}")
            direction = random.choice([1, -1]) # v = direction*θ[i], so -1 = maximize, 1 = minimize
            θopt, cost_opt = optimize_parameter_identifiability(model, data, fixed_parameters=fixed_parameters, direction=direction)


# %% Load the best parameters
θopt = load_best_parameters(f"./Results/{model.name}", param_key="θopt", cost_key='cost', model=model) # The best parameters found so far with respect to the agreement to data
cost_opt,_ = f_cost(θopt, sims, data, print_costs=True)

# %% Plot the best simulation
print(f"θopt = [{', '.join(map(str, θopt))}]")
print(f"\nCost: {cost_opt} for model {model_name}.")
# dof -= len(model.parameternames)
print(f"Reject the model? {cost > chi2.ppf(1-0.05, dof)} (limit = {chi2.ppf(1-0.05, dof)}, df={dof})")

plot_agreement(θopt, sims, data, model_name)

# %% Plot the parameter identifiability table

if os.path.exists(f"./Results_PI/{model.name}"):

    θ_all = [θopt]

    # find all .json files in the directory
    files = os.listdir(f"./Results_PI/{model.name}")

    for file in files:
        if file.endswith(".json"):
            with open(f"./Results_PI/{model.name}/{file}") as f:
                loaded_params = json.load(f)
            if "θopt" in loaded_params.keys() and "cost" in loaded_params.keys():
                cost_ident = f_cost_parameter_identifiability_log(np.log(loaded_params["θopt"]), sims, data, 0, -1, chi2.ppf(1-0.05, dof), θopt[0])
                cost,_ = f_cost(loaded_params["θopt"], sims, data, print_costs=False)
                if cost_ident > loaded_params["cost"] or cost > chi2.ppf(1-0.05, dof):
                    print(f"removing {file}")
                    os.remove(f"./Results_PI/{model.name}/{file}")
                else:
                    θ_all.append(loaded_params["θopt"])

    
    plot_agreement(θ_all, sims, data, model_name, figure_name="agreement_PI")


    print_parameter_table(model)
