import json
import os
import random 
from datetime import datetime
import re

import numpy as np
import pandas as pd

from scipy.stats import qmc, chi2
from scipy.optimize import Bounds, differential_evolution

import sund 

from functions.utils import NumpyPandasArrayEncoder, setup_simulations, get_dof, load_best_parameters

def get_parameter_bounds(param_names, theta, strict_bounds_parameters=[], fixed_parameters=[]):

    lb = [np.log(1.0e-5)]*(len(theta))
    ub = [np.log(1.0e5)]*(len(theta))

    if "n" in param_names:
        lb[param_names.index("n")] = np.log(1)
        ub[param_names.index("n")] = np.log(3)
    

    for param_name in strict_bounds_parameters:
        lb[param_names.index(param_name)] = np.log(theta[param_names.index(param_name)]*np.array(0.95))
        ub[param_names.index(param_name)] = np.log(theta[param_names.index(param_name)]*np.array(1.05))

    for param_name in fixed_parameters:
        if param_name in param_names:
            lb[param_names.index(param_name)] = np.log(theta[param_names.index(param_name)])
            ub[param_names.index(param_name)] = np.log(theta[param_names.index(param_name)])
        else:
            print(f"Warning: Parameter {param_name} is not in the parameter list.")

        if param_name in strict_bounds_parameters:
            print(f"Warning: Parameter {param_name} is both in fixed and strict bounds list. Fixed value will be used.")

    bounds = Bounds(lb, ub)
    return bounds


def f_cost(θ, sim, data, print_costs=False):
    cost = 0 
    all_costs = {}
    try:
        sim["steady"].simulate(t=np.linspace(0, 24*60, 1000), theta=θ, reset=True)
        for experiment,d in data.items():
            times = data[experiment]["all_times"]
            sim[experiment].simulate(t=times, x0=sim["steady"].state_values, theta=θ)

            feature_names = sim[experiment].feature_names
            features = sim[experiment].feature_data
            for observable, obs in d.items():
                if observable not in ["input", "meta", "extra", "all_times"]:
                    idx = feature_names.index(observable)
                    y_sim = features[:, idx]

                    y_sim = y_sim[np.searchsorted(times, obs["time"])]
                    cost += np.square((obs['mean']-y_sim)/obs['sem']).sum()

                    if experiment == "mask application":
                        all_costs[observable] = np.square((obs['mean']-y_sim)/obs['sem']).sum()

                    if print_costs:
                        c = np.square((obs['mean']-y_sim)/obs['sem']).sum()
                        limit = chi2.ppf(1-0.05, len([s for s in obs['sem'] if s != float('inf')]))
                        print(f"{experiment}-{observable}: cost {c}, limit {limit}, pass {c < limit}")
    except Exception as e:
        if "CVODE" not in str(e):
            raise(e)
        else:
            cost += 1e20
    return cost, all_costs


def f_cost_log(θ, sims, data, limit):
    cost, all_costs = f_cost(np.exp(θ), sims, data)

    for measurable, d in data["mask application"].items():
        if measurable not in ["input", "meta", "extra", "all_times"]:
            measurable_limit = chi2.ppf(1-0.05, len([s for s in d['sem'] if s != float('inf')]))
            if all_costs[measurable] > measurable_limit:
                cost += limit * (1 + (all_costs[measurable] - measurable_limit))
    return cost


def f_cost_parameter_identifiability_log(θ, sims, data, param_idx, direction, limit, value_accepted):

    v = direction*θ[param_idx]

    θ = np.exp(θ)

    cost, all_costs = f_cost(θ, sims, data)

    if cost > limit:
        v += (np.abs(v) + np.abs(value_accepted)) * (1 + (cost - limit))

    for measurable, d in data["mask application"].items():
        if measurable not in ["input", "meta", "extra", "all_times", "IL-25"]:
            measurable_limit = chi2.ppf(1-0.05, len([s for s in d['sem'] if s != float('inf')]))
            if all_costs[measurable] > measurable_limit:
                v += (np.abs(v) + np.abs(value_accepted)) * (1 + (all_costs[measurable] - measurable_limit))
    return v


def save_results(θ, model_name, cost, results_path="Results"):
    if not os.path.exists(f"{results_path}/{model_name}"):
        os.makedirs(f"{results_path}/{model_name}", exist_ok=True)
    with open(f"{results_path}/{model_name}/cost={cost}.json", "w") as f:
        json.dump({"cost": cost, "θopt": θ}, f, cls=NumpyPandasArrayEncoder)

def save_results_PI(θ, model_name, cost, param_name,results_path="Results_PI"):
    if not os.path.exists(f"{results_path}/{model_name}"):
        os.makedirs(f"{results_path}/{model_name}", exist_ok=True)
    formatted_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"{results_path}/{model_name}/{model_name}-{param_name}-({cost})-{formatted_time}.json", "w") as f:
        json.dump({"cost": cost, "θopt": θ}, f, cls=NumpyPandasArrayEncoder)


def optimize(model, data, θ0=None, fixed_parameters=[]): 

    sims, θ0_default, parameter_names = setup_simulations(model)

    if θ0 is None:
        θ0 = load_best_parameters(f"./Results/{model.name}", param_key="θopt", cost_key='cost', model=model) # The best parameters found so far with respect to the agreement to data

    bounds = get_parameter_bounds(parameter_names, θ0, fixed_parameters=fixed_parameters)

    limit = chi2.ppf(1-0.05, get_dof(data))

    θ0_log = np.log(θ0)

    # Add a minor perturbation to the initial guess in 5% of the cases
    if random.random() < 0.05:
        θ0_log += np.random.normal(0, 0.1, len(θ0_log))
        θ0_log = np.clip(θ0_log, bounds.lb, bounds.ub)

    res = differential_evolution(f_cost_log, x0=θ0_log, bounds=bounds, args=(sims, data, limit), disp=True)

    θopt = np.exp(res.x)

    # cost_opt,_ = f_cost(θopt, sims, data, print_costs=False)
    cost_opt = f_cost_log(res.x, sims, data, limit)
    save_results(θopt, model.name, cost_opt)

    return θopt, cost_opt


def find_least_tested_parameter(model_name, direction):
    files = os.listdir(f"./Results_PI/{model_name}")
    
    # Find all parameters that have been tested, but ignore those that have already been minimized to 1e-20
    tested = []
    for f in files: 
        if f.endswith(".json"):
            param_name = f.replace(f"{model_name}-",'').split("-")[0]
            if (not any(param_name in file and "e-20" in file for file in files) and direction == 1) or (not any(param_name in file and "e20" in file for file in files) and direction == -1):
                tested.append(param_name)
            else:
                if direction == 1:
                    print(f"Skipping {param_name} as it has already been found to be minimized (1e-20).")
                else:
                    print(f"Skipping {param_name} as it has already been found to be maximised (1e20).")

    min_count = min(map(tested.count, set(tested)))
    least_common_elements = [x for x in set(tested) if tested.count(x) == min_count]

    return random.choice(least_common_elements)


def optimize_parameter_identifiability(model, data, fixed_parameters=[], direction=1):

    sims, θ0_default, parameter_names = setup_simulations(model)

    param_to_optimize = find_least_tested_parameter(model.name, direction)
    param_idx = parameter_names.index(param_to_optimize)

    θ0 = load_best_parameters(f"./Results_PI/{model.name}", key=f'-{param_to_optimize}-', param_key="θopt",  cost_key='cost', model=model, direction=direction)

    accepted_value = θ0[param_idx]

    if fixed_parameters == "all" and "kd" not in param_to_optimize:
        fixed_parameters = [param_name for param_name in parameter_names if re.split('[abt]', param_name)[0] !=  re.split('[abt]', param_to_optimize)[0]]
    elif fixed_parameters == "all" and "kd" in param_to_optimize:
        fixed_parameters = []

    bounds = get_parameter_bounds(parameter_names, θ0, fixed_parameters=fixed_parameters)

    if direction == 1:
        bounds.lb[param_idx] = np.log(1e-20)
    elif direction == -1:
        bounds.ub[param_idx] = np.log(1e20)

    cost,_ = f_cost(θ0, sims, data)
    dof = get_dof(data)
    limit = chi2.ppf(1-0.05, dof)

    if cost > limit:
        raise ValueError("Initial parameters are not identifiable.")

    θ0_log = np.log(θ0)
    res = differential_evolution(f_cost_parameter_identifiability_log, x0=θ0_log, bounds=bounds, args=(sims, data, param_idx, direction, limit, accepted_value), disp=True)
    
    fvalue = np.exp(res.fun*direction) # return the sign to the correct value (min/max)
    θopt = np.exp(res.x)

    cost, _ = f_cost_log(np.log(θopt), sims, data, limit)

    if cost<limit:
        save_results_PI(θopt, f"{model.name}", fvalue, param_to_optimize, results_path="Results_PI")

    return θopt, res.fun