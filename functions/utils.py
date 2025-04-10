import json
import os

from json import JSONEncoder
import numpy as np
import pandas as pd

from scipy.stats import qmc, chi2

import sund

class NumpyPandasArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.values

        return JSONEncoder.default(self, obj)

def load_and_process_data(file_path, print_DOF=0):
    with open(file_path, 'r') as f:
        data = json.load(f)

    for k, d in data.items():
        d.pop("meta", None)
        times = [t for k, var in d.items() if k not in ["input", "meta", "extra"] for t in var["time"]]+[-45, -29, 0, 1] # DEBUGGING 
        times = times+np.arange(times[0], times[-1]+1, 1).tolist()
        d["all_times"] = np.unique(np.array(times))
        for observable, obs in d.items():
            if observable not in ["input", "meta", "extra", "all_times"]:
                obs["mean"] = np.array(obs["mean"])
                obs["sem"] = np.array(obs["sem"])
    
    if print_DOF:
        dof = get_dof(data)
        print(f'DOF = {dof}\nchi2-cutoff = {chi2.ppf(1-0.05, dof)}')
    
    return data


def get_dof(data, print_dof=False):
    dof = 0
    for _, d in data.items():
        for observable, obs in d.items():
            if observable not in ["input", "meta", "extra", "all_times"]:
                dof += len([elem for elem in obs["sem"] if float(elem) != float('inf')])

    if print_dof:
        print(f'DOF = {dof}\nchi2-cutoff = {chi2.ppf(1-0.05, dof)}')
        
    return dof

def setup_simulations(model): 
    
    act = sund.Activity(time_unit='m')
    act.add_output(name='mask_application', type=sund.PIECEWISE_CONSTANT, tvalues=[-30, 0], fvalues=[0, 1, 0])

    sims = {}
    sims["steady"] = sund.Simulation(model, time_unit='m')
    sims["mask application"] = sund.Simulation(model, act, time_unit='m')

    return sims, model.parametervalues, model.parameternames


def load_best_parameters(folder, key=None, param_key='x', cost_key='cost', print_names=False, model=None, direction=1):
    """Load the best parameters from a folder with json files. The best parameters are the ones with the lowest cost.
    The function assumes that the json files have the following structure:
    {
        "cost": [cost of the parameters]
        "θopt": [list of parameters],
    }

    Args:
        folder (string): The folder to search for the results json files.
        key (str, optional): A key that the files must contain to be considered. Defaults to None.
        param_key (str, optional): The key in the json file for parameter values. Defaults to 'x'.
        cost_key (str, optional): The key in the json file for the cost. Defaults to 'cost'.
        print_names (bool, optional): Print the parameter names if True.
        model (sund.model, optional): Model object to load default values from.
        direction (int, optional): Defines if the lowest or greatest solution should be loaded. 

    Returns:
        list: The best parameter values.
    """

    if not os.path.exists(folder) or len(os.listdir(folder))==0:
        print("No best parameters found.")
        if model is None:
            return []
        else:
            print(f"Using default values from model {model.name}")
            return model.parametervalues
        
    files = os.listdir(folder)
    
    if key is not None:
        files = [f for f in files if key in f] 

    results = []
    for file in files:
        if file.endswith(".json"):
            try:
                with open(f"{folder}/{file}", 'r') as f:
                    results.append(json.load(f))
            except json.JSONDecodeError:
                print(f"Could not load {folder}/{file}")
            except FileNotFoundError:
                print(f"The file '{folder}/{file}' does not exist. If it is a temporary file, it might have been removed and this error can be ignored.")
            except PermissionError:
                print(f"Permission denied for file '{folder}/{file}'. If it is a temporary file, it might have been removed and this error can be ignored.")

    # find the best results loaded from files
    if direction == 1:
        best_result = min(results, key=lambda x: x[cost_key])
    else:
        best_result = max(results, key=lambda x: x[cost_key])
    
    if print_names:
        print(best_result)
        
    return best_result[param_key]
