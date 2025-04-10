# %% import modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import os 
import json

from optimization import NumpyPandasArrayEncoder

#%% 
def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

# %% Define functions to help plot the raw data 
def plot_data(data, folder="Plots (data only)", show=False):

    if folder[-1] != "/":
        folder += "/"

    n_patients = max(data['Patient'].unique())

    for col in data.columns:
        if col in ['Patient', 'Sample', 'Time']:
            continue
        fig = plt.figure(1)

        print(col)
        axs = []
        for i in range(1,n_patients+1):
            axs.append(fig.add_subplot(4, 3, i))
            mask_control = (data['Patient'] == i) & (data["Sample"] == "control")
            axs[i-1].plot(data.loc[mask_control, "Time"], data.loc[mask_control, col])
            mask_normal = (data['Patient'] == i) & (data["Sample"] == "mask application")
            axs[i-1].plot(data.loc[mask_normal, "Time"], data.loc[mask_normal, col])
            axs[i-1].set_title(f'Patient {i}')
            axs[i-1].set_xlabel('Time [min]')
            axs[i-1].set_ylabel(col)
        fig.tight_layout()
        fig.savefig(f'{folder}{col.replace("/","+")}.png', dpi=300)
        plt.close(fig)
    if show:
        plt.show()

def plot_data_same(data, folder="Plots", show=False):

    if folder[-1] != "/":
        folder += "/"

    for col in data.columns:
        if col in ['NIV', 'Patient', 'Sample', 'Time']:
            continue
        fig = plt.figure()

        print(col)
        sns.lineplot(data=data, x="Time", y=col, hue="NIV", style="Sample", errorbar=None)
        fig.tight_layout()
        fig.savefig(f'{folder}{col.replace("/","+")}.png', dpi=300)
        plt.close(fig)
    if show:
        plt.show()
# %% read data
data = pd.read_excel('Data/data.xlsx', sheet_name='measurements')

# %% Reformat the data to json/dict structure: 
# (might not be useful, since the time with treatment is different)
data_dict = {}
data_copy = data.copy()
data_copy.loc[data_copy["Time"]<0,"Time"] = -30
for sample in data_copy["Sample"].unique():
    data_dict[sample] = {}
    for col in data_copy.loc[:,"CTACK":].columns:
        data_dict[sample][col] = {}
        data_dict[sample][col]["time"] = data_copy.loc[(data_copy["Patient"]==1) & (data_copy["Sample"]==sample), "Time"]

        measures = []

        values = data_copy.loc[(data_copy["Patient"]==1) &( data_copy["Sample"]==sample), "Sample":"Time"]
        for pat in data_copy["Patient"].unique():
            if pat == 11:
                continue
            patient_values = data_copy.loc[(data_copy["Patient"]==pat) & (data_copy["Sample"]==sample), [c for c in data_copy.columns.to_list() if c in ["Sample", "Time", col]]]

            values = values.merge(patient_values, on=["Sample","Time"], how="left")
            measures.append(values.pop(col).values)
        measures = np.array(measures)
        data_dict[sample][col]["measures"] = measures
        data_dict[sample][col]["mean"] = np.nanmean(measures, axis=0)
        data_dict[sample][col]["std"] = np.nanstd(measures, axis=0)
        data_dict[sample][col]["sem"] = np.nanstd(measures, axis=0)/np.sqrt(np.array(sum(~np.isnan(measures)))-1)

# %% save data
with open("Data/data-latest.json",'w') as f:
    json.dump(data_dict, f, cls=NumpyPandasArrayEncoder)

# %% Test if any of the point is significantly different from the first point

# significant_variables = []
# alpha = 0.05

# data_N = data_dict["mask application"]

# for col in data_N.keys():
   
#     alpha_bonferroni = alpha/sum(~np.isnan(data_N[col]["sem"]))# Divide alpha with the number of non-nan SEM values
#     for idx in range(1,len(data_N[col]["measures"])):
#         t_stat, p_val = stats.ttest_ind(data_N[col]["measures"][idx], data_N[col]["measures"][0])
#         if p_val < alpha_bonferroni:
#             significant_variables.append(col)
#             print(f" {col} {idx} is significantly different from the first measurement, p={p_val:.3f}")

# %% Construct a model from the significant variables using omnipath

# %% Plot the significant and non signficant variables and save to different folders
# make_dir_if_not_exists("Plots (data only)/mean/significant/")
# make_dir_if_not_exists("Plots (data only)/mean/nonsignificant/")

# plt.figure()
# for col in data_N.keys():
#     plt.clf()
#     plt.title(col)
#     for idx in range(len(data_N[col]["measures"])):
#         plt.plot(data_N[col]["time"], data_N[col]["measures"][idx], alpha=0.2)
#     plt.errorbar(data_N[col]["time"], data_N[col]["mean"], yerr=data_N[col]["sem"], label=col)
#     if col in significant_variables:
#         plt.savefig(f'Plots (data only)/mean/significant/{col.replace("/","+")}.png', dpi=300)
#     else: 
#         plt.savefig(f'Plots (data only)/mean/nonsignificant/{col.replace("/","+")}.png', dpi=300)

# %% Plot the mean data
# make_dir_if_not_exists("Plots (data only)/mean")
# data_N = data_dict["mask application"]
# plt.figure()
# for col in data_N.keys():
#     plt.clf()
#     plt.title(col)
#     for idx in range(len(data_N[col]["measures"])):
#         plt.plot(data_N[col]["time"], data_N[col]["measures"][idx], alpha=0.2)
#     plt.errorbar(data_N[col]["time"], data_N[col]["mean"], yerr=data_N[col]["sem"], label=col)
#     plt.savefig(f'Plots (data only)/mean/{col.replace("/","+")}.png', dpi=300)


# %% Plot all data in same plot
plot_data_same(data, folder="Plots (data only)/all_same/")


#%% 
## OLD plots no longer used

## %% Plot all data
# plot_data(data, folder="Plots (data only)/all_split/")

## Normalized data
# # %% Normalize data to the first measurement of the normal measurement, for each patient
# data_normalized = data.copy()
# for col in data_normalized.columns: # for each column
#     if col in ['Patient', 'Sample', 'Time', 'NIV']:
#         continue
#     for i in range(1, max(data_normalized['Patient'].unique())+1): # for each patient
#         mask_normal = (data_normalized['Patient'] == i) & (data["Sample"] == "mask application")
#         mask_control = (data_normalized['Patient'] == i) & (data["Sample"] == "control")
#         data_normalized.loc[mask_control, col] = data_normalized.loc[mask_control, col] / data_normalized.loc[mask_normal, col].iloc[0]
#         data_normalized.loc[mask_normal, col] = data_normalized.loc[mask_normal, col] / data_normalized.loc[mask_normal, col].iloc[0]

# # %% Plot normalized data
# plot_data_same(data_normalized, folder="Plots (data only)/all_same_normalized/")

# # %% Remove 0 values, plot again
# data_measurements = data.loc[:,"CTACK":]
# data_measurements = data_measurements.replace(0.0, np.nan)
# data_trimmed = data.copy()
# data_trimmed.loc[:,"CTACK":] = data_measurements

# # %% plot filtered data
# plot_data(data, folder="Plots (data only)/pruned_split/")

##%% Test the two groups of erythema
# data_erythema = {}
# data_copy = data.copy()
# data_copy.loc[data_copy["Time"]<0,"Time"] = -30
# for erythema in data_copy["Sign_of_erythema_0_minutes"].unique():
#     data_erythema[erythema] = {}
#     for col in data_copy.loc[:,"CTACK":].columns:
#         data_erythema[erythema][col] = {}
#         data_erythema[erythema][col]["time"] = data_copy.loc[(data_copy["Patient"]==1) & (data_copy["Sample"]=="mask application"), "Time"]

#         measures = []
#         values = data_copy.loc[(data_copy["Patient"]==1) & (data_copy["Sample"]=="mask application"), "Sample":"Time"]
#         patients_with_erythema = data_copy.loc[data_copy["Sign_of_erythema_0_minutes"]==erythema, "Patient"].unique()
#         for pat in patients_with_erythema:
#             patient_values = data_copy.loc[(data_copy["Patient"]==pat) & (data_copy["Sign_of_erythema_0_minutes"]==erythema) & (data_copy["Sample"]=="mask application"), [c for c in data_copy.columns.to_list() if c in ["Time", col]]]

#             values = values.merge(patient_values, on=["Time"], how="left")
#             measures.append(values.pop(col).values)##%% Test the two groups of erythema
# data_erythema = {}
# data_copy = data.copy()
# data_copy.loc[data_copy["Time"]<0,"Time"] = -30
# for erythema in data_copy["Sign_of_erythema_0_minutes"].unique():
#     data_erythema[erythema] = {}
#     for col in data_copy.loc[:,"CTACK":].columns:
#         data_erythema[erythema][col] = {}
#         data_erythema[erythema][col]["time"] = data_copy.loc[(data_copy["Patient"]==1) & (data_copy["Sample"]=="mask application"), "Time"]

#         measures = []
#         values = data_copy.loc[(data_copy["Patient"]==1) & (data_copy["Sample"]=="mask application"), "Sample":"Time"]
#         patients_with_erythema = data_copy.loc[data_copy["Sign_of_erythema_0_minutes"]==erythema, "Patient"].unique()
#         for pat in patients_with_erythema:
#             patient_values = data_copy.loc[(data_copy["Patient"]==pat) & (data_copy["Sign_of_erythema_0_minutes"]==erythema) & (data_copy["Sample"]=="mask application"), [c for c in data_copy.columns.to_list() if c in ["Time", col]]]

#             values = values.merge(patient_values, on=["Time"], how="left")
#             measures.append(values.pop(col).values)
#         measures = np.array(measures)
#         data_erythema[erythema][col]["measures"] = measures
#         data_erythema[erythema][col]["mean"] = np.nanmean(measures, axis=0)
#         data_erythema[erythema][col]["std"] = np.nanstd(measures, axis=0)
#         data_erythema[erythema][col]["sem"] = np.nanstd(measures, axis=0)/np.sqrt(np.array(sum(~np.isnan(measures)))-1)

# plt.figure()
# for sample in data_N.keys():
#     plt.clf()
#     plt.title(sample)
#     # plot averages
#     plt.errorbar(data_N[sample]["time"], data_N[sample]["mean"], yerr=data_N[sample]["sem"], label=sample, color = 'k', capsize=5)
#     plt.errorbar(data_erythema["Sign of erythema"][sample]["time"]-1, data_erythema["Sign of erythema"][sample]["mean"], yerr=data_dict["Sign of erythema"][sample]["sem"], label=sample, color = 'r', capsize=5)
#     plt.errorbar(data_erythema["No sign of erythema"][sample]["time"]+1, data_erythema["No sign of erythema"][sample]["mean"], yerr=data_dict["No sign of erythema"][sample]["sem"], label=sample, color = 'b', capsize=5)
#     plt.legend(["All", "Erythema", "No erythema"])
#     # plot individual measurements
#     for idx in range(len(data_erythema["Sign of erythema"][sample]["measures"])):
#         plt.plot(data_erythema["Sign of erythema"][sample]["time"]-1, data_erythema["Sign of erythema"][sample]["measures"][idx], alpha=0.2, color='r')
#     for idx in range(len(data_erythema["No sign of erythema"][sample]["measures"])):
#         plt.plot(data_erythema["No sign of erythema"][sample]["time"]+1, data_erythema["No sign of erythema"][sample]["measures"][idx], alpha=0.2, color='b')
#     plt.savefig(f'Plots (data only)/mean_erythema_grouped/{sample.replace("/","+")}.png', dpi=300)

#         measures = np.array(measures)
#         data_erythema[erythema][col]["measures"] = measures
#         data_erythema[erythema][col]["mean"] = np.nanmean(measures, axis=0)
#         data_erythema[erythema][col]["std"] = np.nanstd(measures, axis=0)
#         data_erythema[erythema][col]["sem"] = np.nanstd(measures, axis=0)/np.sqrt(np.array(sum(~np.isnan(measures)))-1)


# data_N = data_dict["mask application"]
# plt.figure()
# for sample in data_N.keys():
#     plt.clf()
#     plt.title(sample)
#     for idx in range(len(data_N[sample]["measures"])):
#         plt.plot(data_N[sample]["time"], data_N[sample]["measures"][idx], alpha=0.2)
#     plt.errorbar(data_N[sample]["time"], data_N[sample]["mean"], yerr=data_N[sample]["sem"], label=sample)
#     plt.savefig(f'Plots (data only)/mean/{sample.replace("/","+")}.png', dpi=300)

# plt.figure()
# for sample in data_N.keys():
#     plt.clf()
#     plt.title(sample)
#     # plot averages
#     plt.errorbar(data_N[sample]["time"], data_N[sample]["mean"], yerr=data_N[sample]["sem"], label=sample, color = 'k', capsize=5)
#     plt.errorbar(data_erythema["Sign of erythema"][sample]["time"]-1, data_erythema["Sign of erythema"][sample]["mean"], yerr=data_dict["Sign of erythema"][sample]["sem"], label=sample, color = 'r', capsize=5)
#     plt.errorbar(data_erythema["No sign of erythema"][sample]["time"]+1, data_erythema["No sign of erythema"][sample]["mean"], yerr=data_dict["No sign of erythema"][sample]["sem"], label=sample, color = 'b', capsize=5)
#     plt.legend(["All", "Erythema", "No erythema"])
#     # plot individual measurements
#     for idx in range(len(data_erythema["Sign of erythema"][sample]["measures"])):
#         plt.plot(data_erythema["Sign of erythema"][sample]["time"]-1, data_erythema["Sign of erythema"][sample]["measures"][idx], alpha=0.2, color='r')
#     for idx in range(len(data_erythema["No sign of erythema"][sample]["measures"])):
#         plt.plot(data_erythema["No sign of erythema"][sample]["time"]+1, data_erythema["No sign of erythema"][sample]["measures"][idx], alpha=0.2, color='b')
#     plt.savefig(f'Plots (data only)/mean_erythema_grouped/{sample.replace("/","+")}.png', dpi=300)



#%% Plot t=120 vs t=180
plt.close('all')
plt.figure()
mean_at_120 = []
mean_at_180 = []
data = data_dict["mask application"]
for variable in data: 
    if 120 in data[variable]["time"].values and 180 in data[variable]["time"].values:
        mean_at_120.append(data[variable]["mean"][data[variable]["time"]==120])
        mean_at_180.append(data[variable]["mean"][data[variable]["time"]==180])

plt.scatter(mean_at_120, mean_at_180)
plt.plot([0, 5000],[0, 5000])
plt.xlabel("Mean at t=120")
plt.ylabel("Mean at t=180")

plt.figure()
mean_at_120 = []
mean_at_180 = []
data = data_dict["mask application"]
for variable in data: 
    if 120 in data[variable]["time"].values and 180 in data[variable]["time"].values:
        mean_at_120.append(data[variable]["mean"][data[variable]["time"]==120])
        mean_at_180.append(data[variable]["mean"][data[variable]["time"]==180])

plt.scatter(np.log10(mean_at_120), np.log10(mean_at_180))
plt.plot([-2, np.log10(5000)],[-2, np.log10(5000)])
plt.xlabel("log mean at t=120")
plt.ylabel("log mean at t=180")
plt.show()


# %%
