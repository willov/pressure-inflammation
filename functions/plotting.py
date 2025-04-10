import json
import os
import numbers
import copy

import numpy as np

import matplotlib.pyplot as plt
from scipy.stats import qmc, chi2

from functions.utils import load_best_parameters

fast_cytokines = [
    "IL-1α",
    "IL-1β",
    "CHI3L1",
    "CSF1",
    "VEGF-A",
    "CX3CL1",
    "IL-1RA",
    "IL-2Ra",
    "IL-9",
]
slow_cytokines = ["IFN-α2a", "IL-25", "CSF3", "IL-7", "IL-27", "IL-23", "IL-15", "CCL3"]


def close_to_square(n):
    b = np.round(np.sqrt(n))
    a = np.ceil(n / b)
    return int(a), int(b)


def plot_agreement(θ, sims, data, model_name, figure_name="agreement"):
    # Set default font sizes
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )

    n_fast_plots = len(fast_cytokines)
    n_slow_plots = len(slow_cytokines)
    m_fast, n_fast = close_to_square(n_fast_plots)
    m_slow, n_slow = close_to_square(n_slow_plots)

    # Check if we are simulating a single parameter set or multiple. 
    # If multiple, we assume that the first is the optimal/best parmamter set
    if isinstance(θ[0], numbers.Number):
        θ = [θ]

    for experiment, d in data.items():
        times = np.linspace(
            data[experiment]["all_times"][0], data[experiment]["all_times"][-1], 10000
        )

        # Simulate the best parameter set

        θ_best = θ[0]

        sims["steady"].simulate(
            t=np.linspace(0, 24 * 60, 1000), theta=θ_best, reset=True
        )
        sims[experiment].simulate(
            t=times, x0=sims["steady"].state_values.copy(), theta=θ_best
        )

        best_sim = copy.copy(sims[experiment])

        if len(θ) > 1:
            obs_max = {
                obs: sims[experiment].feature_data[:, sims[experiment].feature_names.index(obs)]
                for obs in d.keys() if obs not in ["input", "meta", "extra", "all_times"]
            }
            obs_min = obs_max.copy()

            for it, θ_set in enumerate(θ[::-1]):
                sims["steady"].simulate(
                    t=np.linspace(0, 24 * 60, 1000), theta=θ_set, reset=True
                )

                sims[experiment].simulate(
                    t=times, x0=sims["steady"].state_values.copy(), theta=θ_set
                )

                for observable, obs in d.items():
                    if observable not in ["input", "meta", "extra", "all_times"]:
                        idx = sims[experiment].feature_names.index(observable)
                        y_sim = sims[experiment].feature_data[:, idx]
                        obs_max[observable] = np.maximum(obs_max[observable], y_sim)
                        obs_min[observable] = np.minimum(obs_min[observable], y_sim)
        else: 
            obs_max = None
            obs_min = None

        idx_fast = 0
        idx_slow = 0
        feature_names = best_sim.feature_names
        for observable, obs in d.items():
            if observable not in ["input", "meta", "extra", "all_times"]:
                idx = feature_names.index(observable)
                y_sim = best_sim.feature_data[:, idx]

                if observable in fast_cytokines:
                    marker_color = "#ff0000"
                    plt.figure("fast")
                    idx_fast += 1
                    ax = plt.subplot(m_fast, n_fast, idx_fast)
                elif observable in slow_cytokines:
                    marker_color = "#64914b"
                    plt.figure("slow")
                    idx_slow += 1
                    ax = plt.subplot(m_slow, n_slow, idx_slow)
                else:
                    raise ValueError(
                        f"Observable {observable} not found in fast or slow cytokines"
                    )

                plt.plot(
                    sims[experiment].time_vector,
                    y_sim,
                    label="simulation",
                    color=marker_color,
                    linewidth=2.0,
                )

                plt.errorbar(
                    obs["time"],
                    obs["mean"],
                    yerr=obs["sem"],
                    fmt="o",
                    capsize=5,
                    label="data",
                    ecolor=marker_color,
                    markerfacecolor=marker_color,
                    markeredgecolor=marker_color,
                    linewidth=1.5,
                    markersize=6,
                )

                if obs_max is not None and obs_min is not None:
                    plt.fill_between(
                        sims[experiment].time_vector,
                        obs_min[observable],
                        obs_max[observable],
                        color=marker_color,
                        alpha=0.2,
                        label="uncertainty",
                    )

                plt.ylabel(observable, fontsize=14)
                plt.xlabel("Time after mask removal (min)", fontsize=14)
                plt.axvspan(
                    -30,
                    0,
                    color="darkgrey",
                    alpha=0.3,
                    label="mask application",
                )
                plt.ylim(bottom=0)
                if observable == "IL-1RA":
                    plt.ylim(top=5200)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(
                    axis="both", which="major", labelsize=12, width=1.5
                )


    # Save the plot
    os.makedirs(f"Figures/{model_name}", exist_ok=True)

    plt.figure("fast")
    plt.gcf().set_size_inches(15, 10)
    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    plt.subplot(m_fast, n_fast, n_fast_plots)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend()
    plt.savefig(f"Figures/{model_name}/{figure_name} - fast.png", dpi=300)
    plt.savefig(f"Figures/{model_name}/{figure_name} - fast.svg")

    plt.figure("slow")
    plt.gcf().set_size_inches(15, 10)
    plt.tight_layout(h_pad=2.0, w_pad=2.0)
    plt.subplot(m_slow, n_slow, n_slow_plots)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(f"Figures/{model_name}/{figure_name} - slow.png", dpi=300)
    plt.savefig(f"Figures/{model_name}/{figure_name} - slow.svg")

    # Show the plot
    plt.show()


def print_parameter_table(model):
    θ_best = load_best_parameters(
        f"./Results/{model.name}", param_key="θopt", cost_key="cost"
    )
    param_names = model.parameternames

    with open(f"Figures/{model.name}/PI_table.md", "w") as f:
        print("| Parameter | Value | Range |")
        print("|---|---|---|")

        f.write("| Parameter | Value | Range |\n")
        f.write("|---|---|---|\n")

        for p_idx, p_name in enumerate(param_names):
            θmin = load_best_parameters(
                f"./Results_PI/{model.name}",
                key=p_name,
                param_key="θopt",
                cost_key="cost",
                direction=1,
            )
            θmax = load_best_parameters(
                f"./Results_PI/{model.name}",
                key=p_name,
                param_key="θopt",
                cost_key="cost",
                direction=-1,
            )
            print(
                f"|{p_name} | {θ_best[p_idx]:.2e} | [{θmin[p_idx]:.2e}, {θmax[p_idx]:.2e}] |"
            )
            f.write(
                f"|{p_name} | {θ_best[p_idx]:.2e} | [{θmin[p_idx]:.2e}, {θmax[p_idx]:.2e}] |\n"
            )
