#%% Plot the KM curves and the histogram for all the datasets
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

from data import make_survival_data
from SurvivalEVAL.Evaluations.util import KaplanMeier

sns.set(style="whitegrid")
colors = sns.color_palette("deep")

datasets = [
    "VALCT", "DLBCL", "PBC", "GBM", "NACD", "GBSG", "METABRIC", "SUPPORT",
    "AIDS", "HFCR", "WPBC", "BMT", "churn", "credit", "employee", "PDM",
    "MIMIC-IV_hosp", "MIMIC-IV_all",
    "DBCD", "FLCHAIN", "NWTCO", "NPC", "WHAS", "WHAS500",
    "SEER_liver", "SEER_lung", "SEER_prostate", "SEER_brain", "SEER_thyroid",
    "SEER_stomach", "SEER_urinary", "SEER_kidney", "SEER_breast",
]
for data_name in datasets:
    data, _ = make_survival_data(data_name)
    data = data.astype({'time': 'float64', 'event': 'int32'})
    censor_rate = 1 - data.event.mean()

    event_times = data.time.values[data.event.values == 1]
    censor_times = data.time.values[data.event.values == 0]

    # Sturges formula
    intervals = math.ceil(math.log2(data.shape[0]) + 1)
    bins = np.linspace(0, round(data.time.max()), intervals)

    fig, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))

    km_estimator = KaplanMeier(data.time.values, data.event.values)
    survival_times = km_estimator.survival_times
    survival_probabilities = km_estimator.survival_probabilities
    print(f"{data_name}; last time: {survival_times[-1]}; last survival probability: {survival_probabilities[-1]}")
    if survival_times[0] != 0:
        survival_times = np.insert(survival_times, 0, 0)
        survival_probabilities = np.insert(survival_probabilities, 0, 1.0)
    ax0.step(survival_times, survival_probabilities, linewidth=2.5, color=colors[0],
             clip_on=False, zorder=3)
    # ax0.set_title("Kaplan-Meier Curve")
    ax0.set_ylabel("Survival Probability", color=colors[0], weight='bold')
    ax0.set_xlabel("Time", weight='bold')
    ax0.set_ylim([0, 1.05])
    ax0.tick_params(axis='y', colors=colors[0])
    ax0.set_xlim([0, max(survival_times)])
    ax0.xaxis.grid(False)
    # ax0.set_xticks([])
    xmin, xmax = ax0.get_xlim()

    ax1 = ax0.twinx()
    ax1.hist([event_times, censor_times], bins=bins, histtype='barstacked', stacked=True, alpha=0.9, color=[colors[2], colors[1]], zorder=2)
    # ax1.set_yscale('log')
    ax1.set_ylabel('Counts', color='black', weight='bold')
    ax1.legend(['Event', 'Censored'], loc='best')
    ax1.yaxis.grid(False)
    ax0.set_zorder(ax1.get_zorder() + 1)
    ax0.patch.set_visible(False)

    # ax1.set_title("Event/Censor Time Histogram")

    # fig.set_size_inches(12, 12)
    # plt.suptitle(
    #     '{}\n #Subjects: {}; %Censoring: {:.1f}%'.format(data_rename[data_name], data.shape[0], round(censor_rate * 100, 3))
    # )
    # plt.suptitle(
    #     '{}'.format(data_rename[data_name])
    # )
    # plt.show()
    plt.tight_layout()
    fig.savefig(f'figs/data/{data_name}.png', dpi=300)
    plt.close(fig)