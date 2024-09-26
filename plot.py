from matplotlib import pyplot as plt
import numpy as np
from estimators import KaplanMeier

def compare_km_curves(df1, df2, intervals=None, save_fig=None, show=False):
    """
    Plots the KM curves of the original and modified dataframe
    :param df1: original dataframe
    :param df2: combined truth and censored data
    :return: None
    """
    #results = logrank_test(df1.time.values, df2.time.values, df1.event.values, df2.event.values)

    event_times_1 = df1.time.values[df1.event.values == 1]
    censor_times_1 = df1.time.values[df1.event.values == 0]
    event_times_2 = df2.time.values[df2.event.values == 1]
    censor_times_2 = df2.time.values[df2.event.values == 0]
    if intervals is None:
        intervals = 21  # 20 bins
    bins = np.linspace(0, round(df1.time.max()), intervals)

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    ax0.hist([event_times_1, censor_times_1], bins=bins, histtype='bar', stacked=True)
    ax0.legend(['Event times', 'Censor Times'])
    ax0.set_title("Event/Censor Time Histogram")

    km_estimator = KaplanMeier(df1.time.values, df1.event.values)
    ax1.plot(km_estimator.survival_times, km_estimator.survival_probabilities, linewidth=3)
    ax1.set_title("Kaplan-Meier Curve")
    ax1.set_ylim([0, 1])
    xmin, xmax = ax1.get_xlim()

    ax2.hist([event_times_2, censor_times_2], bins=bins, histtype='bar', stacked=True)
    ax2.legend(['Event times', 'Censor Times'])
    # ax2.set_title("Event/Censor Times Histogram")

    km_estimator = KaplanMeier(df2.time.values, df2.event.values)
    ax3.plot(km_estimator.survival_times, km_estimator.survival_probabilities, linewidth=3)
    # ax3.set_title("Kaplan-Meier Curve")
    ax3.set_ylim([0, 1])
    ax3.set_xlim([xmin, xmax])

    # fig.set_size_inches(12, 12)
    #plt.suptitle('Logrank Test: p-value = {:.5f}'.format(results.p_value))
    plt.setp(ax0, xlabel='Time', ylabel='Counts')
    plt.setp(ax1, xlabel='Time', ylabel='Probabilities')
    plt.setp(ax2, xlabel='Time', ylabel='Counts')
    plt.setp(ax3, xlabel='Time', ylabel='Probabilities')
    # plt.show()
    if save_fig is not None:
        fig.savefig(save_fig, dpi=300)
        
    if show:
        plt.show()
        