import pandas as pd
import config as cfg

matplotlib_style = 'default'
import matplotlib.pyplot as plt; plt.style.use(matplotlib_style)
plt.rcParams.update({'axes.labelsize': 'medium',
                     'axes.titlesize': 'medium',
                     'font.size': 14.0})

# Load the results from CSV
results_df = pd.read_csv(f"{cfg.RESULTS_DIR}/weibull_model_error.csv")

# Extract data for plotting
ci_errors = results_df.pivot(index="top_k", columns="model_name", values="ci_error")
ibs_errors = results_df.pivot(index="top_k", columns="model_name", values="ibs_error")
mae_errors = results_df.pivot(index="top_k", columns="model_name", values="mae_error")

# Define plot settings
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
metrics = [("CI Error", ci_errors), ("IBS Error", ibs_errors), ("MAE Error", mae_errors)]

# Plot each metric
for i, (ax, (title, metric_data)) in enumerate(zip(axes, metrics)):
    for model_name in metric_data.columns:
        ax.plot(metric_data.index, metric_data[model_name], label=model_name, marker="o")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Top-k Features", fontsize=12)
    ax.set_ylabel("Error", fontsize=12)
    ax.legend(title="Model Name")
    ax.grid(True)
    if i == 0:  # Add legend only to the first subplot
        ax.legend(title="Model Name")
    else:
        ax.legend().set_visible(False)

plt.tight_layout()
plt.savefig(f"{cfg.PLOTS_DIR}/weibull_error.pdf", format='pdf', bbox_inches='tight')
plt.show()