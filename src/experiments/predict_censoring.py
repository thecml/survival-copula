import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

from data_loader import SingleEventSyntheticDataLoader

# Configuration for the synthetic dataset
data_cfg = {
    "alpha_e1": 19,
    "alpha_e2": 17,
    "gamma_e1": 6,
    "gamma_e2": 4,
    "n_samples": 10000,
    "n_features": 10,
}

# Repeat experiment 100 times
auc_scores = []

for _ in range(100):
    # Simulate dataset
    dl = SingleEventSyntheticDataLoader().load_data(data_cfg, k_tau=0.2, copula_name="clayton", linear=False)
    df = dl.get_data()

    # Features and labels
    features = df.drop(columns=['time', 'event']).copy()
    features['observed_time'] = df['time']
    labels = df['event'].astype(int)

    # Predict using logistic regression
    model = LogisticRegression(max_iter=1000)
    probs = cross_val_predict(model, features, labels, method='predict_proba', cv=5)[:, 1]
    auc = roc_auc_score(labels, probs)
    auc_scores.append(auc)

# Compute average AUC
avg_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

# Report
result_df = pd.DataFrame({
    'Mean AUC': [avg_auc],
    'Std AUC': [std_auc],
    'Runs': [100]
})

print(result_df)