"""
Predictive maintenance tiny demo:
- Generates synthetic features (age, cycles, temp)
- Creates failure labels with a logistic rule + noise
- Trains LogisticRegression
- Prints ROC-AUC and a simple maintenance decision from probability threshold
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def make_synth(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.uniform(0, 10, n)          # years
    cycles = rng.uniform(100, 2000, n)   # cycle count
    temp = rng.uniform(20, 100, n)       # operating temp
    X = np.column_stack([age, cycles, temp])

    # logistic ground-truth with noise
    w = np.array([0.6, 0.002, 0.03])
    logit = X @ w - 3.5
    p = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, p)
    return X, y

def main():
    X, y = make_synth()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, proba)

    print("✅ predictive-maintenance-planning demo OK")
    print(f"ROC-AUC: {auc:.3f}")

    # Simple maintenance rule: if failure prob >= 0.5, schedule maintenance
    threshold = 0.5
    to_maintain = (proba >= threshold).sum()
    print(f"Units flagged for maintenance at threshold {threshold}: {to_maintain}/{len(proba)}")

if __name__ == "__main__":
    main()
