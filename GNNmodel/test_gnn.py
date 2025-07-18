# test_gcn.py

import numpy as np
import pandas as pd
import tensorflow as tf
from spektral.layers import GCNConv
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp


# Load custom GCN layer
custom_objects = {"GCNConv": GCNConv}


def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)).tocoo()


def evaluate(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


# Load data
data_df = pd.read_csv('data\TRsmiles_10000.csv')
labels_df = pd.read_csv('data\label_10000.csv')
similarity_df = pd.read_csv('data\Tanimoto_filtered.csv')

data_merged = pd.merge(data_df, labels_df, on="SMILES", how="inner")
features_all = data_merged.iloc[:, 1:-1].values
labels_all = data_merged.iloc[:, -1].values.reshape(-1, 1)

scaler = StandardScaler()
features_all = scaler.fit_transform(features_all)

num_nodes = len(data_merged)
smiles_to_index = {smiles: idx for idx, smiles in enumerate(data_merged["SMILES"])}

A = np.zeros((num_nodes, num_nodes))
for _, row in similarity_df.iterrows():
    i = smiles_to_index.get(row["SMILES1"])
    j = smiles_to_index.get(row["SMILES2"])
    if i is not None and j is not None:
        A[i, j] = A[j, i] = row["Similarity"]

A = A + np.eye(num_nodes)
A = normalize_adj(A).toarray()

# Define test indices
test_indices = np.arange(8000, 10000)
X_full = np.expand_dims(features_all, axis=0)
A_full = np.expand_dims(A, axis=0)
y_full = labels_all.flatten()

# Load model
model_path = "models/gcn_model_fold_7.h5"
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Predict
y_prob = model.predict([X_full, A_full], batch_size=1)[0, test_indices, 0]
y_pred = (y_prob > 0.5).astype(int)
y_true = y_full[test_indices]

# Evaluate
metrics = evaluate(y_true, y_pred, y_prob)
print("\n=== Test Set Evaluation (Fold 7 Model) ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")
