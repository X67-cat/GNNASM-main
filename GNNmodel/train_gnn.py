# train_gcn.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from spektral.layers import GCNConv
import scipy.sparse as sp
import os

# -----------------------------
# Utility Functions
# -----------------------------
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


def build_gcn(input_dim, num_nodes, output_dim=1):
    x_in = Input(shape=(num_nodes, input_dim))
    a_in = Input(shape=(num_nodes, num_nodes))

    x = GCNConv(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))([x_in, a_in])
    x = Dropout(0.7)(x)
    x = GCNConv(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))([x_in, a_in])
    x = Dropout(0.5)(x)
    dense_1 = Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    outputs = Dense(output_dim, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(dense_1)

    return Model(inputs=[x_in, a_in], outputs=outputs)


# -----------------------------
# Load Data
# -----------------------------
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

# -----------------------------
# Cross-validation
# -----------------------------
train_size = 8000
X_train = features_all[:train_size]
y_train = labels_all[:train_size].flatten()

X_full = np.expand_dims(features_all, axis=0)
A_full = np.expand_dims(A, axis=0)
y_full = np.expand_dims(labels_all, axis=0)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = []
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
seventh_fold_model_path = os.path.join(model_dir, "gcn_model_fold_7.h5")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-4),
]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n=== Fold {fold} ===")

    sample_weights = np.zeros_like(y_full, dtype=float)
    sample_weights[0, train_idx, 0] = 1.0
    val_weights = np.zeros_like(y_full, dtype=float)
    val_weights[0, val_idx, 0] = 1.0

    model = build_gcn(input_dim=X_train.shape[1], num_nodes=num_nodes)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(
        [X_full, A_full], y_full,
        sample_weight=sample_weights,
        validation_data=([X_full, A_full], y_full, val_weights),
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
        verbose=0
    )

    if fold == 7:
        model.save(seventh_fold_model_path)
        print(f"Saved 7th fold model to: {seventh_fold_model_path}")

    y_prob_val = model.predict([X_full, A_full], batch_size=1)[0, val_idx, 0]
    y_pred_val = (y_prob_val > 0.5).astype(int)
    fold_metrics = evaluate(y_train[val_idx], y_pred_val, y_prob_val)
    cv_results.append(fold_metrics)

    for k, v in fold_metrics.items():
        print(f"{k}: {v:.4f}")

# Print CV summary
print("\n=== Cross-validation summary ===")
for metric in cv_results[0].keys():
    vals = [r[metric] for r in cv_results if not np.isnan(r[metric])]
    print(f"{metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

