import numpy as np
import pandas as pd
import tensorflow as tf
from spektral.layers import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score,
    matthews_corrcoef, confusion_matrix
)
import scipy.sparse as sp

# Register GCN layer for loading
custom_objects = {"GCNConv": GCNConv}

# -----------------------------
# Utility functions
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

# -----------------------------
# Load test data (2000 samples)
# -----------------------------
test_df = pd.read_csv("data\test_2000.csv")          # 包含 SMILES + 1700维特征 + label
similarity_df = pd.read_csv("data\Tanimoto_filtered.csv")  # 包含所有样本的相似度

features = test_df.iloc[:, 1:-1].values
labels = test_df.iloc[:, -1].values.reshape(-1, 1)
smiles_list = test_df["SMILES"].tolist()
num_nodes = len(test_df)

# -----------------------------
# Normalize features
# -----------------------------
scaler = StandardScaler()
features = scaler.fit_transform(features)

# -----------------------------
# Build adjacency matrix from similarity file
# -----------------------------
smiles_to_index = {smiles: idx for idx, smiles in enumerate(smiles_list)}
A = np.zeros((num_nodes, num_nodes))
for _, row in similarity_df.iterrows():
    i = smiles_to_index.get(row["SMILES1"])
    j = smiles_to_index.get(row["SMILES2"])
    if i is not None and j is not None:
        A[i, j] = A[j, i] = row["Similarity"]

# Add self-loops and normalize
A = A + np.eye(num_nodes)
A = normalize_adj(A).toarray()

# -----------------------------
# Prepare input
# -----------------------------
X_input = np.expand_dims(features, axis=0)  # shape: (1, N, F)
A_input = np.expand_dims(A, axis=0)         # shape: (1, N, N)

# -----------------------------
# Load the pretrained model (fold 7)
# -----------------------------
model_path = "models\gcn_model_fold_7.h5"
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
print(f"✅ Loaded pretrained model: {model_path}")

# -----------------------------
# Make predictions
# -----------------------------
y_prob = model.predict([X_input, A_input], batch_size=1)[0, :, 0]
y_pred = (y_prob > 0.5).astype(int)
y_true = labels.flatten()

# -----------------------------
# Evaluation
# -----------------------------
metrics = evaluate(y_true, y_pred, y_prob)

print("\n=== Test Set Evaluation (2000 samples) ===")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

