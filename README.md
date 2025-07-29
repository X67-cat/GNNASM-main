# AnticanNet
A graph neural network -based prediction model for anticancer small molecules 
# 1.Description
![flowchart](page_1.png)
This flowchart presents a complete pipeline for anticancer small molecule activity prediction, consisting of four core stages. First, relevant data are collected from the PubChem BioAssay and CTD databases to construct a dataset of small molecule samples for training and testing. Next, during the feature engineering stage, SMILES representations are processed in two ways: molecular descriptors and high-dimensional features are extracted using PaDEL and ChemGPT, while RDKit is used to generate ECFP fingerprints, and Tanimoto similarity is calculated to construct a weighted molecular graph, resulting in the feature matrix and adjacency matrix required for graph neural networks. Then, a two-layer Graph Convolutional Network (GCN) is applied to extract high-level structural features of the molecules, uncovering potential relationships between structure and activity. Finally, in the prediction phase, a fully connected neural network combined with a Sigmoid function is used for binary classification, where a predicted probability ≥ 0.5 indicates a positive (active) compound. This integrated approach leverages multi-source features and graph-based modeling to achieve efficient and accurate prediction of anticancer activity in small molecules.
# 2.Requirements
Python >= 3.10.15

pytorch = 2.0.0

pytorch-cuda = 11.8

numpy=1.24.4

pandas=1.5.3

tensorflow=2.11.0

scikit-learn=1.2.2

matplotlib=3.7.1

spektral=1.2.0

rdkit=2022.09.5

tqdm=4.65.0

openpyxl=3.1.2
# 3.How to run
## 3.1 Download Project Files
Open the GitHub repository page, click the Code button in the upper right corner, select Download ZIP to download the project compressed package
```
git clone https://github.com/X67-cat/GNNASM-main.git
```
Unzip the compressed package and ensure the local directory structure is complete including code files, images folder, datasets, etc.
## 3.2 Prepare Running Environment
1.Open the terminal (or VS Code terminal) and navigate to the project root directory
Install dependency packages (Python 3.7+ needs to be installed in advance)
```
pip install numpy pandas tensorflow scikit-learn spektral scipy
```
2.After the installation is complete, you can check if all libraries are successfully installed using the following command:
```
python -c "import numpy, pandas, tensorflow, sklearn, spektral, scipy; print('All dependencies are installed successfully!')"
```
If the terminal outputs "All dependencies are installed successfully!", it means the environment preparation is complete.
## 3.3 Train and Test Model (Based on Training Set)
Put your data into the ```GNNmodel\train_gnn.py```:
```
TRsmiles_feature.csv
TRsniles_label.csv
Tanimoto_filtered.csv
```
Code running description:
1.The code will first load and preprocess the data, including reading the above three data files, performing data merging, feature standardization, adjacency matrix construction and normalization, etc.
2.Build a GCN model, which includes two GCNConv layers, Dropout layers, Dense layers, etc., using the adam optimizer and binary_crossentropy as the loss function.
3.Use 10-fold cross-validation for training, and during the training process, the 7th fold model will be saved to 
```
models/gcn_model_fold_7.h5
```
4.After training, it will output the evaluation indicators (accuracy, sensitivity, specificity, auc, mcc) for each fold and the summary results of cross-validation (mean ± standard deviation).
## 3.4 Model Prediction (Based on Test Set) 
Put your data into the ```GNNmodel\test_gnn.py```:
```
TRsmiles_feature.csv
TRsniles_label.csv
Tanimoto_filtered.csv
```
1.The code will load the 7th fold model (models/gcn_model_fold_7.h5) saved during training, and load the above three data files and perform the same data preprocessing operations as during training.
2.Use the loaded model to predict the test set data, and obtain the prediction probability and prediction label.
3.Calculate and output the evaluation indicators (accuracy, sensitivity, specificity, auc, mcc) of the test set to evaluate the performance of the model on the test set.
# 4 Results 
In the experiments, the input files used for both the training and testing sets are the same: data/TRsmiles_feature.csv, data/TRsmiles_label.csv, and data/Tanimoto_filtered.csv. However, the code distinguishes between training and testing data by selecting the first 8,000 samples as the training set and the remaining 2,000 samples as the validation (test) set. After running the code, the results of 10-fold cross-validation on the training set and the evaluation on the test set will be displayed in the terminal for easy observation and selection of experimental results. It is recommended to back up important results in advance. If you wish to preserve historical outputs, you may modify the code accordingly.
