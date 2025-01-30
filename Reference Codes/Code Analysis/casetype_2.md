This code implements a comprehensive text classification pipeline using various models and techniques. Here's a breakdown of the main sections:

1. Library Imports: The code starts by importing necessary libraries for data manipulation, machine learning, and cloud services.
2. Azure and Snowflake Setup: It sets up connections to Azure Workspace and Snowflake database, retrieving credentials from Azure Key Vault.
3. Data Retrieval and Preprocessing: Data is queried from Snowflake, deduplicated, and prepared for modeling.
4. SetFit Model: A SetFit model is trained on the preprocessed data. SetFit is a few-shot learning method for text classification.
5. Logistic Regression Models: Two logistic regression models are trained and evaluated, using different text encoding methods.
6. DistilBERT Model: A DistilBERT model is implemented for text classification. This involves custom text segmentation, tokenization, and model training.
7. Evaluation and Results Saving: The models are evaluated using classification reports, and the results are saved for further analysis.

This pipeline demonstrates a comprehensive approach to text classification, utilizing various models and techniques to compare their performance on the given dataset.

# Import necessary libraries

import matplotlib.pyplot as plt
import snowflake.connector
import joblib
import scipy
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
from sqlalchemy.dialects import registry
from azureml.core import Workspace, Dataset, Run
from azureml.core.model import Model
from azure.identity import ManagedIdentityCredential, InteractiveBrowserCredential, DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE, MDS
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine
import torch
from sklearn.metrics import jaccard_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from datasets import Dataset
import evaluate
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import precision_score, recall_score
from umap import UMAP
from sklearn.metrics import classification_report

# Set pandas display options

pd.options.display.max_columns = None
pd.set_option('display.max_row', 2500)

# Azure Workspace setup

subscription_id = '75310040-ed56-4e53-83e0-80f23ef48cd6'
resource_group = 'dlsap-PaaS-PRD-rgp-001'
workspace_name = 'Finance_DS'
workspace = Workspace(subscription_id, resource_group, workspace_name)

# Azure Key Vault and Snowflake connection setup

credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url="https://financeds.vault.azure.net/", credential=credential)

# Retrieve Snowflake credentials from Azure Key Vault

account = secret_client.get_secret("snowflake-account").value
user = secret_client.get_secret("snowflake-user").value
password = secret_client.get_secret("snowflake-password").value
database = secret_client.get_secret("snowflake-database").value
role = secret_client.get_secret("snowflake-role").value
warehouse = secret_client.get_secret("snowflake-warehouse").value

# Create Snowflake connection

engine_1 = create_engine(URL(
    account = account,
    user = user,
    password = password,
    database = database,
    role = role,
    warehouse=warehouse,
    schema='sinergi'
))

# Query Snowflake database

query_1 = "select * from RW_ACTUALS_ALL_TR"
incidents = pd.read_sql(query_1, con=engine_1)

# Data preprocessing

incidents = incidents.drop_duplicates(['case_description_en'], keep=False)
incidents = incidents[~incidents['case_type'].isna()]

# Prepare data for modeling

texts = incidents['case_description_en'].values
targets = incidents['case_type'].values

# Split data into train and test sets

texts_train, texts_test, targets_train, targets_test = train_test_split(texts, targets, test_size=0.15)

# Encode target variables

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
Y_train = enc.fit_transform(targets_train.reshape(-1, 1))
Y_test = enc.fit_transform(targets_test.reshape(-1, 1))

# SetFit Model Training

model = SetFitModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

# Ordinal encoding of target variables

ord_enc = OrdinalEncoder()
ord_enc.fit(targets_train.reshape(-1, 1))
targets_test_ = ord_enc.transform(targets_test.reshape(-1, 1)).flatten()
targets_train_ = ord_enc.transform(targets_train.reshape(-1, 1)).flatten()

# Prepare datasets for SetFit

eval_dataset = Dataset.from_dict({'text':  texts_test, 'label': targets_test_})
train_dataset = Dataset.from_dict({'text':  texts_train, 'label': targets_train_})

# SetFit Trainer setup and training

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric='accuracy',
    num_iterations=5,
    num_epochs=2,
    batch_size=8,
)

trainer.train()
metrics = trainer.evaluate()

# Save trained model and data

joblib.dump((texts_train, texts_test, targets_train, targets_test), './data_.pkl', compress=9)

# Logistic Regression Model 1

model_ = SentenceTransformer('all-mpnet-base-v2')
X_train = model_.encode(texts_train, convert_to_tensor=False)
X_test = model_.encode(texts_test, convert_to_tensor=False)

clf = LogisticRegression(max_iter=1000).fit(X_train, targets_train_)
labels_pred = clf.predict(X_test)

preds_back = ord_enc.inverse_transform(labels_pred.reshape(-1, 1)).flatten()
targets_back = ord_enc.inverse_transform(targets_test_.reshape(-1, 1)).flatten()

print(classification_report(targets_back, preds_back))

# Logistic Regression Model 2

X_train = model.model_body.encode(texts_train, convert_to_tensor=False)
X_test = model.model_body.encode(texts_test, convert_to_tensor=False)

clf = LogisticRegression(max_iter=1000).fit(X_train, targets_train_)
labels_pred = clf.predict(X_test)

preds_back = ord_enc.inverse_transform(labels_pred.reshape(-1, 1)).flatten()
targets_back = ord_enc.inverse_transform(targets_test_.reshape(-1, 1)).flatten()

print(classification_report(targets_back, preds_back))

# DistilBERT Model

# ... (DistilBERT model definition and training code)

# Helper functions for text processing and sequence handling

# ... (Helper functions definition)

# Prepare data for DistilBERT

texts_train_segmented = [split_into_sentences(x) for x in texts_train]
texts_test_segmented = [split_into_sentences(x) for x in texts_test]

# Tokenization and padding

# ... (Tokenization and padding code)

# DistilBERT model training

# ... (DistilBERT model training code)

# Evaluate DistilBERT model

# ... (DistilBERT model evaluation code)

# Save results

df = pd.DataFrame({'text': texts_test, 'label': y_test_back, 'label_pred': y_pred_back})
joblib.dump(df, 'distill-bert-results.pkl')
