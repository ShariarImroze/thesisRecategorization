1. Data loading and preprocessing:
   * Connects to a Snowflake database using Azure credentials
   * Loads incident data and removes duplicates
   * Splits data into training and testing sets
2. Model training and evaluation:
   * Uses SetFitModel for initial training
   * Implements custom metric computation
   * Trains and evaluates the model
3. Embedding visualization:
   * Generates embeddings using both raw and fine-tuned models
   * Uses UMAP for dimensionality reduction
   * Visualizes embeddings for each target class
4. Logistic Regression experiments:
   * Trains logistic regression models using both fine-tuned and standard inputs
   * Compares performance and misclassifications between the two approaches
5. Misclassification analysis:
   * Identifies and displays misclassified texts
   * Compares misclassifications between fine-tuned and standard models

This code provides a comprehensive approach to text classification, including model training, evaluation, visualization, and error analysis. It demonstrates the use of various machine learning libraries and techniques for natural language processing tasks.

# Import required libraries

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
from sklearn.metrics import pairwise_distances, precision_score, recall_score
from sklearn.manifold import TSNE, MDS
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine
import torch
from sklearn.metrics import jaccard_score, multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance
from datasets import load_dataset, Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import evaluate
from umap import UMAP

# Set pandas display options

pd.options.display.max_columns = None
pd.set_option('display.max_row', 2500)

# Azure workspace configuration

subscription_id = '75310040-ed56-4e53-83e0-80f23ef48cd6'
resource_group = 'dlsap-PaaS-PRD-rgp-001'
workspace_name = 'Finance_DS'

# Create Azure ML workspace

workspace = Workspace(subscription_id, resource_group, workspace_name)

# Set up Azure Key Vault credentials and client

credential = DefaultAzureCredential()
secret_client = SecretClient(vault_url="https://financeds.vault.azure.net/", credential=credential)

# Retrieve Snowflake credentials from Azure Key Vault

account = secret_client.get_secret("snowflake-account").value
user = secret_client.get_secret("snowflake-user").value
password = secret_client.get_secret("snowflake-password").value
database = secret_client.get_secret("snowflake-database").value
role = secret_client.get_secret("snowflake-role").value
warehouse = secret_client.get_secret("snowflake-warehouse").value

# Create Snowflake database engine

engine_1 = create_engine(URL(
            account = account,
            user = user,
            password = password,
            database = database,
            role = role,
            warehouse=warehouse,
            schema='sinergi'
        ))

# SQL query to retrieve data

query_1 = f""" select * from RW_ACTUALS_ALL_TR
        """

# Execute query and load data into pandas DataFrame

incidents = pd.read_sql(query_1, con=engine_1)

# Display specific row from the DataFrame

incidents[incidents['case_description_en'] == 'The old emergency number was still attached to the barrier at the Garstadt power plant']

# Display first 3 rows of the DataFrame

incidents.head(3)

# Remove duplicate cases and rows with missing case_type

incidents = incidents.drop_duplicates(['caseno', 'case_type'])
incidents = incidents[~incidents['case_type'].isna()]

# Extract text and target variables

texts = incidents['case_description_en'].values
targets = incidents['case_type'].values

# Split data into training and testing sets

texts_train, texts_test, targets_train, targets_test = train_test_split(texts, targets, test_size=0.15)

# One-hot encode target variables

enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
Y_train = enc.fit_transform(targets_train.reshape(-1, 1))
Y_test = enc.fit_transform(targets_test.reshape(-1, 1))

# Load pre-trained SetFit model

model = SetFitModel.from_pretrained(
    'sentence-transformers/all-mpnet-base-v2',
    use_differentiable_head=True,
    head_params={'out_features': 6}, )

# Load pre-trained SentenceTransformer model

model_ = SentenceTransformer('all-mpnet-base-v2')

# Ordinal encode target variables

ord_enc = OrdinalEncoder()
ord_enc.fit(targets_train.reshape(-1, 1))

targets_test_ = ord_enc.transform(targets_test.reshape(-1, 1))
targets_train_ = ord_enc.transform(targets_train.reshape(-1, 1))

# Flatten target arrays

targets_test_ = targets_test_.flatten()
targets_train_ = targets_train_.flatten()

# Create datasets for evaluation and training

eval_dataset = Dataset.from_dict({'text':  texts_test, 'label': targets_test_})
train_dataset = Dataset.from_dict({'text':  texts_train, 'label': targets_train_})

# Define function to compute metrics

def compute_metrics(y_pred, y_test):
    y_pred = list(y_pred.cpu().numpy())
    return {
        "presision": precision_score(y_test, y_pred, pos_label='positive', average='micro'),
        "recall": recall_score(y_test, y_pred, pos_label='positive', average='micro'),
    }

# Set up SetFit trainer

trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    metric=compute_metrics,
    num_iterations=5,
    num_epochs=2,
    batch_size=8,
)

# Train the model

trainer.train()

# Evaluate the model

metrics = trainer.evaluate()

# Display evaluation metrics

metrics

# Make predictions using the trained model

preds = model.predict(texts_test)

# Convert predictions back to original labels

preds_back = ord_enc.inverse_transform(preds.cpu().numpy().reshape(-1, 1)).flatten()

# Create DataFrame with original texts, true labels, and predicted labels

df = pd.DataFrame({'text': texts_test, 'label': targets_test, 'label_pred': preds_back})

# Filter DataFrame for misclassifications (predicted as Environment but not actually Environment)

df_ = df[(df['label'] != 'Environment') & (df['label_pred'] == 'Environment')]

# Display misclassified texts

for t in df_['text']:
    display(t)

# Load saved model and data

model = SetFitModel._from_pretrained('./model/')
texts_train, texts_test, targets_train, targets_test = joblib.load('data.pkl')

# Encode texts using both raw and fine-tuned models

embeddings_raw = model_.encode(texts_test, convert_to_tensor=False)
embeddings_finetuned = model.model_body.encode(texts_test, convert_to_tensor=False)

# Reduce dimensionality of embeddings using UMAP

embeddings_raw_2d = UMAP(n_neighbors=32, n_components=2,
                         min_dist=0.0, metric='cosine').fit_transform(embeddings_raw)

embeddings_finetuned_2d = UMAP(n_neighbors=32, n_components=2,
                               min_dist=0.0, metric='cosine').fit_transform(embeddings_finetuned)

# Visualize embeddings for each target class

targets_set = set(targets_test)
for target in targets_set:
    has_label = [True if target in x else False for x in targets_test]
    fig, axs = plt.subplots(1,2, figsize=(8, 4), sharey='row')
    ax1,ax2 = axs
    print(target)
    ax1.scatter(embeddings_raw_2d[:,0], embeddings_raw_2d[:,1], c=has_label, alpha=0.2)
    ax2.scatter(embeddings_finetuned_2d[:,0], embeddings_finetuned_2d[:,1], c=has_label, alpha=0.2)
    plt.show()

# Logistic Regression with fine-tuned inputs

# Encode training and testing texts using fine-tuned model

X_train = model.model_body.encode(texts_train, convert_to_tensor=False)
X_test = model.model_body.encode(texts_test, convert_to_tensor=False)

# Train logistic regression model

clf = LogisticRegression(max_iter=1000).fit(X_train, targets_train_)

# Make predictions

labels_pred = clf.predict(X_test)

# Define function to compute metrics

def compute_metrics(y_pred, y_test):
    return {
        "presision": precision_score(y_test, y_pred, average='micro'),
        "recall": recall_score(y_test, y_pred, average='micro'),
    }

# Compute and display metrics

compute_metrics(labels_pred, targets_test_)

# Convert predictions back to original labels

preds_back = ord_enc.inverse_transform(labels_pred.reshape(-1, 1)).flatten()

# Create DataFrame with original texts, true labels, and predicted labels

df = pd.DataFrame({'text': texts_test, 'label': targets_test, 'label_pred': preds_back})

# Filter DataFrame for misclassifications (predicted as Environment but not actually Environment)

df_ = df[(df['label'] != 'Environment') & (df['label_pred'] == 'Environment')]

# Display misclassified texts

for t in df_['text']:
    display(t)

# Logistic Regression with standard inputs

# Encode training and testing texts using standard model

model_ = SentenceTransformer('all-mpnet-base-v2')
X_train = model_.encode(texts_train, convert_to_tensor=False)
X_test = model_.encode(texts_test, convert_to_tensor=False)

# Train logistic regression model

clf = LogisticRegression(max_iter=1000).fit(X_train, targets_train_)

# Make predictions

labels_pred = clf.predict(X_test)

# Compute and display metrics

compute_metrics(labels_pred, targets_test_)

# Convert predictions back to original labels

preds_back = ord_enc.inverse_transform(labels_pred.reshape(-1, 1)).flatten()

# Create DataFrame with original texts, true labels, and predicted labels

df = pd.DataFrame({'text': texts_test, 'label': targets_test, 'label_pred': preds_back})

# Filter DataFrame for misclassifications (actual Environment but not predicted as Environment)

df__ = df[(df['label'] == 'Environment') & (df['label_pred'] != 'Environment')]

# Display misclassified texts

for t in df__['text']:
    display(t)

# Compare misclassifications between fine-tuned and standard models

a = set(df_['text']) - set(df__['text'])
b = set(df__['text']) - set(df_['text'])

# Print number of unique misclassifications for each model

print(len(a))
print(len(b))
print(len(set(df_['text']) & set(df__['text'])))

# Display unique misclassifications for fine-tuned model

for a_ in a:
    display(a_)

# Display unique misclassifications for standard model

for b_ in b:
    display(b_)
