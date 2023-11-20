# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 20:10:26 2023

@author: Yigitalp
"""
# Import relevant libraries
import warnings
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import math
from lifetimes import GammaGammaFitter
from lifetimes import BetaGeoFitter
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default = 'svg'
warnings.filterwarnings('ignore')

# Load dataset
original_df = pd.read_csv('flo_data_20k.csv', parse_dates=True)
df = original_df.copy()

# Check nulls and dtypes
df.info()

# Check unique master_id: 19945 (same as # of records)
# So, it can be replaced with index since index can also represent id
unique_master_id = df['master_id'].nunique()
df['master_id'] = df.index

# Find columns including date and convert them to datetime
column_dates = df.columns[df.columns.str.contains('date')].to_list()
df[column_dates] = df[column_dates].apply(pd.to_datetime)

# Find total number of orders and values
df['total_order'] = df['order_num_total_ever_online'] + \
    df['order_num_total_ever_offline']
df['total_value'] = df['customer_value_total_ever_offline'] + \
    df['customer_value_total_ever_online']

#%%
# RFM analysis
# Group per customer to calculate last order date, total order, and total value
df_rfm = df.groupby('master_id', as_index=False).agg(
    last_order_date_max=('last_order_date', np.max),
    sum_of_total_orders=('total_order', np.sum),
    sum_of_total_values=('total_value', np.sum))

# Calculate recency
# Rename relevant columns as frequency and monetary for better traceability
df_rfm['recency'] = (datetime.today() - df_rfm['last_order_date_max']).dt.days
df_rfm.rename(columns={'sum_of_total_orders': 'frequency',
                       'sum_of_total_values': 'monetary'}, inplace=True)

# Encode RFM metrics:
# Recency score works inversely; min value with max label
df_rfm['recency_score'] = pd.qcut(
    df_rfm['recency'], 5, labels=list(range(5, 0, -1)))

# Prevent frequency values fall into multiple labels, rank method is used
# It works regularly; min value with min label
df_rfm['frequency_score'] = pd.qcut(df_rfm['frequency'].rank(
    method="first"), 5, labels=list(range(1, 6)))

# Monetary score works regularly; min value with min label
df_rfm['monetary_score'] = pd.qcut(
    df_rfm['monetary'], 5, labels=list(range(1, 6)))

# For segmentation map, rf score is constructed
df_rfm["rf_score"] = (df_rfm['recency_score'].astype(str) +
                      df_rfm['frequency_score'].astype(str))

# Segmentation map
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# Replace RF scores with segments: regualar expression is true to catch the pairs in map
df_rfm['segment'] = df_rfm['rf_score'].replace(seg_map, regex=True)

# Radar Plot segments per mean and sum monetary
# Convert monetary score to int
df_rfm['monetary_score'] = df_rfm['monetary_score'].astype(int)

# Create aggregations of monetary and monetary score per segment
group_rfm_segment = df_rfm.groupby('segment', as_index=False).agg(
    monetary_mean=('monetary', np.mean),
    monetary_sum=('monetary', np.sum),
    monetary_score_mean=('monetary_score', np.mean),
    monetary_score_sum=('monetary_score', np.sum))

# Find numeric columns
columns_numeric = group_rfm_segment.select_dtypes(
    include=np.number).columns.to_list()

# Normalize numeric columns
for col in columns_numeric:
    group_rfm_segment[col] = (group_rfm_segment[col]-group_rfm_segment[col].min())/(
        group_rfm_segment[col].max()-group_rfm_segment[col].min())

# Create categories from segments
categories = group_rfm_segment['segment'].to_list()

# Plot radar chart and save it
fig = go.Figure()
for col in columns_numeric:
    fig.add_trace(go.Scatterpolar(
        r=group_rfm_segment[col].to_list(),
        theta=categories,
        fill='toself',
        name=f'{col}'
    ))
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True
)
fig.write_image('Monetary Distribution per RF Segment Radar.pdf')

""" OPTIONAL
# Bar Plot segments per total monetary
group_rfm_segment = df_rfm.groupby('segment', as_index=False)['monetary'].sum()
order = group_rfm_segment.sort_values('monetary', ascending=False)['segment']
fig, ax = plt.subplots(figsize=(12, 8))
sns.barplot(data=group_rfm_segment, x='segment',
            y='monetary', order=order, ax=ax)
plt.xticks(rotation=90, ha='center')
plt.suptitle('Monetary Distribution')
plt.title('RF Segment')
for label in ax.containers:
    ax.bar_label(label,)
plt.tight_layout()
plt.savefig('Monetary Distribution per RF Segment Bar.pdf', dpi=250)
"""

#%%
# CLTV analysis
# Create new df for cltv
df_cltv = original_df.copy()

# Outlier detection and replacement
# Create a list of numeric columns
columns_numeric = df_cltv.select_dtypes(include=np.number).columns.to_list()

# Plot box plots to observe distributions and outliers
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(df_cltv[columns_numeric])
plt.suptitle('Numeric Columns Dist.')
plt.title('Box Plots')
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.savefig('Box Plots before Outlier Handling 1.pdf', dpi=250)

# Detect outliers and replace them with lower and upper limits accordingly
for col in columns_numeric:

    # Calculate MinMax values per iqr of each numeric column
    iqr = df_cltv[col].quantile(0.75)-df_cltv[col].quantile(0.25)
    lower_limit = df_cltv[col].quantile(0.25)-1.5*iqr
    upper_limit = df_cltv[col].quantile(0.75)+1.5*iqr

    # Create conditions and choices to replace outliers per MinMax values
    conditions = (df_cltv[col] < lower_limit, df_cltv[col] > upper_limit)
    choices = (lower_limit, upper_limit)
    default = df_cltv[col]
    df_cltv[col] = np.select(conditions, choices, default)

# Replot box plots to observe if outliers are handled
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(df_cltv[columns_numeric])
plt.suptitle('Numeric Columns Dist.')
plt.title('Box Plots')
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.savefig('Box Plots after Outlier Handling 1.pdf', dpi=250)

# Check columns and their number of unique values
print(df_cltv.nunique())

# Same wrangling is applied
df_cltv[column_dates] = df_cltv[column_dates].apply(pd.to_datetime)
df_cltv['total_order'] = df_cltv['order_num_total_ever_online'] + \
    df_cltv['order_num_total_ever_offline']
df_cltv['total_value'] = df_cltv['customer_value_total_ever_offline'] + \
    df_cltv['customer_value_total_ever_online']

# Create cltv dataframe
cltv = pd.DataFrame()

# Insert master_id
cltv['master_id'] = df_cltv.index

# Frequency is calculated by total order
# Round down the frequency to the nearest integer
cltv['frequency'] = df_cltv['total_order']
cltv['frequency'] = cltv['frequency'].apply(int)

# Recency per the latest transaction (weekly)
cltv['recency_weekly'] = (
    (datetime.today()-df_cltv['last_order_date']).dt.days)/7

# Customer age per the first transaction (weekly)
cltv['T_weekly'] = (
    (datetime.today()-df_cltv['first_order_date']).dt.days)/7

# Average monetary per order
cltv["monetary_avg"] = df_cltv['total_value'] / df_cltv['total_order']

# Create BG/NBD model
bgf = BetaGeoFitter(penalizer_coef=0.001)

# Fit model per relevant features
bgf.fit(cltv['frequency'],
        cltv['recency_weekly'],
        cltv['T_weekly'])

# 3 months sales predicted for customer
cltv['pred_sales_3_month'] = bgf.predict(12,
                                         cltv['frequency'],
                                         cltv['recency_weekly'],
                                         cltv['T_weekly'])

# 6 months sales predicted for customer
cltv['pred_sales_6_month'] = bgf.predict(24,
                                         cltv['frequency'],
                                         cltv['recency_weekly'],
                                         cltv['T_weekly'])


# Create GG model
ggf = GammaGammaFitter(penalizer_coef=0.01)

# Fit model per relevant features
ggf.fit(cltv['frequency'], cltv['monetary_avg'])

# Calculate predicted average profit
cltv['pred_average_value'] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                     cltv['monetary_avg'])

# Calculate CLTV
customer_lifetime_value = ggf.customer_lifetime_value(bgf,
                                                      cltv['frequency'],
                                                      cltv['recency_weekly'],
                                                      cltv['T_weekly'],
                                                      cltv['monetary_avg'],
                                                      time=6,  # 6 months
                                                      freq="W",  # Unit is week
                                                      discount_rate=0.01)

# Create cltv column
cltv['cltv'] = customer_lifetime_value

# Create equal amount of segments per rfm
cltv['segment'] = pd.qcut(cltv['cltv'], 10, labels=list(range(10)))
cltv['segment'] = cltv['segment'].astype(int)

#%%
# Machine Learning model

# Create dataframe for ML
df_ML = original_df.copy()

# Same wrangling is applied
df_ML.drop('master_id', axis=1, inplace=True)
df_ML[column_dates] = df_ML[column_dates].apply(pd.to_datetime)

# Only difference is converting datetime values to numeric
df_ML[column_dates] = df_ML[column_dates].apply(
    lambda x: (datetime.today()-x).dt.days)

# Split interested_in_categories_12:
df_ML['aktifcocuk_12'] = np.where(
    df_ML['interested_in_categories_12'].str.contains('AKTIFCOCUK'), 1, 0)
df_ML['aktifspor_12'] = np.where(
    df_ML['interested_in_categories_12'].str.contains('AKTIFSPOR'), 1, 0)
df_ML['cocuk_12'] = np.where(
    df_ML['interested_in_categories_12'].str.contains('COCUK'), 1, 0)
df_ML['erkek_12'] = np.where(
    df_ML['interested_in_categories_12'].str.contains('ERKEK'), 1, 0)
df_ML['kadin_12'] = np.where(
    df_ML['interested_in_categories_12'].str.contains('KADIN'), 1, 0)
df_ML.drop('interested_in_categories_12', axis=1, inplace=True)

# Insert rfm label and encode label per hierarchy
df_ML['rfm_label'] = df_rfm['segment']
ordered_label_list = list(seg_map.values())
df_ML['rfm_label'].replace(ordered_label_list, list(
    range(len(ordered_label_list))), inplace=True)

# Insert cltv label
df_ML['cltv_label'] = cltv['segment']

# Calculate label, roundup its final score, and drop rfm and cltv labels
df_ML['label'] = 0.5*(df_ML['rfm_label']+df_ML['cltv_label'])
df_ML['label'] = df_ML['label'].apply(math.ceil)
df_ML.drop(['rfm_label', 'cltv_label'], axis=1, inplace=True)

# Encode order channel per label mean
group_order_channel = df_ML.groupby('order_channel', as_index=False)[
    'label'].mean().sort_values('label').reset_index()
group_order_channel.drop('index', axis=1, inplace=True)
df_ML['order_channel'].replace(group_order_channel['order_channel'].to_list(
), group_order_channel.index.to_list(), inplace=True)

# Encode last order channel per label mean
group_last_order_channel = df_ML.groupby('last_order_channel', as_index=False)[
    'label'].mean().sort_values('label').reset_index()
group_last_order_channel.drop('index', axis=1, inplace=True)
df_ML['last_order_channel'].replace(group_last_order_channel['last_order_channel'].to_list(
), group_last_order_channel.index.to_list(), inplace=True)

# Save label column for further supervised learning
y = df_ML['label']

# Drop label column for further unsupervised learning
df_ML.drop('label', axis=1, inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df_ML = pd.DataFrame(scaler.fit_transform(df_ML), columns=df_ML.columns)

# Plot box plots to observe distributions and outliers
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(df_ML[columns_numeric])
plt.suptitle('Numeric Columns Dist.')
plt.title('Box Plots')
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.savefig('Box Plots before Outlier Handling 2.pdf', dpi=250)

# Outlier detection and replacement
for col in df_ML.columns:
    # Calculate MinMax values per iqr of each numeric column
    iqr = df_ML[col].quantile(0.75)-df_ML[col].quantile(0.25)
    lower_limit = df_ML[col].quantile(0.25)-1.5*iqr
    upper_limit = df_ML[col].quantile(0.75)+1.5*iqr

    # Create conditions and choices to replace outliers per MinMax values
    conditions = (df_ML[col] < lower_limit, df_ML[col] > upper_limit)
    choices = (lower_limit, upper_limit)
    default = df_ML[col]
    df_ML[col] = np.select(conditions, choices, default)

# Recreate boxplots to observe if outliers are handled
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(df_ML[columns_numeric])
plt.suptitle('Numeric Columns Dist.')
plt.title('Box Plots')
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.savefig('Box Plots after Outlier Handling 2.pdf', dpi=250)

# Check columns and their number of unique values
print(df_ML.nunique())

# Recreate aktifcocuk_12, since it has 1 unique value after outlier detection
df_ML['aktifcocuk_12'] = np.where(
    original_df['interested_in_categories_12'].str.contains('AKTIFCOCUK'), 1, 0)

# Recheck columns and their number of unique values
print(df_ML.nunique())


# Clustering algorithm with silhoutte score

# Set a pool for number of clusters
num_clusters = range(2, 12)

# Create an empty list to track silhouette scores
silhouettes = []

# For each cluster number: create a model, fit the model, get labels,
# calculate silhouette scores, and then save them to the list
for num_cluster in num_clusters:
    model = KMeans(n_clusters=num_cluster, random_state=42)
    model.fit(df_ML)
    labels = model.labels_
    sil_score = silhouette_score(df_ML, labels)
    silhouettes.append(sil_score)
    print(num_cluster, sil_score)

# Plot silhoutte scores to find optimal number of clusters
# I chose 10 clusters where silhouette score is optimal
# Coincidentally equal amount of clusters per label
fig, ax = plt.subplots(figsize=(12, 8))
plt.plot(num_clusters, silhouettes, '-', c='b')
plt.scatter(num_clusters, silhouettes, marker='x', c='r')
plt.suptitle('Optimal Clustering')
plt.title('Silhouette Scores')
plt.xticks(num_clusters)
plt.xlabel('Cluster')
plt.ylabel('Silhouette')
plt.tight_layout()
plt.savefig('Optimal Clustering per Silhouette Scores.pdf', dpi=250)

# Optimal clustering
model_optimal = KMeans(n_clusters=10, random_state=42)
model_optimal.fit(df_ML)
df_ML['cluster'] = model_optimal.labels_

# Reinsert label
df_ML['label'] = y

# Calculate score
df_ML['score'] = 0.5*(df_ML['cluster']+df_ML['label'])
df_ML['score'] = df_ML['score'].apply(math.ceil)

# Drop cluster and label to only keep score as target
df_ML.drop(['cluster', 'label'], axis=1, inplace=True)

# Encode order channel per score mean
df_ML['order_channel'] = original_df['order_channel']
group_order_channel = df_ML.groupby('order_channel', as_index=False)[
    'score'].mean().sort_values('score').reset_index()
group_order_channel.drop('index', axis=1, inplace=True)
df_ML['order_channel'].replace(group_order_channel['order_channel'].to_list(
), group_order_channel.index.to_list(), inplace=True)

# Encode last order channel per score mean
df_ML['last_order_channel'] = original_df['last_order_channel']
group_last_order_channel = df_ML.groupby('last_order_channel', as_index=False)[
    'score'].mean().sort_values('score').reset_index()
group_last_order_channel.drop('index', axis=1, inplace=True)
df_ML['last_order_channel'].replace(group_last_order_channel['last_order_channel'].to_list(
), group_last_order_channel.index.to_list(), inplace=True)

# Normalize theese columns
scaler2 = MinMaxScaler()
df_ML[['order_channel', 'last_order_channel']] = pd.DataFrame(scaler2.fit_transform(
    df_ML[['order_channel', 'last_order_channel']]), columns=['order_channel', 'last_order_channel'])

# Plot box plots to observe distributions and outliers
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(df_ML[['order_channel', 'last_order_channel']])
plt.suptitle('Numeric Columns Dist.')
plt.title('Box Plots')
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.savefig('Box Plots before Outlier Handling 3.pdf', dpi=250)

# Outlier detection and replacement for only order channel and last order channel
for col in ['order_channel', 'last_order_channel']:
    # Calculate MinMax values per iqr of each numeric column
    iqr = df_ML[col].quantile(0.75)-df_ML[col].quantile(0.25)
    lower_limit = df_ML[col].quantile(0.25)-1.5*iqr
    upper_limit = df_ML[col].quantile(0.75)+1.5*iqr

    # Create conditions and choices to replace outliers per MinMax values
    conditions = (df_ML[col] < lower_limit, df_ML[col] > upper_limit)
    choices = (lower_limit, upper_limit)
    default = df_ML[col]
    df_ML[col] = np.select(conditions, choices, default)

# Recreate boxplots to observe if outliers are handled
fig, ax = plt.subplots(figsize=(12, 8))
sns.boxplot(df_ML[['order_channel', 'last_order_channel']])
plt.suptitle('Numeric Columns Dist.')
plt.title('Box Plots')
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.savefig('Box Plots after Outlier Handling 3.pdf', dpi=250)

# Recheck columns and their number of unique values
print(df_ML.nunique())

# Create X and y
X = df_ML.drop('score', axis=1)
y = df_ML['score']

# Perform train and test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create classifier and fit it
classifier = XGBClassifier(seed=42)
classifier.fit(X_train, y_train)

# Plot importance
plot_importance(classifier)
plt.tight_layout()
plt.savefig('Feature Importance Bar Plot.pdf', dpi=250)

# Plot Best Decision Tree
fig, ax = plt.subplots(figsize=(20, 15))
plot_tree(classifier, ax=ax,
          num_trees=classifier.get_booster().best_iteration, rankdir='LR')
plt.tight_layout()
plt.savefig('XGBoost Classifier Decision Tree.pdf', dpi=250)

# Make predictions and create classification report
y_pred = classifier.predict(X_test)
report = classification_report(y_test, y_pred)

# Insert Predictions and Control
y_test = y_test.reset_index()
y_test['Predictions'] = y_pred
y_test['Control'] = y_test['score'] == y_test['Predictions']
print(y_test['Control'].value_counts())
