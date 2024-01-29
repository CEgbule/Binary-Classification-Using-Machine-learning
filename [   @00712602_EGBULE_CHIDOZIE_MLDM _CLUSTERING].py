#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# In[2]:


# Load the dataset
df = pd.read_csv('BodyFat-extended.csv')


# In[3]:


df.head()


# In[4]:


# Display basic information about the dataset
print("Dataset Info:")
print(df.info())


# In[5]:


# Display the first few rows of the dataset
print("\nFirst Few Rows:")
print(df.head())


# In[6]:


# Summary statistics
print(df.describe())


# In[7]:


# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns


# In[8]:


# Impute missing values for numeric columns with mean
numeric_imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])


# In[9]:


# Impute missing values for categorical columns with most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])


# In[10]:


# Display missing values after handling
print("\nMissing Values After Handling:")
print(df.isnull().sum())


# In[11]:


# Data Preprocessing
print("\nData Preprocessing:")


# In[12]:


# Standardize the data
scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)


# In[13]:


# To ignore warinings
import warnings
warnings.filterwarnings('ignore')


# In[14]:


# Clustering using K-Means
print("\nClustering using K-Means:")
# Determine the optimal number of clusters using the Elbow method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df_standardized)
    inertia.append(kmeans.inertia_)


# In[15]:


# Based on the Elbow method,3 to fit K-Means
optimal_k = 3  
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(df_standardized)


# In[16]:


# Plot the Elbow method
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Distances')
plt.show()


# In[17]:


# Create a dendrogram to determine the optimal number of clusters
linked = linkage(df_standardized, method='ward')
plt.figure(figsize=(8, 4))  # Specify both width and height
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram for Hierarchical Clustering')
plt.show()


# In[18]:


# Based on the dendrogram,i chose the optimal number of clusters 3 and fit Hierarchical Clustering
optimal_clusters_hierarchical = 3 
hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters_hierarchical)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(df_standardized)


# In[19]:


# Evaluate the clusters
print("\nSilhouette Score for K-Means:", silhouette_score(df_standardized, df['KMeans_Cluster']))
print("Silhouette Score for Hierarchical Clustering:",
      silhouette_score(df_standardized, df['Hierarchical_Cluster']))


# In[20]:


# Visualize the clusters
print("\nVisualizing Clusters:")
# Plot K-Means clusters
sns.scatterplot(data=df, x='BodyFat', y='Weight', hue='KMeans_Cluster', palette='viridis')
plt.title('K-Means Clustering')
plt.show()


# In[21]:


# Plot Hierarchical Clustering clusters
sns.scatterplot(data=df, x='BodyFat', y='Weight', hue='Hierarchical_Cluster', palette='viridis')
plt.title('Hierarchical Clustering')
plt.show()


# In[22]:


# Display the first few rows with cluster labels
print(df.head())


# In[23]:


# Create a pair plot
df = df.iloc[:,1:9]
sns.pairplot(df)


# In[24]:


# Specify the number of components i want to keep 
n_components = 2


# In[25]:


# Instantiate the PCA object
pca = PCA(n_components=n_components)


# In[26]:


# Fit and transform the data using PCA
X_pca = pca.fit_transform(df_standardized)


# In[27]:


# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)


# In[28]:


sum(explained_variance_ratio)


# In[29]:


# Create a DataFrame with the PCA results
pca_df = pd.DataFrame(X_pca, columns=['BodyFat', 'Weight'])


# In[30]:


# Add the K-Means cluster labels to the DataFrame
pca_df['BodyFat'] = df['Weight']


# In[31]:


# Assuming 'pca_df' is the DataFrame after applying PCA and the column name for K-Means labels is 'KMeans_Cluster'
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_df['BodyFat'], pca_df['Weight'], c=pca_df['Weight'], cmap='viridis')
plt.title('K-Means Clustering Visualization (PCA)')
plt.xlabel('BodyFat')
plt.ylabel('Weight')
plt.colorbar(scatter, label='K-Means Cluster')
plt.show()


# In[32]:


# Based on the Elbow method, i have the optimal K and fit my K-Means
optimal_k = 3 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(df_standardized)


# In[33]:


# Assuming 'df_standardized' is the standardized feature matrix and 'y_kmeans' is the K-Means cluster labels
plt.figure(figsize=(4, 4))
plt.scatter(df_standardized['BodyFat'][df['KMeans_Cluster'] == 0], df_standardized['Weight'][df['KMeans_Cluster'] == 0], s=100, c='red', label='Cluster 1')
plt.scatter(df_standardized['BodyFat'][df['KMeans_Cluster'] == 1], df_standardized['Weight'][df['KMeans_Cluster'] == 1], s=100, c='blue', label='Cluster 2')
plt.scatter(df_standardized['BodyFat'][df['KMeans_Cluster'] == 2], df_standardized['Weight'][df['KMeans_Cluster'] == 2], s=100, c='green', label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('K-Means Clustering Visualization')
plt.xlabel('BodyFat (Standardized)')
plt.ylabel('Weight (Standardized)')
plt.legend()
plt.show()


# In[34]:


# performing dimentional pca reduction and scatter plot


# In[35]:


# Perform PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(df_standardized)


# In[36]:


# Perform K-Means clustering
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(df_standardized)


# In[37]:


# Visualize the clusters in the reduced-dimensional space
colours = ['Red', 'Blue', 'Green']


# In[38]:


plt.figure(figsize=(6, 6))
for i in range(num_clusters):
    plt.scatter(X_reduced[y_kmeans == i, 0], X_reduced[y_kmeans == i, 1],
                s=100, c=colours[i], label='Cluster ' + str(i + 1))
plt.title('K-Means Clustering Visualization (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()


# In[ ]:




