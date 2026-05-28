import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# load dataset
df = pd.read_csv("Mall_Customers.csv.csv")
print(df.head())

# taking only numeric columns
data = df.select_dtypes(include=[np.number])

# removing customer id
if 'CustomerID' in data.columns:
    data = data.drop('CustomerID', axis=1)

# removing null values
data = data.dropna()

print("\nColumns used:")
print(data.columns)

# scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# elbow method
wcss = []

for i in range(1,11):
    model = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )

    model.fit(scaled_data)
    wcss.append(model.inertia_)

# graph for elbow method
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# training model
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)

pred = kmeans.fit_predict(scaled_data)

# adding cluster column
df['Cluster'] = pred

# plotting clusters
plt.figure(figsize=(10,6))

colors = ['red','blue','green','orange','purple']

for i in range(5):
    plt.scatter(
        scaled_data[pred == i,0],
        scaled_data[pred == i,1],
        s=80,
        c=colors[i],
        label=f'Cluster {i+1}'
    )

# centroids
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=250,
    c='black',
    marker='X',
    label='Centroids'
)

plt.title('Customer Segmenation using K-Means')
plt.xlabel(data.columns[0])
plt.ylabel(data.columns[1])
plt.legend()
plt.grid(True)
plt.show()

# cluster summary
print("\nCluster Analysis:\n")
summary = df.groupby('Cluster')[data.columns].mean()
print(summary)