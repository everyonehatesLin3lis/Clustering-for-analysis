import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import numpy as np
import seaborn as sns
import warnings


def clustering_data(data: pd.DataFrame, column: str):
    """
    This code take pandas DataFrame and chosen column name.
    1. Reshape given column from given DataFrame to values (-1: 1).
    2. Uses Silhouette method to determine clusters size between 4 and 6.
    3. Apply clusters values on 'winner' DataFrame of chosen column.
    4. Calculates count of values near clusters.
    5. Plots graphic.
    """

    pd.options.mode.chained_assignment = None  # default='warn'

    data_for_clusters = data[column].values.reshape(-1, 1)

    range_n_clusters = range(4, 6)

    best_score = 0.25
    best_n_clusters = None

    # Iterate over the range of cluster numbers
    for n_clusters in range_n_clusters:
        # Fit KMeans clustering model
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clusterer.fit_predict(data_for_clusters)
        
        # Compute silhouette score
        silhouette_avg = silhouette_score(data_for_clusters, cluster_labels)
        
        print(silhouette_avg)
        # Determine if this silhouette score is the best so far
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_n_clusters = n_clusters
        #else: best_n_clusters = 4

    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    kmeans.fit(data_for_clusters)

    # Predicting the clusters
    clusters = kmeans.predict(data_for_clusters)

    # Counting the number of counties in each cluster
    cluster_counts = pd.Series(clusters).value_counts().sort_index()

    # Creating a DataFrame to display both counts and cluster values
    cluster_info = pd.DataFrame({'Cluster': cluster_counts.index, 'Count': cluster_counts.values})

    # Printing the cluster centers (values)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Cluster Value'])
    cluster_info = pd.concat([cluster_info, cluster_centers], axis=1)

    cluster_info_sorted = cluster_info.sort_values(by='Cluster Value')

    unique_winners = data['winner'].unique()

    # Iterate over each winner
    for winner in unique_winners:
        # Filter data for the current winner
        winner_data = data[data['winner'] == winner]
        
        # Calculate Euclidean distance for each data point to each centroid
        for i in range(len(cluster_info)):
            winner_data[f'Cluster: {i}'] = np.sqrt((winner_data[column] - cluster_info.at[i, 'Cluster Value'])**2)
        
        # Determine the nearest centroid for each data point
        winner_data['Nearest_Centroid'] = winner_data[[f'Cluster: {i}' for i in range(len(cluster_info))]].idxmin(axis=1)
        
        # Count the number of data points assigned to each centroid
        centroid_counts = winner_data['Nearest_Centroid'].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(centroid_counts.index, centroid_counts.values, color=sns.color_palette('RdPu', n_colors=4), alpha=0.6, label='Cluster Count')
        plt.xlabel('Clusters')
        plt.ylabel('Count')
        plt.title(f'{column}, {winner}')
        plt.xticks(centroid_counts.index, round((cluster_info['Cluster Value']), 2))
        plt.yticks([])

        # Add cluster values on top of bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{centroid_counts.values[i]:.0f}", 
                    ha='center', va='bottom')

        plt.show()