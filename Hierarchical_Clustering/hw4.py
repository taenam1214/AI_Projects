import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram  # Import dendrogram for plotting

def load_data(filepath):
    data = []
    with open(filepath, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(dict(row))
    return data

def calc_features(row):
    features = np.array([
        float(row['Population']),
        float(row['Net migration']),
        float(row['GDP ($ per capita)']),
        float(row['Literacy (%)']),
        float(row['Phones (per 1000)']),
        float(row['Infant mortality (per 1000 births)'])
    ], dtype=np.float64)
    return features

def normalize_features(features):
    features = np.array(features)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / stds
    return list(normalized_features)

import numpy as np

def hac(data_points):
    num_points = len(data_points)
    distance_matrix = np.full((num_points, num_points), np.inf)
    
    clusters = {label: [label] for label in range(num_points)}
    
    # Calculate distance matrix
    for i in range(num_points):
        for j in range(i + 1, num_points):
            distance_matrix[i][j] = distance_matrix[j][i] = np.linalg.norm(data_points[i] - data_points[j])
    
    linkage_matrix = np.empty((0, 4))
    next_cluster_label = num_points
    
    for _ in range(num_points - 1):
        min_distance = np.inf
        closest_pair = (0, 0)
        
        for i in clusters.keys():
            for j in clusters.keys():
                if i >= j:
                    continue
                distances = [distance_matrix[ci][cj] for ci in clusters[i] for cj in clusters[j]]
                max_distance = max(distances)
                if max_distance < min_distance:
                    min_distance = max_distance
                    closest_pair = (i, j)
        
        # Merge clusters and update linkage matrix
        new_cluster = clusters[closest_pair[0]] + clusters[closest_pair[1]]
        linkage_matrix = np.vstack([linkage_matrix, [*closest_pair, min_distance, len(new_cluster)]])
        
        # Update clusters
        clusters[next_cluster_label] = new_cluster
        del clusters[closest_pair[0]], clusters[closest_pair[1]]
        
        next_cluster_label += 1
        
    return linkage_matrix


def fig_hac(Z, names):
    fig = plt.figure(figsize=(10, 8))
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    plt.show()
    return fig

if __name__ == "__main__":
    # Update the filepath according to your file location
    filepath = 'countries.csv'
    data = load_data(filepath)
    country_names = [row["Country"].strip() for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    
    n = 20  # Adjust as needed for testing
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    
    fig_raw = fig_hac(Z_raw, country_names[:n])
    fig_normalized = fig_hac(Z_normalized, country_names[:n])