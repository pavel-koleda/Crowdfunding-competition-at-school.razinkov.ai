import os

import cv2
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE
from tqdm import tqdm

from configs.data_config import data_config
from configs.experiment_config import experiment_config
from dataset.clustering.agglomerative import AgglomerativeClustering
from utils.common_functions import read_dataframe_file, read_file


def read_data():
    """Reads all data to clusterize."""
    annotation = read_dataframe_file(os.path.join(data_config.path_to_data, data_config.annotation_filename))
    images = []

    # Read images
    for i, row in tqdm(annotation.iterrows(), total=len(annotation), desc='Images reading'):
        img = cv2.imread(os.path.join(data_config.path_to_data, row['path']), cv2.IMREAD_GRAYSCALE)
        images.append(img)
    images = np.stack(images).astype(np.float32).reshape(len(images), -1)

    # Normalize images
    a, b = data_config.preprocess_params['a'], data_config.preprocess_params['b']
    images_min, images_max = np.min(images), np.max(images)
    images = a + (b - a) * ((images - images_min) / (images_max - images_min))

    return annotation, images


def make_plots():
    """Makes plots with the obtained clusterization data."""
    annotation, images = read_data()
    labels = read_file('labels.npy')
    merge_distances = read_file('merge_distances.pickle')
    distance_diffs = read_file('distance_differences.pickle')

    # Plot the distances between clusters that are merged on each iteration
    fig = px.line(
        y=merge_distances, labels={'y': 'Distance', 'x': 'Iteration'},
        title='Distance between clusters merged on each iteration'
    )
    fig.show()
    fig.write_html('clustering_by_iterations.html')

    # Plot the difference between distances of merged clusters on each iteration
    fig = px.line(
        y=distance_diffs, labels={'y': 'Distance Difference', 'x': 'Iteration'},
        title='Difference in distance between clusters merged on each iteration'
    )
    fig.show()
    fig.write_html('clustering_iterations_differences.html')

    tsne = TSNE(n_components=2, random_state=experiment_config.seed)
    projections = tsne.fit_transform(images)

    # Plot 2D projection of clusters
    fig = px.scatter(
        x=projections[:, 0], y=projections[:, 1], color=labels,
        hover_data={'path': annotation.path, 'set': annotation.set, 'target': annotation.target},
        title='Data labeled with clusters'
    )
    fig.show()
    fig.write_html('data_tsne_projection_clusters_2D.html')

    # Plot 2D projection of data annotated with set type
    fig = px.scatter(
        x=projections[:, 0], y=projections[:, 1], color=annotation.set.values,
        hover_data={'path': annotation.path, 'set': annotation.set, 'target': annotation.target},
        title='Data labeled with set type'
    )
    fig.show()
    fig.write_html('data_tsne_projection_sets_2D.html')

    # Plot 3D projection of data annotated with set type
    tsne = TSNE(n_components=3, random_state=experiment_config.seed)
    projections = tsne.fit_transform(images)
    fig = px.scatter_3d(
        x=projections[:, 0], y=projections[:, 1], z=projections[:, 2], color=annotation.set.values,
        hover_data={'path': annotation.path, 'set': annotation.set, 'target': annotation.target},
        title='Data labeled with set type'
    )
    fig.show()
    fig.write_html('data_tsne_projection_sets_3D.html')


def clusterize_data():
    """Generates clusters using hierarchical agglomerative clustering (HAC) method."""
    _, images = read_data()
    clusterization = AgglomerativeClustering(images, data_config.clusterization)
    clusterization.run()


if __name__ == '__main__':
    clusterize_data()
    make_plots()
