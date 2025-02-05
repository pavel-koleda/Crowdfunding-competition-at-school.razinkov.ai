import sys

import numpy as np
from tqdm import tqdm

from dataset.clustering.distance_metrics import euclidean_distance
from utils.common_functions import write_file
from utils.enums import StoppingCriteria, LinkageMethod


class AgglomerativeClustering:
    """A class for hierarchical agglomerative clustering (HAC) algorithm implementation."""

    def __init__(self, data: np.ndarray, config):
        """Data and measurement methods initialization."""
        self.config = config

        self._init_measures()
        self._init_data(data)

    def _init_measures(self):
        """Initializes measurement methods."""
        self.get_distance = getattr(sys.modules[__name__], f'{self.config.distance_metric}_distance', None)
        if self.get_distance is None:
            raise ValueError(f'Distance method ({self.config.distance_metric}) is not implemented.')

        self.link = getattr(self, f'{self.config.linkage_method.name}_link', None)
        if self.link is None:
            raise ValueError(f'Linkage method ({self.config.linkage_method.name}) is not implemented.')

        self.self_distance_value = np.inf if self.config.linkage_method == LinkageMethod.single else np.nan

    def _init_data(self, data: np.ndarray):
        """Initializes data and pre-computes pairwise distances."""
        self.data = data
        self.current_distance = 0
        self.distances = self.get_distance(data, self.self_distance_value)

    def _init_clusters(self):
        """Initializes clusters for all data samples based on their indices."""
        self.clusters = {i: [i] for i in range(len(self.data))}

    def single_link(self, observed_cluster: list, other_clusters: np.ndarray) -> np.ndarray:
        """Single Linkage method.

        This method calculates pairwise Single Link distances between the observed cluster and all other clusters.

        The Single Link distance between two clusters is computed as follows:
            D(C_i, C_j) = min_{x ∈ C_i, y ∈ C_j} d(x, y),

            where:
                - C_i, C_j - clusters which distance D(C_i, C_j) is measured,
                - x, y - vectors from the given clusters,
                - d - pairwise vector distance metric.

        Args:
            observed_cluster: The vector indices from the observed (newly formed) cluster.
            other_clusters: The other cluster indices.

        Returns:
            np.ndarray: The vector (of size equal to the number of other clusters) of distances between
                            the observed cluster and the others.
        """
        return np.min(self.distances[np.ix_(observed_cluster, other_clusters)], axis=0)

    def complete_link(self, observed_cluster: list, other_clusters: np.ndarray) -> np.ndarray:
        """Complete Linkage method.

        This method calculates pairwise Complete Link distances between the observed cluster and all other clusters.

        The Complete Link distance between two clusters is computed as follows:
            D(C_i, C_j) = max_{x ∈ C_i, y ∈ C_j} d(x, y),

            where:
                - C_i, C_j - clusters which distance D(C_i, C_j) is measured,
                - x, y - vectors from the given clusters,
                - d - pairwise vector distance metric.

        Args:
            observed_cluster: The vector indices from the observed (newly formed) cluster.
            other_clusters: The other cluster indices.

        Returns:
            np.ndarray: The vector (of size equal to the number of other clusters) of distances between
                            the observed cluster and the others.
        """
        return np.nanmax(self.distances[np.ix_(observed_cluster, other_clusters)], axis=0)

    def average_link(self, observed_cluster: list, other_clusters: np.ndarray) -> np.ndarray:
        """Average Linkage method.

        This method calculates pairwise Average Link distances between the observed cluster and all other clusters.

        The Average Link distance between two clusters is computed as follows:
            D(C_i, C_j) =  (1 / (|C_i| * |C_j|)) * Σ((x, y) ∈ C_i × C_j) d(x, y),

            where:
                - C_i, C_j - clusters which distance D(C_i, C_j) is measured,
                - |C_i|, |C_j| - clusters' sizes,
                - x, y - vectors from the given clusters,
                - d - pairwise vector distance metric.

        Args:
            observed_cluster: The vector indices from the observed (newly formed) cluster.
            other_clusters: The other cluster indices.

        Returns:
            np.ndarray: The vector (of size equal to the number of other clusters) of distances between
                            the observed cluster and the others.
        """
        return np.nanmean(self.distances[np.ix_(observed_cluster, other_clusters)], axis=0)

    def stop(self) -> bool:
        """Stopping criteria.

        The method is used to check whether the current clusters merging operation is final w.r.t the criteria.

        'Distance' criteria can be formulated as follows:
            S = min_{i, j} D(C_i, C_j) > T,

            where:
                - S - boolean indicating whether to stop clusters merging,
                - C_i, C_j - clusters which distance D(C_i, C_j) is measured,
                - T - the specified threshold.

        'Clusters number' criteria can be formulated as follows:
            S = |C| <= N,

            where:
                - S - boolean indicating whether to stop clusters merging,
                - |C| - the number of all clusters at the moment,
                - N - the specified threshold.
        """
        if self.config.stopping_criteria == StoppingCriteria.distance:
            return self.current_distance > self.config.stopping_criteria_params.get('distance_th')
        elif self.config.stopping_criteria == StoppingCriteria.clusters_num:
            return len(self.clusters) <= self.config.stopping_criteria_params.get('clusters_num_min')
        else:
            raise Exception(f'Stopping criteria ({self.config.stopping_criteria}) is not implemented.')

    def run(self):
        """Runs clustering algorithm.

        This method represents an iterative process of clusters merging.

        Generates and saves:
            np.ndarray: An array of cluster indices for all data samples in the right order.
            list: A list of distances between two merged clusters at each iteration.
            list: A list of differences between merged clusters distances across iterations.
        """
        self._init_clusters()
        merge_distances, distance_differences, previous_distance = [], [], None
        argmin_func = np.argmin if self.config.linkage_method == LinkageMethod.single else np.nanargmin

        with tqdm() as pbar:
            while not self.stop():
                cluster_1, cluster_2 = np.unravel_index(argmin_func(self.distances), self.distances.shape)
                self.current_distance = self.distances[cluster_1, cluster_2]
                merge_distances.append(self.current_distance)

                if previous_distance is not None:
                    distance_differences.append(self.current_distance - previous_distance)
                previous_distance = self.current_distance

                # Merge clusters
                self.clusters[cluster_1].extend(self.clusters[cluster_2])
                del self.clusters[cluster_2]

                # Update distances
                all_indices = np.array(list(self.clusters.keys()))
                other_clusters = all_indices[all_indices != cluster_1]

                new_distances = self.link(self.clusters[cluster_1], other_clusters)
                self.distances[cluster_1, other_clusters] = new_distances
                self.distances[other_clusters, cluster_1] = new_distances

                self.distances[cluster_2, :] = self.self_distance_value
                self.distances[:, cluster_2] = self.self_distance_value

                pbar.set_description(
                    f'Merged clusters: {cluster_1} and {cluster_2}'
                    f'\tDistance: {self.current_distance}'
                    f'\tTotal clusters num: {len(self.clusters)}'
                )
                pbar.update()

        labels = np.zeros(len(self.data), dtype=int)
        all_points = np.concatenate(list(self.clusters.values()))
        cluster_ids = np.concatenate(
            [[cluster_id] * len(points) for cluster_id, points in enumerate(self.clusters.values())]
        )
        labels[all_points] = cluster_ids

        write_file(labels, 'labels.npy')
        write_file(merge_distances, 'merge_distances.pickle')
        write_file(distance_differences, 'distance_differences.pickle')
