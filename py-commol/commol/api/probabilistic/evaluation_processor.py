"""Evaluation processor for handling calibration evaluations.

This module handles collecting, deduplicating, filtering, and clustering
calibration evaluations from multiple optimization runs.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from commol.commol_rs import commol_rs

if TYPE_CHECKING:
    from commol.commol_rs.commol_rs import CalibrationResultWithHistoryProtocol

from commol.context.probabilistic_calibration import CalibrationEvaluation

logger = logging.getLogger(__name__)


class EvaluationProcessor:
    """Handles processing of calibration evaluations.

    This class is responsible for:
    - Collecting evaluations from calibration results
    - Deduplicating similar evaluations
    - Filtering evaluations by loss percentile
    - Clustering evaluations using K-means
    - Selecting cluster representatives

    Parameters
    ----------
    deduplication_tolerance : float
        Tolerance for parameter deduplication (default: 1e-6)
    seed : int
        Random seed for reproducibility (required, must be 32-bit for sklearn
        compatibility)
    min_evaluations_for_clustering : int
        Minimum number of evaluations required for clustering analysis
    identical_solutions_atol : float
        Absolute tolerance for checking if all solutions are identical
    silhouette_threshold : float
        Silhouette score threshold for determining if clustering is beneficial
    silhouette_excellent_threshold : float
        Early stopping threshold for silhouette score search
    kmeans_max_iter : int
        Maximum iterations for K-means clustering
    kmeans_algorithm : str
        K-means algorithm to use
    """

    def __init__(
        self,
        seed: int,
        deduplication_tolerance: float = 1e-6,
        min_evaluations_for_clustering: int = 10,
        identical_solutions_atol: float = 1e-10,
        silhouette_threshold: float = 0.2,
        silhouette_excellent_threshold: float = 0.7,
        kmeans_max_iter: int = 100,
        kmeans_algorithm: str = "elkan",
    ):
        self.deduplication_tolerance = deduplication_tolerance
        self.seed = seed
        self.min_evaluations_for_clustering = min_evaluations_for_clustering
        self.identical_solutions_atol = identical_solutions_atol
        self.silhouette_threshold = silhouette_threshold
        self.silhouette_excellent_threshold = silhouette_excellent_threshold
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_algorithm = kmeans_algorithm

    def collect_evaluations(
        self, results: list["CalibrationResultWithHistoryProtocol"]
    ) -> list[CalibrationEvaluation]:
        """Collect all parameter evaluations from calibration results.

        Parameters
        ----------
        results : list[CalibrationResultWithHistoryProtocol]
            List of calibration results with evaluation history

        Returns
        -------
        list[CalibrationEvaluation]
            List of all evaluations collected from the results
        """
        evaluations: list[CalibrationEvaluation] = []

        for idx, result in enumerate(results):
            # Collect ALL evaluations from this run, not just the best one
            # This gives us a diverse set of parameter combinations explored during
            # optimization
            if hasattr(result, "evaluations") and len(result.evaluations) > 0:
                for eval_obj in result.evaluations:
                    evaluations.append(
                        CalibrationEvaluation(
                            parameters=list(eval_obj.parameters),
                            loss=eval_obj.loss,
                            parameter_names=list(result.parameter_names),
                        )
                    )
                logger.debug(
                    f"Run {idx + 1}: collected {len(result.evaluations)} evaluations, "
                    f"best loss={result.final_loss:.6f}"
                )
            else:
                # Fallback: if no evaluations history, just use the best result
                evaluations.append(
                    CalibrationEvaluation(
                        parameters=list(result.best_parameters.values()),
                        loss=result.final_loss,
                        parameter_names=list(result.best_parameters.keys()),
                    )
                )
                logger.debug(
                    f"Run {idx + 1}: no evaluation history, using best only: "
                    f"loss={result.final_loss:.6f}"
                )

        return evaluations

    def deduplicate(
        self, evaluations: list[CalibrationEvaluation]
    ) -> list[CalibrationEvaluation]:
        """Remove duplicate evaluations based on parameter similarity using Rust.

        Parameters
        ----------
        evaluations : list[CalibrationEvaluation]
            List of evaluations to deduplicate

        Returns
        -------
        list[CalibrationEvaluation]
            List of unique evaluations
        """
        if not evaluations:
            return []

        # Convert to Rust CalibrationEvaluation objects
        rust_evaluations = [
            commol_rs.calibration.CalibrationEvaluation(
                parameters=e.parameters,
                loss=e.loss,
                predictions=e.predictions or [],
            )
            for e in evaluations
        ]

        # Call Rust deduplication (O(n) average case using spatial hashing)
        unique_rust = commol_rs.calibration.deduplicate_evaluations(
            rust_evaluations, self.deduplication_tolerance
        )

        # Convert back to Python dataclass
        unique: list[CalibrationEvaluation] = []
        param_names = evaluations[0].parameter_names
        for eval_obj in unique_rust:
            unique.append(
                CalibrationEvaluation(
                    parameters=list(eval_obj.parameters),
                    loss=eval_obj.loss,
                    parameter_names=param_names,
                    predictions=list(eval_obj.predictions)
                    if eval_obj.predictions
                    else None,
                )
            )

        return unique

    def filter_by_loss_percentile(
        self,
        evaluations: list[CalibrationEvaluation],
        percentile: float,
    ) -> list[CalibrationEvaluation]:
        """Filter evaluations to keep only the best N% by loss.

        Parameters
        ----------
        evaluations : list[CalibrationEvaluation]
            List of evaluations to filter
        percentile : float
            Fraction (0.0 - 1.0] of best solutions to keep

        Returns
        -------
        list[CalibrationEvaluation]
            Filtered list containing only the best solutions by loss
        """
        if not evaluations or percentile >= 1.0:
            return evaluations

        # Sort by loss (ascending - lower is better)
        sorted_evaluations = sorted(evaluations, key=lambda e: e.loss)

        # Calculate how many to keep
        n_to_keep = max(1, int(len(sorted_evaluations) * percentile))

        # Return the best N%
        return sorted_evaluations[:n_to_keep]

    def find_optimal_k(self, evaluations: list[CalibrationEvaluation]) -> int:
        """Automatically determine optimal number of clusters using silhouette analysis.

        Returns 1 if there's no clear clustering structure (all solutions are similar),
        otherwise returns the optimal K based on silhouette scores.

        Parameters
        ----------
        evaluations : list[CalibrationEvaluation]
            List of evaluations to analyze

        Returns
        -------
        int
            Optimal number of clusters (1 if no clear structure)
        """
        n_evaluations = len(evaluations)

        # If we have very few evaluations, no need to cluster
        if n_evaluations < self.min_evaluations_for_clustering:
            logger.info(
                f"Too few evaluations ({n_evaluations}) for clustering, "
                "using single cluster"
            )
            return 1

        # Extract parameter vectors
        param_vectors = np.array([e.parameters for e in evaluations])

        # Check if all solutions are essentially identical (no variance)
        if np.allclose(
            param_vectors.std(axis=0), 0, atol=self.identical_solutions_atol
        ):
            logger.info("All solutions are identical, using single cluster")
            return 1

        # Determine range for K to test
        # Use a more conservative upper bound to reduce computation
        # Test K values up to sqrt(n)/2 or 10, whichever is smaller
        min_k = 2
        max_k = min(max(2, int(np.sqrt(n_evaluations)) // 2), 10)

        # If we can't test multiple K values, return 1 cluster
        if min_k > max_k or n_evaluations < 2 * min_k:
            logger.info(
                f"Not enough evaluations to test clustering (n={n_evaluations}), "
                "using single cluster"
            )
            return 1

        # Calculate silhouette scores for different K values
        silhouette_scores: dict[int, float] = {}
        k_range = range(min_k, max_k + 1)

        for k in k_range:
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=self.seed,
                    n_init="auto",
                    max_iter=self.kmeans_max_iter,
                    algorithm=self.kmeans_algorithm,
                )
                labels = kmeans.fit_predict(param_vectors)

                # Silhouette score requires at least 2 clusters with samples
                if len(np.unique(labels)) < 2:
                    silhouette_scores[k] = -1.0
                else:
                    score = silhouette_score(param_vectors, labels)
                    silhouette_scores[k] = score

                    # Early stopping if excellent clustering found
                    if score > self.silhouette_excellent_threshold:
                        logger.info(
                            f"Found excellent clustering at k={k} "
                            f"(silhouette={score:.3f}), stopping search"
                        )
                        break

            except Exception as e:
                logger.warning(f"Failed to compute silhouette score for k={k}: {e}")
                silhouette_scores[k] = -1.0

        # Get the best silhouette score and corresponding K
        if not silhouette_scores:
            logger.warning("No valid silhouette scores computed, using single cluster")
            return 1

        best_k = max(silhouette_scores.items(), key=lambda x: x[1])
        optimal_k, best_score = best_k

        if best_score < self.silhouette_threshold:
            logger.info(
                f"Best silhouette score ({best_score:.3f}) below threshold "
                f"({self.silhouette_threshold}), indicating no clear clustering "
                "structure. Using single cluster."
            )
            return 1

        logger.info(
            f"Found {optimal_k} clusters with silhouette score {best_score:.3f}"
        )

        return optimal_k

    def cluster_evaluations(
        self,
        evaluations: list[CalibrationEvaluation],
        k: int,
    ) -> list[int]:
        """Cluster evaluations using K-means.

        If k=1, all evaluations are assigned to a single cluster.

        Parameters
        ----------
        evaluations : list[CalibrationEvaluation]
            List of evaluations to cluster
        k : int
            Number of clusters

        Returns
        -------
        list[int]
            List of cluster labels (one per evaluation)
        """
        if k == 1:
            # Single cluster - no need for K-means
            logger.info("Single cluster: all evaluations grouped together")
            return [0] * len(evaluations)

        param_vectors = np.array([e.parameters for e in evaluations])

        kmeans = KMeans(n_clusters=k, random_state=self.seed, n_init="auto")
        labels = kmeans.fit_predict(param_vectors)

        return list(labels)

    def select_representatives(
        self,
        evaluations: list[CalibrationEvaluation],
        cluster_labels: list[int],
        max_representatives: int,
        elite_fraction: float,
        strategy: str,
        selection_method: str,
        quality_temperature: float,
        k_neighbors_min: int,
        k_neighbors_max: int,
        sparsity_weight: float,
        stratum_fit_weight: float,
    ) -> list[int]:
        """Select representative evaluations from clusters using Rust.

        Parameters
        ----------
        evaluations : list[CalibrationEvaluation]
            List of all evaluations
        cluster_labels : list[int]
            Cluster assignment for each evaluation
        max_representatives : int
            Maximum total representatives to select
        elite_fraction : float
            Fraction of best solutions to always include (0.0-1.0)
        strategy : str
            Distribution strategy ("proportional" or "equal")
        selection_method : str
            Diversity method ("crowding_distance", "maximin_distance", or
            "latin_hypercube")
        quality_temperature : float
            Temperature for quality weighting in maximin method
        k_neighbors_min : int
            Minimum k for k-nearest neighbors density estimation
        k_neighbors_max : int
            Maximum k for k-nearest neighbors density estimation
        sparsity_weight : float
            Exponential weight for sparsity in density-aware selection
        stratum_fit_weight : float
            Weight for stratum fit quality vs diversity in Latin hypercube

        Returns
        -------
        list[int]
            Indices of selected representative evaluations
        """
        # Convert to Rust types
        rust_evaluations = [
            commol_rs.calibration.CalibrationEvaluation(
                parameters=e.parameters,
                loss=e.loss,
                predictions=e.predictions or [],
            )
            for e in evaluations
        ]

        return commol_rs.calibration.select_cluster_representatives(
            rust_evaluations,
            cluster_labels,
            max_representatives,
            elite_fraction,
            strategy,
            selection_method,
            quality_temperature,
            self.seed,
            k_neighbors_min,
            k_neighbors_max,
            sparsity_weight,
            stratum_fit_weight,
        )
