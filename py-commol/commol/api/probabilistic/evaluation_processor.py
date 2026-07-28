"""Evaluation processor for handling calibration evaluations.

This module handles collecting, deduplicating, filtering, and clustering
calibration evaluations from multiple optimization runs.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from commol.commol_rs import _commol_rs as commol_rs

if TYPE_CHECKING:
    from commol.commol_rs._commol_rs import CalibrationResultWithHistoryProtocol

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
    max_k : int
        Maximum K tested during automatic silhouette search
    silhouette_sample_size : int | None
        Deterministic sample size for silhouette scoring
    minibatch_kmeans_threshold : int | None
        Use MiniBatchKMeans at or above this evaluation count
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
        max_k: int = 10,
        silhouette_sample_size: int | None = None,
        minibatch_kmeans_threshold: int | None = None,
    ):
        self.deduplication_tolerance = deduplication_tolerance
        self.seed = seed
        self.min_evaluations_for_clustering = min_evaluations_for_clustering
        self.identical_solutions_atol = identical_solutions_atol
        self.silhouette_threshold = silhouette_threshold
        self.silhouette_excellent_threshold = silhouette_excellent_threshold
        self.kmeans_max_iter = kmeans_max_iter
        self.kmeans_algorithm = kmeans_algorithm
        self.max_k = max_k
        self.silhouette_sample_size = silhouette_sample_size
        self.minibatch_kmeans_threshold = minibatch_kmeans_threshold

    def _build_kmeans(self, k: int, n_evaluations: int):
        if (
            self.minibatch_kmeans_threshold is not None
            and k >= 1
            and n_evaluations >= self.minibatch_kmeans_threshold
        ):
            return MiniBatchKMeans(
                n_clusters=k,
                random_state=self.seed,
                n_init="auto",
                max_iter=self.kmeans_max_iter,
            )

        return KMeans(
            n_clusters=k,
            random_state=self.seed,
            n_init="auto",
            max_iter=self.kmeans_max_iter,
            algorithm=self.kmeans_algorithm,
        )

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
            result_evaluations = (
                list(result.evaluations) if hasattr(result, "evaluations") else []
            )
            # Collect ALL evaluations from this run, not just the best one
            # This gives us a diverse set of parameter combinations explored during
            # optimization
            if result_evaluations:
                for eval_obj in result_evaluations:
                    evaluations.append(
                        CalibrationEvaluation(
                            parameters=list(eval_obj.parameters),
                            loss=eval_obj.loss,
                            parameter_names=list(result.parameter_names),
                        )
                    )
                logger.debug(
                    f"Run {idx + 1}: collected {len(result_evaluations)} evaluations, "
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

    @staticmethod
    def filter_by_relative_loss(
        evaluations: list[CalibrationEvaluation],
        max_loss_ratio: float | None,
    ) -> list[CalibrationEvaluation]:
        """Keep evaluations whose loss is plausibly close to the best fit.

        Percentile filtering limits candidate count, but it does not guarantee
        that retained candidates explain the observations. This gate is applied
        after percentile filtering so uncertainty candidates remain calibrated
        solutions rather than merely diverse optimizer states.
        """
        if not evaluations or max_loss_ratio is None:
            return evaluations

        best_loss = min(evaluation.loss for evaluation in evaluations)
        # The epsilon term keeps the gate meaningful when the best loss is zero:
        # a purely multiplicative threshold would then admit only exact zeros.
        threshold = best_loss * max_loss_ratio + np.finfo(float).eps
        return [
            evaluation for evaluation in evaluations if evaluation.loss <= threshold
        ]

    @staticmethod
    def select_prediction_novel_candidates(
        selected_indices: list[int],
        candidate_indices: list[int],
        feature_vectors: np.ndarray,
        max_candidates: int,
    ) -> list[int]:
        """Select tail candidates that maximize prediction-space novelty.

        This is a deterministic farthest-point selector over the same feature
        coordinates used for clustering. It is intended for controlled admission
        of wider-loss candidates: candidates are considered only after they have
        passed an explicit loss gate, and are retained only when they add new
        observed-prediction shapes relative to the already selected core.
        """
        if max_candidates <= 0 or not candidate_indices:
            return []

        vectors = np.asarray(feature_vectors, dtype=float)
        if vectors.ndim != 2:
            raise ValueError("feature_vectors must be a two-dimensional array")
        if vectors.shape[1] == 0:
            raise ValueError("feature_vectors must contain at least one feature")

        all_indices = selected_indices + candidate_indices
        if any(index < 0 or index >= vectors.shape[0] for index in all_indices):
            raise ValueError("selected_indices and candidate_indices must be in range")

        selected = list(dict.fromkeys(selected_indices))
        remaining = list(dict.fromkeys(candidate_indices))
        chosen: list[int] = []

        while remaining and len(chosen) < max_candidates:
            scores = EvaluationProcessor._prediction_novelty_scores(
                vectors, selected, remaining
            )
            next_index = max(
                remaining,
                key=lambda index: (scores[index], -index),
            )
            chosen.append(next_index)
            selected.append(next_index)
            remaining.remove(next_index)

        return chosen

    @staticmethod
    def _prediction_novelty_scores(
        vectors: np.ndarray,
        selected: list[int],
        remaining: list[int],
    ) -> dict[int, float]:
        """Score each remaining candidate by its distance to the selected core.

        With a non-empty core the score is the squared distance to the nearest
        selected member (farthest-point novelty). With no core yet, candidates
        are scored by distance to the remaining candidates' centroid so the
        first pick starts from the prediction-space edge.
        """
        if selected:
            selected_vectors = vectors[selected]
            return {
                index: float(
                    np.min(np.sum((selected_vectors - vectors[index]) ** 2, axis=1))
                )
                for index in remaining
            }
        centroid = np.mean(vectors[remaining], axis=0)
        return {
            index: float(np.sum((vectors[index] - centroid) ** 2))
            for index in remaining
        }

    def find_optimal_k(
        self,
        evaluations: list[CalibrationEvaluation],
        feature_vectors: np.ndarray | None = None,
    ) -> int:
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

        vectors = self._feature_vectors(evaluations, feature_vectors)

        # Check if all solutions are essentially identical (no variance)
        if np.allclose(vectors.std(axis=0), 0, atol=self.identical_solutions_atol):
            logger.info("All solutions are identical, using single cluster")
            return 1

        # Determine range for K to test
        # Use a more conservative upper bound to reduce computation
        # Test K values up to sqrt(n)/2 or 10, whichever is smaller
        min_k = 2
        max_k = min(max(2, int(np.sqrt(n_evaluations)) // 2), self.max_k)

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
                kmeans = self._build_kmeans(k, n_evaluations)
                labels = kmeans.fit_predict(vectors)

                # Silhouette score requires at least 2 clusters with samples
                if len(np.unique(labels)) < 2:
                    silhouette_scores[k] = -1.0
                else:
                    sample_size = (
                        min(self.silhouette_sample_size, n_evaluations)
                        if self.silhouette_sample_size is not None
                        else None
                    )
                    score = silhouette_score(
                        vectors,
                        labels,
                        sample_size=sample_size,
                        random_state=self.seed,
                    )
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
        feature_vectors: np.ndarray | None = None,
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

        vectors = self._feature_vectors(evaluations, feature_vectors)

        kmeans = self._build_kmeans(k, len(evaluations))
        labels = kmeans.fit_predict(vectors)

        return list(labels)

    @staticmethod
    def _feature_vectors(
        evaluations: list[CalibrationEvaluation],
        feature_vectors: np.ndarray | None,
    ) -> np.ndarray:
        """Return validated feature vectors for clustering.

        Clustering runs in parameter space when no vectors are supplied. Callers
        may instead provide standardized transformed prediction vectors to
        cluster by the shape of the predicted series.
        """
        if feature_vectors is None:
            return np.array([evaluation.parameters for evaluation in evaluations])

        vectors = np.asarray(feature_vectors, dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(evaluations):
            raise ValueError(
                "feature_vectors must be a two-dimensional array with one row per "
                "calibration evaluation"
            )
        if vectors.shape[1] == 0:
            raise ValueError("feature_vectors must contain at least one feature")
        return vectors

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
        feature_vectors: np.ndarray | None = None,
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
        feature_vectors : np.ndarray | None
            Optional vectors used by the diversity selector. When supplied,
            these must have one row per evaluation and replace raw parameter
            vectors for representative-space distances.

        Returns
        -------
        list[int]
            Indices of selected representative evaluations
        """
        vectors = self._feature_vectors(evaluations, feature_vectors)

        # The Rust selector uses CalibrationEvaluation.parameters as its
        # diversity coordinates. Substitute the supplied prediction features
        # while retaining loss and row order, so returned indices still refer
        # to the original calibration evaluations.
        rust_evaluations = [
            commol_rs.calibration.CalibrationEvaluation(
                parameters=vector.tolist(),
                loss=e.loss,
                predictions=e.predictions or [],
            )
            for e, vector in zip(evaluations, vectors)
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
