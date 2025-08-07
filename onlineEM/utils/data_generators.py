"""Data generators for Gaussian and Student-t mixture models.

This module provides unified functions for generating synthetic data from
mixture models with controllable parameters for testing and evaluation.
"""

import numpy as np
from scipy import linalg
from scipy.stats import wishart, dirichlet
from typing import Optional, Tuple, Union, Dict, Any


def generate_separated_centers(
    n_clusters: int, n_features: int, separation: float = 3.0, random_state: Optional[int] = None
) -> np.ndarray:
    """Generate well-separated cluster centers.

    Args:
        n_clusters: Number of cluster centers to generate
        n_features: Dimensionality of the centers
        separation: Minimum distance between centers
        random_state: Random seed for reproducibility

    Returns:
        Array of cluster centers, shape (n_clusters, n_features)
    """
    if random_state is not None:
        np.random.seed(random_state)

    centers = []
    max_attempts = 1000

    for i in range(n_clusters):
        attempts = 0
        while attempts < max_attempts:
            # Generate random center
            if n_features == 1:
                center = np.random.uniform(-5, 5, 1)
            elif n_features == 2:
                # Use polar coordinates for better 2D distribution
                if i == 0:
                    center = np.zeros(2)
                else:
                    angle = 2 * np.pi * i / n_clusters + np.random.normal(0, 0.3)
                    radius = separation * (1 + 0.3 * np.random.random())
                    center = radius * np.array([np.cos(angle), np.sin(angle)])
            else:
                # For higher dimensions, use random placement with separation constraint
                center = np.random.normal(0, separation, n_features)

            # Check minimum distance to existing centers
            if len(centers) == 0:
                centers.append(center)
                break

            distances = [np.linalg.norm(center - existing) for existing in centers]
            if min(distances) >= separation:
                centers.append(center)
                break

            attempts += 1

        # If we couldn't find a well-separated center, place it randomly
        if attempts == max_attempts:
            center = np.random.normal(0, separation * 2, n_features)
            centers.append(center)

    return np.array(centers)


def generate_random_covariances(
    n_clusters: int, n_features: int, cluster_std: float = 1.0, random_state: Optional[int] = None
) -> np.ndarray:
    """Generate random positive definite covariance matrices.

    Args:
        n_clusters: Number of covariance matrices to generate
        n_features: Dimensionality of the matrices
        cluster_std: Controls the scale of the covariances
        random_state: Random seed for reproducibility

    Returns:
        Array of covariance matrices, shape (n_clusters, n_features, n_features)
    """
    if random_state is not None:
        np.random.seed(random_state)

    covariances = []

    # Degrees of freedom for Wishart (must be >= n_features)
    df = max(n_features + 1, n_features + 5)

    for i in range(n_clusters):
        # Generate random covariance using Wishart distribution
        scale_matrix = (cluster_std**2) * np.eye(n_features)

        # Add some correlation structure
        if n_features > 1:
            # Generate random correlation matrix
            correlations = np.random.uniform(-0.5, 0.5, (n_features, n_features))
            correlations = (correlations + correlations.T) / 2  # Make symmetric
            np.fill_diagonal(correlations, 1.0)

            # Ensure positive definiteness
            eigenvals, eigenvecs = np.linalg.eigh(correlations)
            eigenvals = np.maximum(eigenvals, 0.1)  # Minimum eigenvalue
            correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            # Scale by cluster_std
            std_diag = cluster_std * (0.5 + np.random.random(n_features))
            scale_matrix = np.outer(std_diag, std_diag) * correlations

        # Generate from Wishart and scale appropriately
        try:
            cov = wishart.rvs(df=df, scale=scale_matrix / df)
        except Exception:
            # Fallback to simple diagonal matrix if Wishart fails
            cov = (cluster_std**2) * np.eye(n_features)

        # Ensure covariance is always 2D array
        if n_features == 1 and np.isscalar(cov):
            cov = np.array([[cov]])
        elif n_features == 1:
            cov = cov.reshape(1, 1)

        covariances.append(cov)

    return np.array(covariances)


def sample_cluster_weights(
    n_clusters: int, concentration: float = 1.0, random_state: Optional[int] = None
) -> np.ndarray:
    """Sample mixing weights from Dirichlet distribution.

    Args:
        n_clusters: Number of mixture components
        concentration: Dirichlet concentration parameter (higher = more uniform)
        random_state: Random seed for reproducibility

    Returns:
        Mixing weights that sum to 1, shape (n_clusters,)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Use symmetric Dirichlet
    alpha = np.full(n_clusters, concentration)
    weights = dirichlet.rvs(alpha)[0]

    return weights


def generate_gaussian_mixture(
    n_clusters: int,
    n_features: int,
    n_samples: int = 500,
    cluster_std: float = 1.0,
    separation: float = 3.0,
    weights: Optional[np.ndarray] = None,
    centers: Optional[np.ndarray] = None,
    covariances: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate samples from a Gaussian mixture model.

    Args:
        n_clusters: Number of mixture components
        n_features: Dimensionality of the data
        n_samples: Number of samples to generate
        cluster_std: Standard deviation within clusters
        separation: Minimum distance between cluster centers
        weights: Optional mixing weights (will be generated if None)
        centers: Optional cluster centers (will be generated if None)
        covariances: Optional covariance matrices (will be generated if None)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X, y, params) where:
        - X: Generated samples, shape (n_samples, n_features)
        - y: True cluster labels, shape (n_samples,)
        - params: Dictionary with true parameters
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate parameters if not provided
    if weights is None:
        weights = sample_cluster_weights(n_clusters, random_state=random_state)

    if centers is None:
        centers = generate_separated_centers(n_clusters, n_features, separation, random_state=random_state)

    if covariances is None:
        covariances = generate_random_covariances(n_clusters, n_features, cluster_std, random_state=random_state)

    # Sample cluster assignments
    cluster_assignments = np.random.choice(n_clusters, size=n_samples, p=weights)

    # Generate samples
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        k = cluster_assignments[i]
        if n_features == 1:
            # Handle 1D case - covariances[k] is 1x1 matrix
            X[i] = np.random.normal(centers[k][0], np.sqrt(covariances[k][0, 0]))
        else:
            X[i] = np.random.multivariate_normal(centers[k], covariances[k])
        y[i] = k

    # Store true parameters
    params = {"weights": weights, "means": centers, "covariances": covariances, "type": "gaussian"}

    return X, y, params


def generate_student_mixture(
    n_clusters: int,
    n_features: int,
    n_samples: int = 500,
    cluster_std: float = 1.0,
    degrees_of_freedom: Union[float, np.ndarray] = 5.0,
    separation: float = 3.0,
    weights: Optional[np.ndarray] = None,
    centers: Optional[np.ndarray] = None,
    covariances: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Generate samples from a Student-t mixture model.

    Args:
        n_clusters: Number of mixture components
        n_features: Dimensionality of the data
        n_samples: Number of samples to generate
        cluster_std: Standard deviation within clusters
        degrees_of_freedom: Degrees of freedom (scalar or array of length n_clusters)
        separation: Minimum distance between cluster centers
        weights: Optional mixing weights (will be generated if None)
        centers: Optional cluster centers (will be generated if None)
        covariances: Optional covariance matrices (will be generated if None)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X, y, params) where:
        - X: Generated samples, shape (n_samples, n_features)
        - y: True cluster labels, shape (n_samples,)
        - params: Dictionary with true parameters
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Handle degrees of freedom parameter
    if np.isscalar(degrees_of_freedom):
        df_array = np.full(n_clusters, float(degrees_of_freedom))
    else:
        df_array = np.asarray(degrees_of_freedom)
        if len(df_array) != n_clusters:
            raise ValueError(f"degrees_of_freedom must be scalar or array of length {n_clusters}")

    # Generate parameters if not provided (reuse Gaussian generation functions)
    if weights is None:
        weights = sample_cluster_weights(n_clusters, random_state=random_state)

    if centers is None:
        centers = generate_separated_centers(n_clusters, n_features, separation, random_state=random_state)

    if covariances is None:
        covariances = generate_random_covariances(n_clusters, n_features, cluster_std, random_state=random_state)

    # Sample cluster assignments
    cluster_assignments = np.random.choice(n_clusters, size=n_samples, p=weights)

    # Generate samples from multivariate t-distribution
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        k = cluster_assignments[i]
        df = df_array[k]

        # Sample from multivariate t-distribution
        # Method: X = μ + Σ^{1/2} * Z * sqrt(ν/U)
        # where Z ~ N(0, I) and U ~ χ²(ν)

        if n_features == 1:
            # Handle 1D case
            z = np.random.normal(0, 1)
            u = np.random.gamma(df / 2.0) / (df / 2.0)  # This gives χ²(ν)/ν
            scale = np.sqrt(df / u)
            X[i] = centers[k][0] + np.sqrt(covariances[k][0, 0]) * z * scale
        else:
            # Sample from standard multivariate normal
            z = np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features))

            # Sample from chi-square and scale
            u = np.random.gamma(df / 2.0) / (df / 2.0)  # This gives χ²(ν)/ν
            scale = np.sqrt(df / u)

            # Transform to desired mean and covariance
            try:
                chol = linalg.cholesky(covariances[k], lower=True)
                X[i] = centers[k] + chol @ (z * scale)
            except linalg.LinAlgError:
                # Fallback if Cholesky fails
                X[i] = centers[k] + np.random.multivariate_normal(np.zeros(n_features), covariances[k])

        y[i] = k

    # Store true parameters
    params = {
        "weights": weights,
        "means": centers,
        "covariances": covariances,
        "degrees_of_freedom": df_array,
        "type": "student",
    }

    return X, y, params


def generate_mix(
    type: str = "gmm",
    n_clusters: int = 3,
    n_features: int = 2,
    n_samples: int = 500,
    cluster_std: float = 1.0,
    degrees_of_freedom: Union[float, np.ndarray] = 5.0,
    separation: float = 3.0,
    random_state: Optional[int] = None,
    return_params: bool = False,
    **kwargs,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Generate samples from Gaussian or Student-t mixture models.

    This is the main unified interface for generating mixture data.

    Args:
        type: Type of mixture ("gmm" for Gaussian, "student" for Student-t)
        n_clusters: Number of mixture components
        n_features: Dimensionality of the data
        n_samples: Number of samples to generate
        cluster_std: Standard deviation within clusters
        degrees_of_freedom: Degrees of freedom for Student-t (ignored for Gaussian)
        separation: Minimum distance between cluster centers
        random_state: Random seed for reproducibility
        return_params: Whether to return the true parameters
        **kwargs: Additional arguments passed to specific generators

    Returns:
        If return_params=False: (X, y)
        If return_params=True: (X, y, params)

        Where:
        - X: Generated samples, shape (n_samples, n_features)
        - y: True cluster labels, shape (n_samples,)
        - params: Dictionary with true parameters (if requested)

    Examples:
        >>> # Generate Gaussian mixture
        >>> X, y = generate_mix("gmm", n_clusters=3, n_features=2)

        >>> # Generate Student-t mixture with heavy tails
        >>> X, y = generate_mix("student", n_clusters=4, degrees_of_freedom=3.0)

        >>> # Get true parameters for evaluation
        >>> X, y, params = generate_mix("gmm", n_clusters=2, return_params=True)
        >>> print(params['weights'])  # True mixing weights
    """
    if type.lower() in ["gmm", "gaussian"]:
        X, y, params = generate_gaussian_mixture(
            n_clusters=n_clusters,
            n_features=n_features,
            n_samples=n_samples,
            cluster_std=cluster_std,
            separation=separation,
            random_state=random_state,
            **kwargs,
        )
    elif type.lower() in ["student", "student-t", "stm"]:
        X, y, params = generate_student_mixture(
            n_clusters=n_clusters,
            n_features=n_features,
            n_samples=n_samples,
            cluster_std=cluster_std,
            degrees_of_freedom=degrees_of_freedom,
            separation=separation,
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown mixture type: {type}. Use 'gmm' or 'student'.")

    if return_params:
        return X, y, params
    else:
        return X, y


def generate_outlier_contaminated_data(
    base_type: str = "gmm", contamination_rate: float = 0.1, outlier_scale: float = 5.0, **kwargs
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, Any]]]:
    """Generate mixture data with additional outlier contamination.

    Args:
        base_type: Base mixture type ("gmm" or "student")
        contamination_rate: Fraction of samples that are outliers (0-1)
        outlier_scale: Scale of outlier distribution
        **kwargs: Arguments passed to generate_mix()

    Returns:
        Same as generate_mix(), but with outliers added
    """
    # Get base data
    if kwargs.get("return_params", False):
        X_clean, y_clean, params = generate_mix(type=base_type, **kwargs)
    else:
        X_clean, y_clean = generate_mix(type=base_type, **kwargs)
        params = None

    n_samples_clean = len(X_clean)
    n_features = X_clean.shape[1]
    n_outliers = int(n_samples_clean * contamination_rate / (1 - contamination_rate))

    # Generate outliers
    random_state = kwargs.get("random_state", None)
    if random_state is not None:
        np.random.seed(random_state + 999)  # Different seed for outliers

    outliers = np.random.normal(0, outlier_scale, (n_outliers, n_features))
    outlier_labels = -1 * np.ones(n_outliers, dtype=int)  # Mark as outliers

    # Combine data
    X = np.vstack([X_clean, outliers])
    y = np.hstack([y_clean, outlier_labels])

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    if params is not None:
        params["contamination_rate"] = contamination_rate
        params["outlier_scale"] = outlier_scale
        return X, y, params
    else:
        return X, y
