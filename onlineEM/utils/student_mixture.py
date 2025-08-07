"""Student t-mixture model with scikit-learn compatible interface using NumPy.

This module implements a Student t-mixture model using pure NumPy for efficient
computation. The interface mirrors scikit-learn's GaussianMixture for
easy drop-in replacement.
"""

import numpy as np
from scipy import linalg
from scipy.special import gammaln
from sklearn.mixture import GaussianMixture
from typing import Optional


class StudentMixture:
    """Student t-mixture model with scikit-learn compatible interface.

    This class implements a mixture of multivariate Student t-distributions
    using the EM algorithm. The interface mirrors scikit-learn's GaussianMixture
    for easy drop-in replacement.

    Args:
        n_components: Number of mixture components
        covariance_type: Type of covariance ('full', 'diag', 'tied', 'spherical')
                        Currently only 'full' is implemented
        tol: Convergence threshold
        max_iter: Maximum number of EM iterations
        n_init: Number of initializations to try
        init_params: Method for parameter initialization ('kmeans', 'random')
        random_state: Random seed for reproducibility
        reg_covar: Regularization added to covariance diagonal
        degrees_of_freedom: Initial degrees of freedom (can be scalar or array)
    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        tol: float = 1e-3,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: Optional[int] = None,
        reg_covar: float = 1e-6,
        degrees_of_freedom: float = 5.0,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state
        self.reg_covar = reg_covar
        self.initial_df = degrees_of_freedom

        if covariance_type != "full":
            raise NotImplementedError(
                f"covariance_type='{covariance_type}' not implemented. Currently only 'full' covariance is supported."
            )

        self._fitted = False

    def _multivariate_t_logpdf(self, X, mu, sigma, df):
        """Compute log-probability density of multivariate Student t-distribution.

        Args:
            X: Data points, shape (n_samples, n_features)
            mu: Mean vector, shape (n_features,)
            sigma: Covariance matrix, shape (n_features, n_features)
            df: Degrees of freedom

        Returns:
            Log probability densities, shape (n_samples,)
        """
        n_samples, d = X.shape

        # Compute Mahalanobis distance
        diff = X - mu
        try:
            sigma_chol = linalg.cholesky(sigma, lower=True)
            solved = linalg.solve_triangular(sigma_chol, diff.T, lower=True)
            mahal_dist = np.sum(solved**2, axis=0)
            log_det = 2 * np.sum(np.log(np.diag(sigma_chol)))
        except linalg.LinAlgError:
            # Fallback to eigendecomposition if Cholesky fails
            sigma += self.reg_covar * np.eye(d)
            sigma_inv = linalg.inv(sigma)
            mahal_dist = np.sum((diff @ sigma_inv) * diff, axis=1)
            log_det = np.linalg.slogdet(sigma)[1]

        # Log normalization constant
        log_norm = gammaln((df + d) / 2.0) - gammaln(df / 2.0) - (d / 2.0) * np.log(df * np.pi) - 0.5 * log_det

        # Log probability
        log_prob = log_norm - ((df + d) / 2.0) * np.log(1.0 + mahal_dist / df)

        return log_prob

    def _e_step(self, X):
        """E-step of Student t-mixture EM algorithm.

        Args:
            X: Data points, shape (n_samples, n_features)

        Returns:
            responsibilities: shape (n_samples, n_components)
            latent_weights: shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape

        # Compute log probabilities for each component
        log_probs = np.zeros((n_samples, self.n_components))
        latent_weights = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_probs[:, k] = self._multivariate_t_logpdf(
                X, self.means_[k], self.covariances_[k], self.degrees_of_freedom_[k]
            )

            # Compute latent weights (specific to Student t-distribution)
            diff = X - self.means_[k]
            try:
                sigma_inv = linalg.inv(self.covariances_[k])
                mahal_dist = np.sum((diff @ sigma_inv) * diff, axis=1)
            except linalg.LinAlgError:
                # Add regularization if matrix is singular
                sigma_reg = self.covariances_[k] + self.reg_covar * np.eye(n_features)
                sigma_inv = linalg.inv(sigma_reg)
                mahal_dist = np.sum((diff @ sigma_inv) * diff, axis=1)

            df = self.degrees_of_freedom_[k]
            latent_weights[:, k] = (df + n_features) / (df + mahal_dist)

        # Compute responsibilities
        weighted_log_probs = log_probs + np.log(self.weights_)
        log_prob_norm = np.logaddexp.reduce(weighted_log_probs, axis=1)
        responsibilities = np.exp(weighted_log_probs - log_prob_norm[:, np.newaxis])

        return responsibilities, latent_weights

    def _compute_log_likelihood(self, X):
        """Compute log-likelihood without checking if fitted."""
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            log_probs[:, k] = self._multivariate_t_logpdf(
                X, self.means_[k], self.covariances_[k], self.degrees_of_freedom_[k]
            )

        weighted_log_probs = log_probs + np.log(self.weights_)
        return np.sum(np.logaddexp.reduce(weighted_log_probs, axis=1))

    def _update_degrees_of_freedom_ml(
        self, responsibilities: np.ndarray, latent_weights: np.ndarray, current_df: float, n_features: int
    ) -> float:
        """Update degrees of freedom using a robust moment-based approach.

        Since the exact ML estimation for degrees of freedom is notoriously difficult
        and numerically unstable, we use a combination of moment-based methods
        and conservative updates that work reliably in practice.

        Args:
            responsibilities: Responsibilities for this component, shape (n_samples,)
            latent_weights: Latent weights u_nk for this component, shape (n_samples,)
            current_df: Current degrees of freedom value
            n_features: Number of features

        Returns:
            Updated degrees of freedom
        """
        N_k = np.sum(responsibilities)
        if N_k < 1e-8:
            return current_df

        # Method 1: Use the relationship between u statistics and df
        weighted_u_mean = np.sum(responsibilities * latent_weights) / N_k
        weighted_u_var = np.sum(responsibilities * (latent_weights - weighted_u_mean) ** 2) / N_k

        # For the latent variable u in Student-t EM, we have theoretical relationships
        # E[u] should be close to 1, and the variance gives information about df

        # Method 3: Check the range of u values
        u_min = np.min(latent_weights[responsibilities > 1e-8])
        u_max = np.max(latent_weights[responsibilities > 1e-8])
        u_range = u_max - u_min

        # Combine multiple heuristics for a robust update
        new_df = current_df

        # Heuristic 1: If mean u is significantly different from 1, adjust df
        if weighted_u_mean < 0.8:  # u too low, decrease df
            new_df = max(current_df * 0.9, 2.0)
        elif weighted_u_mean > 1.2:  # u too high, increase df
            new_df = min(current_df * 1.1, 50.0)

        # Heuristic 2: Use variance to refine estimate
        if weighted_u_var > 1e-8 and weighted_u_var < 0.5:
            # For large df, Var[u] ≈ 2/(df-4)
            if current_df > 6:
                var_based_df = 4.0 + 2.0 / max(weighted_u_var, 1e-4)
                var_based_df = np.clip(var_based_df, 2.0, 100.0)
                # Blend with current estimate
                new_df = 0.7 * new_df + 0.3 * var_based_df

        # Heuristic 3: If u values have very small range, increase df
        if u_range < 0.1 and current_df < 20:
            new_df = min(new_df * 1.05, 50.0)
        elif u_range > 0.8 and current_df > 3:
            new_df = max(new_df * 0.95, 2.5)

        # Apply conservative bounds and smoothing
        alpha = 0.3  # Learning rate for df updates
        smoothed_df = (1 - alpha) * current_df + alpha * new_df
        final_df = np.clip(smoothed_df, 2.0, 50.0)

        return float(final_df)

    def _m_step(self, X, responsibilities, latent_weights):
        """M-step of Student t-mixture EM algorithm.

        Args:
            X: Data points, shape (n_samples, n_features)
            responsibilities: shape (n_samples, n_components)
            latent_weights: shape (n_samples, n_components)
        """
        n_samples, n_features = X.shape

        # Update weights
        effective_samples = np.sum(responsibilities, axis=0)
        self.weights_ = effective_samples / n_samples

        # Update means
        weighted_responsibilities = responsibilities * latent_weights
        effective_weighted_samples = np.sum(weighted_responsibilities, axis=0)

        for k in range(self.n_components):
            if effective_weighted_samples[k] > 1e-8:
                self.means_[k] = (
                    np.sum(weighted_responsibilities[:, k : k + 1] * X, axis=0) / effective_weighted_samples[k]
                )

        # Update covariances
        for k in range(self.n_components):
            if effective_samples[k] > 1e-8:
                diff = X - self.means_[k]
                weighted_diff = weighted_responsibilities[:, k : k + 1] * diff
                self.covariances_[k] = (diff.T @ weighted_diff) / effective_samples[k]

                # Add regularization
                self.covariances_[k] += self.reg_covar * np.eye(n_features)

        # Update degrees of freedom using ML estimation
        for k in range(self.n_components):
            if effective_samples[k] > 1e-8:
                self.degrees_of_freedom_[k] = self._update_degrees_of_freedom_ml(
                    responsibilities[:, k], latent_weights[:, k], self.degrees_of_freedom_[k], n_features
                )

    def _initialize_parameters(self, X):
        """Initialize parameters using Gaussian mixture model.

        Args:
            X: Training data, shape (n_samples, n_features)
        """
        # Use sklearn's GaussianMixture for initialization
        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
        )
        gmm.fit(X)

        # Initialize parameters
        self.weights_ = gmm.weights_.copy()
        self.means_ = gmm.means_.copy()
        self.covariances_ = gmm.covariances_.copy()

        # Initialize degrees of freedom
        if np.isscalar(self.initial_df):
            self.degrees_of_freedom_ = np.full(self.n_components, self.initial_df)
        else:
            self.degrees_of_freedom_ = np.array(self.initial_df).copy()

    def fit(self, X, y=None):
        """Fit the Student t-mixture model to data.

        Args:
            X: Training data, shape (n_samples, n_features)
            y: Ignored (for sklearn compatibility)

        Returns:
            Self (fitted estimator)
        """
        X = np.asarray(X, dtype=np.float64)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        best_log_likelihood = -np.inf
        best_params = None

        for init in range(self.n_init):
            # Initialize parameters
            self._initialize_parameters(X)

            # EM algorithm
            log_likelihood_prev = -np.inf

            for iteration in range(self.max_iter):
                # E-step
                responsibilities, latent_weights = self._e_step(X)

                # M-step
                self._m_step(X, responsibilities, latent_weights)

                # Compute log-likelihood for convergence check
                log_likelihood = self._compute_log_likelihood(X)

                # Check convergence
                if abs(log_likelihood - log_likelihood_prev) < self.tol:
                    break

                log_likelihood_prev = log_likelihood

            # Keep best initialization
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = {
                    "weights": self.weights_.copy(),
                    "means": self.means_.copy(),
                    "covariances": self.covariances_.copy(),
                    "degrees_of_freedom": self.degrees_of_freedom_.copy(),
                }

        # Set best parameters
        if best_params is not None:
            self.weights_ = best_params["weights"]
            self.means_ = best_params["means"]
            self.covariances_ = best_params["covariances"]
            self.degrees_of_freedom_ = best_params["degrees_of_freedom"]

        self._fitted = True
        return self

    def predict(self, X):
        """Predict component labels for samples.

        Args:
            X: Samples, shape (n_samples, n_features)

        Returns:
            Component labels, shape (n_samples,)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """Predict posterior probabilities for samples.

        Args:
            X: Samples, shape (n_samples, n_features)

        Returns:
            Posterior probabilities, shape (n_samples, n_components)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")

        responsibilities, _ = self._e_step(X)
        return responsibilities

    def score(self, X):
        """Compute log-likelihood of samples.

        Args:
            X: Samples, shape (n_samples, n_features)

        Returns:
            Log-likelihood of samples
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before scoring")

        return self._compute_log_likelihood(X)

    def sample(self, n_samples=1):
        """Generate samples from the fitted model.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tuple of (samples, component_labels)
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before sampling")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Sample component assignments
        component_labels = np.random.choice(self.n_components, size=n_samples, p=self.weights_)

        samples = np.zeros((n_samples, self.means_.shape[1]))

        for i in range(n_samples):
            k = component_labels[i]

            # Sample from multivariate t-distribution
            # This is done by sampling from multivariate normal and scaling by chi-square
            df = self.degrees_of_freedom_[k]

            # Sample from standard multivariate normal
            z = np.random.multivariate_normal(np.zeros(self.means_.shape[1]), np.eye(self.means_.shape[1]))

            # Sample from chi-square and scale
            u = np.random.gamma(df / 2.0) / (df / 2.0)  # This gives chi2/df
            scale = np.sqrt(df / u)

            # Transform to desired mean and covariance
            chol = linalg.cholesky(self.covariances_[k], lower=True)
            samples[i] = self.means_[k] + chol @ (z * scale)

        return samples, component_labels

    def aic(self, X):
        """Compute Akaike Information Criterion."""
        if not self._fitted:
            raise ValueError("Model must be fitted before computing AIC")

        log_likelihood = self.score(X)
        n_features = X.shape[1]
        n_params = (
            self.n_components
            - 1  # weights (one is determined by the others)
            + self.n_components * n_features  # means
            + self.n_components * n_features * (n_features + 1) // 2  # covariances
            + self.n_components  # degrees of freedom
        )
        return -2 * log_likelihood + 2 * n_params

    def bic(self, X):
        """Compute Bayesian Information Criterion."""
        if not self._fitted:
            raise ValueError("Model must be fitted before computing BIC")

        log_likelihood = self.score(X)
        n_samples, n_features = X.shape
        n_params = (
            self.n_components
            - 1  # weights
            + self.n_components * n_features  # means
            + self.n_components * n_features * (n_features + 1) // 2  # covariances
            + self.n_components  # degrees of freedom
        )
        return -2 * log_likelihood + np.log(n_samples) * n_params
