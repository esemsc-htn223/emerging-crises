import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from features.base import DimensionalityReduction
from data.log_utilities import setup_logging

class BaseTSNE(DimensionalityReduction):
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation for data.
    
    t-SNE is particularly good for visualization of high-dimensional data in 2D or 3D space.
    Note: t-SNE is primarily a visualization technique and doesn't support inverse transforms.
    """
    
    def __init__(self, n_components: int = 2, perplexity: float = 30.0, 
                 learning_rate: Union[float, str] = 'auto', max_iter: int = 1000,
                 init: str = 'pca', random_state: int = 42):
        """
        Initialize the t-SNE class.
        
        Args:
            n_components: Number of components (dimensions) for the embedding (typically 2 or 3).
            perplexity: The perplexity parameter.
            learning_rate: Learning rate for optimization ('auto' or float).
            max_iter: Maximum number of iterations for optimization.
            init: Initialization method ('pca', 'random').
            random_state: Random seed for reproducibility.
        """
        # Override initialization and validation
        self.n_components = n_components
        self.logger = setup_logging()
        self._validate_tsne_components()
        
        # t-SNE specific parameters
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        
        # Model attributes
        self.tsne = None
        self.scaler = None
        self.numeric_columns = None
        self.fitted_data = None  # Store fitted data since t-SNE doesn't support new data transforms
        
    def _validate_tsne_components(self):
        """Validate n_components parameter for t-SNE."""
        if not isinstance(self.n_components, int):
            raise TypeError("n_components must be an integer for t-SNE")
        if self.n_components <= 0 or self.n_components > 3:
            raise ValueError("n_components must be 1, 2, or 3 for t-SNE visualization")
    
    def fit(self, data: pd.DataFrame, labels: np.array) -> 'BaseTSNE':
        """
        Fit t-SNE to the provided data.
        
        Args:
            data: DataFrame with time series data.
            labels: Array of class labels corresponding to the data.
        
        Returns:
            Self for method chaining.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(self.numeric_columns) == 0:
            raise ValueError("No numeric columns found in the data")
        
        # Standardize the data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data[self.numeric_columns])
        
        # Validate perplexity
        if self.perplexity >= len(data):
            self.logger.warning(f"Perplexity {self.perplexity} is >= number of samples {len(data)}. "
                                f"Setting perplexity to {min(30, len(data)-1)}")
            self.perplexity = min(30, len(data) - 1)
        
        # Fit t-SNE
        self.tsne = TSNE(
            n_components=self.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            init=self.init,
            random_state=self.random_state,
        )
        
        # Fit and transform the data (t-SNE doesn't separate fit/transform)
        self.fitted_data = self.tsne.fit_transform(scaled_data)
        
        self.logger.info(f"Fitted t-SNE with {self.n_components} components, "
                        f"perplexity={self.perplexity}, max_iter={self.max_iter}")
        return self
    
    def transform(self, data: pd.DataFrame, labels: np.array = None) -> pd.DataFrame:
        """
        Transform the data using the fitted t-SNE model.
        
        Note: t-SNE cannot transform new data. This method only works with the same
        data used for fitting and returns the stored embedding.
        
        Args:
            data: DataFrame with time series data to transform.
            labels: Array of class labels corresponding to the data (optional).
            
        Returns:
            Transformed data as a DataFrame with preserved index.
        """
        if self.tsne is None or self.fitted_data is None:
            raise ValueError("t-SNE model not fitted. Call fit() first.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if labels is not None and not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if labels is not None and len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        
        # Check if this is the same data used for fitting
        if len(data) != len(self.fitted_data):
            raise ValueError("t-SNE cannot transform new data. Only the original fitted data can be transformed.")
        
        self.logger.info(f"Returning t-SNE embedding with shape: {self.fitted_data.shape}")
        return self.to_dataframe(self.fitted_data, data.index, "tSNE")
    
    def fit_transform(self, data: pd.DataFrame, labels: np.array) -> pd.DataFrame:
        """
        Fit t-SNE and transform the data in one step.
        
        Args:
            data: DataFrame with time series data.
            labels: Array of class labels corresponding to the data.
            
        Returns:
            Transformed data as a DataFrame with preserved index.
        """
        self.fit(data, labels)
        return self.transform(data, labels)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in t-SNE fitting."""
        if self.numeric_columns is None:
            raise ValueError("t-SNE model not fitted. Call fit() first.")
        return self.numeric_columns.tolist()
    
    def get_component_names(self) -> List[str]:
        """Get component names after transformation."""
        if self.tsne is None:
            raise ValueError("t-SNE model not fitted. Call fit() first.")
        return [f"tSNE{i+1}" for i in range(self.n_components)]
    
    def get_kl_divergence(self) -> float:
        """
        Get the Kullback-Leibler divergence of the final embedding.
        Lower values indicate better preservation of local structure.
        """
        if self.tsne is None:
            raise ValueError("t-SNE model not fitted. Call fit() first.")
        return self.tsne.kl_divergence_
    
    def plot_perplexity_analysis(self, data: pd.DataFrame, labels: np.array,
                                perplexity_range: List[float] = [5, 10, 20, 30, 50],
                                save_path: str = None, show: bool = True):
        """
        Plot t-SNE embeddings for different perplexity values to help choose optimal perplexity.
        
        Args:
            data: DataFrame with time series data.
            labels: Array of class labels.
            perplexity_range: List of perplexity values to test.
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
        """
        n_plots = len(perplexity_range)
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        if n_plots == 1:
            axes = [axes]
        
        original_perplexity = self.perplexity
        
        for i, perp in enumerate(perplexity_range):
            # Create temporary t-SNE with different perplexity
            temp_tsne = BaseTSNE(n_components=2, perplexity=perp, 
                               random_state=self.random_state, max_iter=500)
            
            # Fit and transform
            embedding = temp_tsne.fit_transform(data, labels)
            
            # Plot
            if labels is not None:
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    axes[i].scatter(embedding.iloc[mask, 0], embedding.iloc[mask, 1], 
                                  c=[colors[j]], label=f'Label {label}', alpha=0.7, s=20)
                axes[i].legend()
            else:
                axes[i].scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], alpha=0.7, s=20)
            
            axes[i].set_title(f'Perplexity = {perp}\nKL = {temp_tsne.get_kl_divergence():.3f}')
            axes[i].set_xlabel('tSNE1')
            axes[i].set_ylabel('tSNE2')
            axes[i].grid(True, alpha=0.3)
        
        # Restore original perplexity
        self.perplexity = original_perplexity
        
        plt.suptitle('t-SNE Perplexity Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()