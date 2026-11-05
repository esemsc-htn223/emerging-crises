import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, List
import umap
from sklearn.preprocessing import StandardScaler

from features.base import DimensionalityReduction
from features.pca import BasePCA
from data.log_utilities import setup_logging

class BaseUMAP(DimensionalityReduction):
    """
    Uniform Manifold Approximation and Projection (UMAP) implementation for data.
    
    UMAP is excellent for both visualization and general dimensionality reduction,
    preserving both local and global structure better than t-SNE.
    Unlike t-SNE, UMAP supports transforming new data points.
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 15, 
                 min_dist: float = 0.1, metric: str = 'euclidean',
                 learning_rate: float = 1.0, n_epochs: Optional[int] = None,
                 pca_preprocess: bool = True, pca_components: Union[int, float] = 0.9,
                 supervised: bool = False,
                 random_state: int = 42):
        """
        Initialize the UMAP class.
        
        Args:
            n_components: Number of components (dimensions) for the embedding.
            n_neighbors: Number of neighbors to consider for local structure (2-100).
            min_dist: Minimum distance between points in the embedding (0.0-1.0).
            metric: Distance metric to use ('euclidean', 'manhattan', etc.).
            learning_rate: Learning rate for optimization.
            n_epochs: Number of training epochs (None for auto).
            random_state: Random seed for reproducibility.
        """
        # Override initialization and validation
        self.n_components = n_components
        self.logger = setup_logging()
        self._validate_umap_components()
        
        # UMAP specific parameters
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.pca_preprocess = pca_preprocess
        self.pca_components = pca_components
        self.supervised = supervised

        # Model attributes
        self.umap_model = None
        self.pca_model = None
        self.scaler = None
        self.numeric_columns = None
        self.original_numeric_columns = None
        
    def _validate_umap_components(self):
        """Validate n_components parameter for UMAP."""
        if not isinstance(self.n_components, int):
            raise TypeError("n_components must be an integer for UMAP")
        if self.n_components <= 0:
            raise ValueError("n_components must be positive for UMAP")
    
    def fit(self, data: pd.DataFrame, labels: np.array) -> 'BaseUMAP':
        """
        Fit UMAP to the provided data.
        
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

        self.original_numeric_columns = data.select_dtypes(include=[np.number]).columns

        if len(self.original_numeric_columns) == 0:
            raise ValueError("No numeric columns found in the data")
        
        # Standardize the data
        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data[self.original_numeric_columns])
        
        # If PCA preprocessing is enabled, apply PCA first
        if self.pca_preprocess:
            self.pca_model = BasePCA(n_components=self.pca_components)
            
            # Create DataFrame for BasePCA
            scaled_df = pd.DataFrame(scaled_data,
                                     columns=self.original_numeric_columns,
                                     index=data.index)
            
            # Fit and transform with PCA
            pca_result = self.pca_model.fit(scaled_df, labels).transform(scaled_df, labels)
            final_data = pca_result.values  # Convert to numpy array for UMAP
            self.numeric_columns = self.pca_model.get_component_names()
            
            explained_var = self.pca_model.get_explained_variance_ratio().sum()
            self.logger.info(f"Applied PCA preprocessing: {len(self.numeric_columns)} → "
                           f"{pca_result.shape[1]} components (explained variance: {explained_var:.3f})")
        else:
            final_data = scaled_data
            self.numeric_columns = self.original_numeric_columns.tolist()

        # Validate n_neighbors
        if self.n_neighbors >= len(data):
            self.logger.warning(f"n_neighbors {self.n_neighbors} is >= number of samples {len(data)}. "
                                f"Setting n_neighbors to {len(data)-1}")
            self.n_neighbors = len(data) - 1
        
        # Fit UMAP
        self.umap_model = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            verbose=False,
            n_jobs=1
        )
        
        if self.supervised and labels is not None:
            self.umap_model.fit(final_data, y=labels)
            self.logger.info(f"Fitted UMAP in supervised mode with labels")
        else:
            self.umap_model.fit(final_data)
            self.logger.info(f"Fitted UMAP in unsupervised mode")
        
        self.logger.info(f"Fitted UMAP with {self.n_components} components, "
                        f"n_neighbors={self.n_neighbors}, min_dist={self.min_dist}")
        return self
    
    def transform(self, data: pd.DataFrame, labels: np.array = None) -> pd.DataFrame:
        """
        Transform the data using the fitted UMAP model.
        
        Args:
            data: DataFrame with time series data to transform.
            labels: Array of class labels corresponding to the data (optional).
            
        Returns:
            Transformed data as a DataFrame with preserved index.
        """
        if self.umap_model is None:
            raise ValueError("UMAP model not fitted. Call fit() first.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if labels is not None and not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if labels is not None and len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        if self.original_numeric_columns is None:
            raise ValueError("Numeric columns not set. Call fit() first.")
        if not all(col in data.columns for col in self.original_numeric_columns):
            raise ValueError("Data does not contain all numeric columns used in fitting UMAP")
        
        # Standardize the data
        scaled_data = self.scaler.transform(data[self.original_numeric_columns])
        
        if self.pca_preprocess and self.pca_model is not None:
            # Apply PCA transformation
            scaled_df = pd.DataFrame(scaled_data, 
                                     columns=self.original_numeric_columns, 
                                     index=data.index)
            pca_result = self.pca_model.transform(scaled_df, labels)
            final_data = pca_result.values
        else:
            final_data = scaled_data
        
        # Transform using UMAP
        transformed_data = self.umap_model.transform(final_data)

        self.logger.info(f"Transformed data with UMAP, shape: {transformed_data.shape}")
        return self.to_dataframe(transformed_data, data.index, "UMAP")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in UMAP fitting."""
        if self.numeric_columns is None:
            raise ValueError("UMAP model not fitted. Call fit() first.")
        return self.numeric_columns.tolist()
    
    def get_component_names(self) -> List[str]:
        """Get component names after transformation."""
        if self.umap_model is None:
            raise ValueError("UMAP model not fitted. Call fit() first.")
        return [f"UMAP{i+1}" for i in range(self.n_components)]
    
    def plot_parameter_analysis(self, data: pd.DataFrame, labels: np.array,
                                n_neighbors_range: List[int] = [5, 15, 30, 50],
                                min_dist_range: List[float] = [0.01, 0.1, 0.5, 0.9],
                                save_path: str = None, show: bool = True):
        """
        Plot UMAP embeddings for different parameter combinations to help choose optimal parameters.
        
        Args:
            data: DataFrame with time series data.
            labels: Array of class labels.
            n_neighbors_range: List of n_neighbors values to test.
            min_dist_range: List of min_dist values to test.
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
        """
        fig, axes = plt.subplots(len(min_dist_range), len(n_neighbors_range), 
                               figsize=(4*len(n_neighbors_range), 4*len(min_dist_range)))
        
        if len(min_dist_range) == 1:
            axes = axes.reshape(1, -1)
        if len(n_neighbors_range) == 1:
            axes = axes.reshape(-1, 1)
        
        for i, min_dist_val in enumerate(min_dist_range):
            for j, n_neighbors_val in enumerate(n_neighbors_range):
                # Create temporary UMAP with different parameters
                temp_umap = BaseUMAP(
                    n_components=2, 
                    n_neighbors=n_neighbors_val,
                    min_dist=min_dist_val,
                    random_state=self.random_state,
                    n_epochs=200,
                    # other parameters remain the same
                    metric=self.metric,
                    learning_rate=self.learning_rate,
                    pca_preprocess=self.pca_preprocess,
                    pca_components=self.pca_components,
                    supervised=self.supervised
                )
                
                # Fit and transform
                embedding = temp_umap.fit(data, labels).transform(data, labels)
                
                # Plot
                ax = axes[i, j] if len(min_dist_range) > 1 else axes[j]
                
                if labels is not None:
                    unique_labels = np.unique(labels)
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                    
                    for k, label in enumerate(unique_labels):
                        mask = labels == label
                        ax.scatter(embedding.iloc[mask, 0], embedding.iloc[mask, 1], 
                                 c=[colors[k]], label=f'Label {label}', alpha=0.7, s=20)
                    if i == 0 and j == 0:  # Only show legend once
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    ax.scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], alpha=0.7, s=20)
                
                ax.set_title(f'n_neighbors={n_neighbors_val}\nmin_dist={min_dist_val}')
                ax.set_xlabel('UMAP1')
                ax.set_ylabel('UMAP2')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('UMAP Parameter Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    def plot_metric_comparison(self, data: pd.DataFrame, labels: np.array,
                              metrics: List[str] = ['euclidean', 'manhattan'],
                              save_path: str = None, show: bool = True):
        """
        Compare UMAP embeddings using different distance metrics.
        
        Args:
            data: DataFrame with time series data.
            labels: Array of class labels.
            metrics: List of distance metrics to compare.
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(metrics):
            # Create temporary UMAP with different metric
            temp_umap = BaseUMAP(
                n_components=2,
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                metric=metric_name,
                random_state=self.random_state,
                n_epochs=200,
                # other parameters remain the same
                learning_rate=self.learning_rate,
                pca_preprocess=self.pca_preprocess,
                pca_components=self.pca_components,
                supervised=self.supervised
            )
            
            # Fit and transform
            embedding = temp_umap.fit(data, labels).transform(data, labels)
            
            # Plot
            if labels is not None:
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                
                for j, label in enumerate(unique_labels):
                    mask = labels == label
                    axes[i].scatter(embedding.iloc[mask, 0], embedding.iloc[mask, 1], 
                                  c=[colors[j]], label=f'Label {label}', alpha=0.7, s=20)
                if i == 0:  # Only show legend once
                    axes[i].legend()
            else:
                axes[i].scatter(embedding.iloc[:, 0], embedding.iloc[:, 1], alpha=0.7, s=20)
            
            axes[i].set_title(f'Metric: {metric_name}')
            axes[i].set_xlabel('UMAP1')
            axes[i].set_ylabel('UMAP2')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('UMAP Distance Metric Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    def get_graph_properties(self) -> dict:
        """
        Get properties of the constructed neighborhood graph.
        
        Returns:
            Dictionary with graph properties.
        """
        if self.umap_model is None:
            raise ValueError("UMAP model not fitted. Call fit() first.")
        
        # Access the internal graph structure
        graph = self.umap_model.graph_
        
        properties = {
            'n_vertices': graph.shape[0],
            'n_edges': graph.nnz,
            'edge_density': graph.nnz / (graph.shape[0] ** 2),
            'graph_shape': graph.shape
        }
        return properties