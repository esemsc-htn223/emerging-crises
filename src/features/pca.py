import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from features.base import DimensionalityReduction

class BasePCA(DimensionalityReduction):
    """
    Principal Component Analysis (PCA) implementation for data, applied row-wise.
    """
    def __init__(self, n_components: Union[int, float] = 0.95):
        """
        Initialize the PCA class.
        
        Parameters:
        -----------
        n_components: int or float
            Number of components to keep or variance threshold (0-1).
        """
        super().__init__(n_components)
        self.pca = None
        self.scaler = None
        self.numeric_columns = None

    def fit(self, data: pd.DataFrame, labels: np.array) -> 'BasePCA':
        """
        Fit PCA to the provided data.
        
        Parameters:
        -----------
        data: pd.DataFrame
            DataFrame with time series data.
        labels: np.ndarray
            Array of class labels corresponding to the data.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns

        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data[self.numeric_columns])

        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(scaled_data)
        
        self.logger.info(f"Fitted PCA with {self.n_components} components")
        return self
    
    def transform(self, data: pd.DataFrame, labels: np.array = None) -> pd.DataFrame:
        """
        Transform the data using the fitted PCA model.
        
        Parameters:
        -----------
        data: pd.DataFrame
            DataFrame with time series data to transform.
        labels: np.ndarray
            Array of class labels corresponding to the data.

        Returns:
        --------
        Transformed data as a DataFrame with preserved index.
        """
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if labels is not None and not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if labels is not None and len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        if self.numeric_columns is None:
            raise ValueError("Numeric columns not set. Call fit() first.")
        if not all(col in data.columns for col in self.numeric_columns):
            raise ValueError("Data does not contain all numeric columns used in fitting PCA")
        
        # Standardize the data
        scaled_data = self.scaler.transform(data[self.numeric_columns])
        
        # Transform using PCA
        transformed_data = self.pca.transform(scaled_data)
        
        self.logger.info(f"Transformed data with PCA, shape: {transformed_data.shape}")
        return self.to_dataframe(transformed_data, data.index, "PC")
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in PCA fitting."""
        if self.numeric_columns is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        return self.numeric_columns.tolist()

    def get_component_names(self) -> List[str]:
        """Get component names after transformation."""
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        return [f"PC{i+1}" for i in range(self.pca.n_components_)]
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio for each component."""
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        return self.pca.explained_variance_ratio_

    def plot_explained_variance(self, cumulative: bool = True, title: str = None, save_path: str = None, show: bool = True):
        """Plot explained variance ratio."""
        if self.pca is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        plt.figure(figsize=(5, 4))
        explained_var = self.pca.explained_variance_ratio_
        
        if cumulative:
            cum_var = np.cumsum(explained_var)
            plt.plot(range(1, len(explained_var) + 1), cum_var, 'bo-')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.title(title or 'Cumulative Explained Variance by PCA Components')
            
            if self.n_components > 0.8:
                n_components_80 = np.argmax(cum_var >= 0.8) + 1
                plt.axhline(y=0.8, color='r', linestyle='--')
                plt.text(5, 0.82, f"{n_components_80} components", color='r', fontsize=10, ha='center')
        else:
            plt.bar(range(1, len(explained_var) + 1), explained_var)
            plt.ylabel('Explained Variance Ratio')
            plt.title(title or 'Explained Variance by PCA Components')

        plt.xlabel('Principal Component')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()

    def get_top_components(self, PC: int, n: int = 10) -> List[str]:
        """
        Get top n features contributing to the specified principal component.
        
        Parameters:
        -----------
        PC: int
            Principal component index (1-based).
        n: int
            Number of top features to return.

        Returns:
        --------
        List of feature names contributing to the specified principal component.
        """
        if self.pca is None or self.scaler is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        if PC < 1 or PC > self.pca.n_components_:
            raise ValueError(f"PC must be between 1 and {self.pca.n_components_}")
        
        components = self.pca.components_[PC - 1]
        top_indices = np.argsort(np.abs(components))[-n:]
        top_features = [self.numeric_columns[i] for i in top_indices]
        
        return top_features
    
class ClassSpecificPCA(DimensionalityReduction):
    """
    Class-specific PCA implementation for data. Two transformations are applied:
    - One for the majority class
    - One for the minority class
    """
    def __init__(self, n_components: Union[int, float] = 0.95):
        """
        Initialize the PCA class.
        
        Parameters:
        -----------
        n_components: int or float
            Number of components to keep or variance threshold (0-1).
        """
        super().__init__(n_components)
        self.pca_majority = None
        self.pca_minority = None
        self.scaler = None
        self.numeric_columns = None

    def fit(self, data: pd.DataFrame, labels: np.array) -> 'ClassSpecificPCA':
        """
        Fit PCA to the provided data for both majority and minority classes.
        
        Parameters:
        ----------
        data: pd.DataFrame
            DataFrame with time series data.
        labels: np.ndarray
            Array of class labels corresponding to the data.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns

        self.scaler = StandardScaler()
        scaled_data = self.scaler.fit_transform(data[self.numeric_columns])

        majority_data = scaled_data[labels == 0]
        minority_data = scaled_data[labels == 1]

        self.pca_majority = PCA(n_components=self.n_components)
        self.pca_majority.fit(majority_data)
        
        self.pca_minority = PCA(n_components=self.n_components)
        self.pca_minority.fit(minority_data)
        
        self.logger.info(f"Fitted ClassSpecificPCA with {self.n_components} components for both classes")
        return self
    
    def transform(self, data: pd.DataFrame, labels: np.array = None) -> pd.DataFrame:
        """
        Transform the data using the fitted PCA model for both classes.
        
        Parameters:
        -----------
        data: pd.DataFrame
            DataFrame with time series data to transform.
        labels: np.ndarray
            Array of class labels corresponding to the data.

        Returns:
        --------
        Transformed data as a DataFrame with preserved index.
        """
        if self.pca_majority is None or self.pca_minority is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if labels is not None and not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if labels is not None and len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        if self.numeric_columns is None:
            raise ValueError("Numeric columns not set. Call fit() first.")
        if not all(col in data.columns for col in self.numeric_columns):
            raise ValueError("Data does not contain all numeric columns used in fitting PCA")
        
        scaled_data = self.scaler.transform(data[self.numeric_columns])
        transformed_majority = self.to_dataframe(self.pca_majority.transform(scaled_data), data.index, "Majority_PC")
        transformed_minority = self.to_dataframe(self.pca_minority.transform(scaled_data), data.index, "Minority_PC")
        transformed_data = pd.concat([transformed_majority, transformed_minority], axis=1)

        self.logger.info(f"Transformed data with ClassSpecificPCA, shape: {transformed_data.shape}")
        return transformed_data
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in PCA fitting."""
        if self.numeric_columns is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        return self.numeric_columns.tolist()
    
    def get_component_names(self) -> List[str]:
        """Get component names after transformation."""
        if self.pca_majority is None or self.pca_minority is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        return [f"Majority_PC{i+1}" for i in range(self.pca_majority.n_components_)] + \
               [f"Minority_PC{i+1}" for i in range(self.pca_minority.n_components_)]

    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio for each component."""
        if self.pca_majority is None or self.pca_minority is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        return np.concatenate((self.pca_majority.explained_variance_ratio_,
                               self.pca_minority.explained_variance_ratio_))
    
    def plot_explained_variance(self, cumulative: bool = True, title: str = None, save_path: str = None, show: bool = True):
        """Plot explained variance ratio for both majority and minority classes."""
        if self.pca_majority is None or self.pca_minority is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        # Plot for majority class
        explained_var_maj = self.pca_majority.explained_variance_ratio_
        
        if cumulative:
            cum_var_maj = np.cumsum(explained_var_maj)
            ax1.plot(range(1, len(explained_var_maj) + 1), cum_var_maj, 'ko-')
            ax1.set_ylabel('Cumulative Explained Variance Ratio', fontsize=16)
            ax1.set_title('Label = 0 (Negative Crisis Label)', fontsize=16)

            # Add 80% line for majority class
            n_components_80_maj = np.argmax(cum_var_maj >= 0.8) + 1
            ax1.axhline(y=0.8, color='r', linestyle='--')
            ax1.text(1.5, 0.82, f"{n_components_80_maj} components", 
                    color='r', fontsize=10, ha='left', va='bottom')
        else:
            ax1.bar(range(1, len(explained_var_maj) + 1), explained_var_maj)
            ax1.set_ylabel('Explained Variance Ratio', fontsize=16)
            ax1.set_title('Label = 0 (Negative Crisis Label)', fontsize=16)

        ax1.set_xlabel('Principal Component')
        ax1.grid(True, alpha=0.3)
        
        # Plot for minority class
        explained_var_min = self.pca_minority.explained_variance_ratio_
        
        if cumulative:
            cum_var_min = np.cumsum(explained_var_min)
            ax2.plot(range(1, len(explained_var_min) + 1), cum_var_min, 'ko-')
            ax2.set_title('Label = 1 (Positive Crisis Label)', fontsize=16)

            # Add 80% line for minority class
            n_components_80_min = np.argmax(cum_var_min >= 0.8) + 1
            ax2.axhline(y=0.8, color='r', linestyle='--')
            ax2.text(1.5, 0.82, f"{n_components_80_min} components", 
                    color='r', fontsize=10, ha='left', va='bottom')
        else:
            ax2.bar(range(1, len(explained_var_min) + 1), explained_var_min)
            ax2.set_title('Label = 1 (Positive Crisis Label)', fontsize=16)

        ax2.set_xlabel('Principal Component')
        ax2.grid(True, alpha=0.3)
        
        # Remove y-axis label and ticks from the right subplot
        ax2.tick_params(left=False, labelleft=False)
        
        # Set main title
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    def get_top_components(self, pc_type: str, PC: int, n: int = 10) -> List[str]:
        """
        Get top n features contributing to the specified principal component.

        Parameters:
        -----------
        pc_type: str
            Type of PCA ('majority' or 'minority').
        PC: int
            Principal component index (1-based).
        n: int
            Number of top features to return.

        Returns:
        --------
        List of feature names contributing to the specified principal component.
        """
        if self.pca_majority is None or self.pca_minority is None or self.scaler is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        if PC < 1 or PC > (self.pca_majority.n_components_ + self.pca_minority.n_components_):
            raise ValueError(f"PC must be between 1 and {self.pca_majority.n_components_ + self.pca_minority.n_components_}")

        if pc_type == "majority":
            if PC > self.pca_majority.n_components_:
                raise ValueError(f"PC must be between 1 and {self.pca_majority.n_components_} for majority PCA")
            components = self.pca_majority.components_[PC - 1]
        elif pc_type == "minority":
            if PC > self.pca_minority.n_components_:
                raise ValueError(f"PC must be between 1 and {self.pca_minority.n_components_} for minority PCA")
            components = self.pca_minority.components_[PC - 1]
        else:
            raise ValueError("Invalid pc_type. Must be 'majority' or 'minority'.")
        
        if components.ndim > 1:
            components = components.flatten()
        top_indices = np.argsort(np.abs(components))[-n:]
        top_features = [self.numeric_columns[i] for i in top_indices]

        return top_features
    
    def get_top_rank_components(self, n: int = 40) -> List[str]:
        """
        Get top n features ranked by their importance across all principal components
        for both majority and minority classes.
        
        Parameters:
        -----------
        n: int
            Number of top features to return.
        
        Returns:
        --------
        List of feature names ordered by their total rank across all PCs.
        """
        if self.pca_majority is None or self.pca_minority is None or self.scaler is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        # Get the number of components for each class
        n_maj_components = self.pca_majority.n_components_
        n_min_components = self.pca_minority.n_components_
        n_features = len(self.numeric_columns)
        
        # Initialize rank scores for each feature
        feature_ranks = {feature: 0 for feature in self.numeric_columns}
        
        # Calculate ranks for majority class PCs
        for pc_idx in range(n_maj_components):
            components = self.pca_majority.components_[pc_idx]
            if components.ndim > 1:
                components = components.flatten()
            
            # Get indices sorted by absolute importance (most important first)
            sorted_indices = np.argsort(np.abs(components))[::-1]
            
            # Assign ranks (higher rank = more important)
            for rank, feature_idx in enumerate(sorted_indices):
                feature_name = self.numeric_columns[feature_idx]
                feature_ranks[feature_name] += n_features - rank
        
        # Calculate ranks for minority class PCs
        for pc_idx in range(n_min_components):
            components = self.pca_minority.components_[pc_idx]
            if components.ndim > 1:
                components = components.flatten()
            
            # Get indices sorted by absolute importance (most important first)
            sorted_indices = np.argsort(np.abs(components))[::-1]
            
            # Assign ranks (higher rank = more important)
            for rank, feature_idx in enumerate(sorted_indices):
                feature_name = self.numeric_columns[feature_idx]
                feature_ranks[feature_name] += n_features - rank
        
        # Sort features by total rank (highest first) and return top n
        sorted_features = sorted(feature_ranks.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, _ in sorted_features[:n]]
        
        return top_features
    
    def plot_component_loadings(self, n: int = 40, title: str = None, save_path: str = None, show: bool = True):
        """
        Plot component loadings as a bubble plot showing the importance of top n features
        across majority and minority class principal components.
        
        Parameters:
        -----------
        n: int
            Number of top features to display.
        title: str
            Title for the plot.
        save_path: str
            Path to save the plot.
        show: bool
            Whether to display the plot.
        """
        if self.pca_majority is None or self.pca_minority is None or self.scaler is None:
            raise ValueError("PCA model not fitted. Call fit() first.")
        
        # Get top n components by rank
        top_features = self.get_top_rank_components(n=n)
        
        # Get feature indices
        feature_indices = [list(self.numeric_columns).index(feature) for feature in top_features]
        
        # Prepare data for plotting
        n_maj_components = min(2, self.pca_majority.n_components_)  # Show max 2 PCs
        n_min_components = min(2, self.pca_minority.n_components_)  # Show max 2 PCs
        
        # Create y-axis labels in the requested order: Majority PC1, Minority PC1, Majority PC2, Minority PC2
        y_labels = []
        pc_info = []  # Store (class_type, pc_index) for each row
        
        for i in range(max(n_maj_components, n_min_components)):
            if i < n_maj_components:
                y_labels.append(f'Majority PC{i+1}')
                pc_info.append(('majority', i))
            if i < n_min_components:
                y_labels.append(f'Minority PC{i+1}')
                pc_info.append(('minority', i))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(max(12, n * 0.35), 6))
        
        # Create meshgrid for positions
        x_pos = np.arange(len(top_features))
        y_pos = np.arange(len(y_labels))
        
        # Plot bubbles for each PC
        for j, (class_type, pc_idx) in enumerate(pc_info):
            if class_type == 'majority':
                components = self.pca_majority.components_[pc_idx]
            else:  # minority
                components = self.pca_minority.components_[pc_idx]
            
            if components.ndim > 1:
                components = components.flatten()
            
            # Get loadings for the top features
            loadings = np.abs(components[feature_indices])
            
            # Normalize loadings to percentages (0-100)
            max_loading = np.max(np.abs(components))
            percentages = (loadings / max_loading) * 100
            
            # Create bubble sizes (scale for visibility)
            bubble_sizes = percentages * 3  # Scale factor for visibility
            
            # Plot bubbles
            scatter = ax.scatter(x_pos, [j] * len(x_pos), 
                            s=bubble_sizes, 
                            alpha=0.9, 
                            c=percentages, 
                            cmap='Blues', 
                            vmin=0, vmax=100,
                            edgecolors='black', 
                            linewidth=0.5)
        
        # Customize the plot
        ax.set_xlim(-0.5, len(top_features) - 0.5)
        ax.set_ylim(-0.5, len(y_labels) - 0.5)
        
        # Set labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(top_features, rotation=45, ha='right')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('% Importance', rotation=270, labelpad=20)
        
        # Set title
        if title:
            ax.set_title(title, fontsize=14, pad=20)
        else:
            ax.set_title(f'Component Loadings for Top {n} Features\nAcross Class-Specific Principal Components', 
                        fontsize=14, pad=20)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()