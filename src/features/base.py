import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Union

from data.log_utilities import setup_logging

class DimensionalityReduction(ABC):
    """
    Abstract base class for Dimensionality Reduction implementations.
    
    This class defines the interface for analysis on time series data,
    including methods for fitting, transforming, and visualizing results.
    """
    def __init__(self, n_components: Union[int, float] = 0.95):
        """
        Initialize the Dimensionality Reduction implementation.
        
        Parameters:
        ----------
        n_components : int or float
            Number of components to keep or, for PCA, variance threshold (0-1).
        """
        self.n_components = n_components
        self.logger = setup_logging()
        self._validate_n_components()

    def _validate_n_components(self):
        """Validate n_components parameter."""
        if isinstance(self.n_components, float):
            if not 0 < self.n_components <= 1:
                raise ValueError("When n_components is float, it must be between 0 and 1")
        elif isinstance(self.n_components, int):
            if self.n_components <= 0:
                raise ValueError("When n_components is int, it must be positive")
        else:
            raise TypeError("n_components must be int or float")
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, labels: np.array) -> 'DimensionalityReduction':
        """
        Fit the Dimensionality Reduction model to the provided data.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with time series data.
        labels : np.array
            Array of class labels corresponding to the data.
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame, labels: np.array = None) -> np.ndarray:
        """
        Transform the data using the fitted Dimensionality Reduction model.

        Parameters:
        -----------
        data: pd.DataFrame
            DataFrame with time series data to transform.
        labels: np.array
            Array of class labels corresponding to the data.

        Returns:
        --------
        Transformed data as a DataFrame with preserved index.
        """
        pass
    
    @staticmethod
    def to_dataframe(transformed_data: np.ndarray, index: pd.Index, column_prefix: str) -> pd.DataFrame:
        """
        Convert transformed data to a DataFrame with preserved index.
        
        Parameters:
        -----------
        transformed_data: np.ndarray
            Transformed data.
        index: pd.Index
            Original index of the data.
        column_prefix: str
            Prefix for the column names in the transformed DataFrame.

        Returns:
        --------
            DataFrame with transformed data and preserved index.
        """
        if not isinstance(transformed_data, np.ndarray):
            raise TypeError("Transformed data must be a NumPy array")
        if not isinstance(index, pd.Index):
            raise TypeError("Index must be a pandas Index")
        if not isinstance(column_prefix, str):
            raise TypeError("Column prefix must be a string")
        if len(index) != transformed_data.shape[0]:
            raise ValueError("Index length must match number of rows in transformed data")
        
        if transformed_data.ndim != 2:
            raise ValueError("Transformed data must be 2-dimensional")
        n_components = transformed_data.shape[1]
        columns = [f'{column_prefix}{i+1}' for i in range(n_components)]

        if len(columns) != n_components:
            raise ValueError("Number of columns must match number of components in transformed data")
        
        return pd.DataFrame(transformed_data, index=index, columns=columns)
    
    @staticmethod
    def label_to_colors(labels: Union[pd.Series, pd.DataFrame, list, np.ndarray]) -> Tuple[np.array, List[str]]:
        """
        Convert labels to color codes for plotting.

        Parameters:
        -----------
        labels: Labels as a Series, DataFrame, list, or NumPy array.
            
        Returns:
        --------
        Tuple of color codes and unique label categories.
        """
        if isinstance(labels, pd.Series) or isinstance(labels, pd.DataFrame):
            colors = labels.astype('category').cat.codes
            key = labels.astype('category').cat.categories.tolist()
        elif isinstance(labels, list) or isinstance(labels, np.ndarray):
            colors = pd.Series(labels).astype('category').cat.codes
            key = pd.Series(labels).astype('category').cat.categories.tolist()
        else:
            raise TypeError("Labels must be a pandas Series, DataFrame, list, or NumPy array")
        return colors.to_numpy(), key
    
    def _add_color_legend(self, ax, category_color_map):
        """
        Add a color legend showing the color mapping for categories.
        
        Parameters:
        -----------
        ax: Matplotlib axes object
        category_color_map: Dictionary
            Mapps categories to colors
        """
        if not category_color_map:
            return
            
        # Create legend entries for categories
        legend_elements = []
        for category, color in category_color_map.items():
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7,
                                               label=f'{category}'))
        
        # Add color legend in upper right corner
        color_legend = ax.legend(handles=legend_elements,
                                 loc='upper right',
                                 bbox_to_anchor=(0.0, 1.0))
    
    def plot_1D_feature_space(self, transformed_data: pd.DataFrame, labels: np.array = None,
                              title: str = "1D Feature Space", idx: int = 0,
                              save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot the 1D feature space of the transformed data as horizontal half violin plots.

        Parameters:
        -----------
        transformed_data : pd.DataFrame
            Transformed data
        labels : np.array
            Array of class labels corresponding to the data
        title : str
            Plot title
        idx : int
            Index of component to be visualized
        save_path : str
            Path to save the plot (optional)
        show : bool
            Whether to display the plot
        """
        if transformed_data.shape[1] < 1:
            raise ValueError("Transformed data must have at least 1 dimension for 1D plotting.")
        if idx > transformed_data.shape[1]:
            raise ValueError(f"Component {idx} not found in transformed data.")
        if labels is not None and transformed_data.shape[0] != len(labels):
            raise ValueError("Transformed data and labels must have the same length.")

        # Extract the component data
        component_data = transformed_data.iloc[:, idx]
        
        if labels is not None:
            # Separate data by label
            label_0_data = component_data[labels == 0]
            label_1_data = component_data[labels == 1]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create horizontal violin plots
            parts0 = ax.violinplot([label_0_data.values], positions=[0.6], 
                                vert=False, widths=0.8,
                                showmeans=False, showmedians=False, showextrema=False)
            parts1 = ax.violinplot([label_1_data.values], positions=[0.4], 
                                vert=False, widths=0.8,
                                showmeans=False, showmedians=False, showextrema=False)

            # Set colors
            colors = sns.color_palette("Set1", n_colors=2)

            # Color and modify Label 0 violin (top half only)
            for pc in parts0['bodies']:
                pc.set_facecolor(colors[0])
                pc.set_alpha(0.7)
                # Get the path and modify vertices to keep only top half
                path = pc.get_paths()[0]
                vertices = path.vertices.copy()
                # For horizontal violins, keep only vertices above the center (y=0.6)
                center_y = 0.6
                vertices[:, 1] = np.maximum(vertices[:, 1], center_y)
                path.vertices = vertices
            
            # Color and modify Label 1 violin (bottom half only)  
            for pc in parts1['bodies']:
                pc.set_facecolor(colors[1])
                pc.set_alpha(0.7)
                # Get the path and modify vertices to keep only bottom half
                path = pc.get_paths()[0]
                vertices = path.vertices.copy()
                # Keep only vertices below the center (y=0.4)
                center_y = 0.4
                vertices[:, 1] = np.minimum(vertices[:, 1], center_y)
                path.vertices = vertices
            
            # Set labels and formatting with adjusted positions
            ax.set_yticks([0.4, 0.6])
            ax.set_yticklabels(['Label = 1', 'Label = 0'])
            ax.set_xlabel(f'Component {idx + 1}', fontsize=12)
            
        else:
            # Single horizontal violin plot when no labels
            fig, ax = plt.subplots(figsize=(8, 4))
            
            parts = ax.violinplot([component_data.values], positions=[0.5], 
                                vert=False, widths=0.8,
                                showmeans=False, showmedians=False, showextrema=False)
            
            # Color the single violin
            for pc in parts['bodies']:
                pc.set_facecolor('steelblue')
                pc.set_alpha(0.7)
            
            ax.set_yticks([])
            ax.set_xlabel(f'Component {idx + 1}', fontsize=12)
        
        # Common formatting
        ax.set_title(f"{title} - Component {idx + 1}", fontsize=14)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    def plot_2D_feature_space(self, transformed_data: pd.DataFrame, labels: np.array = None,
                              title: str = "2D Feature Space", color_by: np.array = None, 
                              save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Plot the 2D feature space of the transformed data as separate subplots for each class.
        
        Parameters:
        -----------
        transformed_data : pd.DataFrame
            Transformed data
        labels : np.array
            Array of class labels corresponding to the data
        title : str
            Plot title
        color_by : np.array
            Categories for coloring points (optional)
        save_path : str
            Path to save the plot (optional)
        show : bool
            Whether to display the plot
        """
        if transformed_data.shape[1] < 2:
            raise ValueError("Transformed data must have at least 2 dimensions for 2D plotting.")
        
        # Validate color_by array if provided
        if color_by is not None and len(color_by) != transformed_data.shape[0]:
            raise ValueError("color_by array must have the same length as transformed_data.")

        # Extract component 1 and 2
        pc1 = transformed_data.iloc[:, 0]
        pc2 = transformed_data.iloc[:, 1]
        
        # Set up color palette for color_by categories
        if color_by is not None:
            unique_categories = np.unique(color_by)
            category_colors = sns.color_palette("Set1", n_colors=len(unique_categories))
            category_color_map = dict(zip(unique_categories, category_colors))
        else:
            # Default color when no color_by is provided
            default_color = 'steelblue'
        
        if labels is not None:
            # Create subplots: one for each class
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            
            # Get unique labels
            unique_labels = np.unique(labels)
            
            # Determine axis limits to keep them consistent
            x_min, x_max = pc1.min(), pc1.max()
            y_min, y_max = pc2.min(), pc2.max()
            x_margin = (x_max - x_min) * 0.05
            y_margin = (y_max - y_min) * 0.05
            
            # Plot for each class
            for idx, label in enumerate(sorted(unique_labels)):
                ax = ax1 if int(label) == 0 else ax2
                
                # Filter data for current label
                mask = labels == label
                pc1_label = pc1[mask]
                pc2_label = pc2[mask]
                color_by_label = color_by[mask] if color_by is not None else None
                
                if len(pc1_label) == 0:
                    continue
                
                # Determine colors for points
                if color_by is not None:
                    point_colors = [category_color_map[cat] for cat in color_by_label]
                else:
                    point_colors = default_color
                
                marker = 'x'
                size = 30
                
                # Create scatter plot
                ax.scatter(pc1_label, pc2_label, s=size, 
                           c=point_colors, alpha=0.7, marker=marker,
                           linewidths=1.0)
                
                # Set subplot title and formatting
                ax.set_title(f'Label = {int(label)}', fontsize=14)
                ax.set_xlabel(f"{transformed_data.columns[0]}", fontsize=12)
                if idx == 0:  # Only set ylabel for the first subplot
                    ax.set_ylabel(f"{transformed_data.columns[1]}", fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
            
            # Set main title
            fig.suptitle(title, fontsize=16)
            
        else:
            # Single plot when no labels provided
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Determine colors for points
            if color_by is not None:
                point_colors = [category_color_map[cat] for cat in color_by]
            else:
                point_colors = default_color
                
            # Create scatter plot
            ax.scatter(pc1, pc2, s=20, c=point_colors, alpha=0.7, marker='x',
                       linewidths=0.3)
            
            # Formatting
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(f"{transformed_data.columns[0]}", fontsize=12)
            ax.set_ylabel(f"{transformed_data.columns[1]}", fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Add color legend for color_by categories
        if color_by is not None:
            if labels is not None:
                self._add_color_legend(ax2, category_color_map)
            else:
                self._add_color_legend(ax, category_color_map)

        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    def plot_feature_space(self, transformed_data: pd.DataFrame, labels: np.array = None,
                   title: str = "2D Feature Space", color_by: np.array = None, 
                   highlight_categories: List[str] = None,
                   save_path: Optional[str] = None, show: bool = True) -> Tuple[int, int]:
        """
        Plot the feature space of the transformed data, selecting components with greatest spread.
        
        Parameters:
        -----------
        transformed_data : pd.DataFrame
            Transformed data
        labels : np.array
            Array of class labels corresponding to the data
        title : str
            Plot title
        color_by : np.array
            Categories for coloring points (optional)
        highlight_categories : List[str]
            Specific categories to highlight in color. If None, uses Mahalanobis distance to select top 3.
        save_path : str
            Path to save the plot (optional)
        show : bool
            Whether to display the plot
            
        Returns:
        --------
        Tuple[int, int]
            Indices of the two components that were plotted
        """
        # 1. Select components that show greatest spread in feature space
        idx1, idx2 = choose_components(transformed_data, method='determinant')
        selected_data = transformed_data.iloc[:, [idx1, idx2]].copy()
        
        # Extract component data
        pc1 = selected_data.iloc[:, 0]
        pc2 = selected_data.iloc[:, 1]
        
        # Set up colors
        default_color = 'grey'
        
        if color_by is not None:
            # Determine which categories to highlight
            if highlight_categories is not None:
                # Use manually specified categories
                top_categories = [cat for cat in highlight_categories if cat in np.unique(color_by)]
                if not top_categories:
                    self.logger.warning("None of the specified highlight_categories found in color_by. Using Mahalanobis distance selection.")
                    top_categories = choose_categories(transformed_data, labels, color_by)
            else:
                # Use Mahalanobis distance to select top 3 categories
                top_categories = choose_categories(transformed_data, labels, color_by)
            
            # Create color map only for top categories
            category_colors = sns.color_palette("Set1", n_colors=len(top_categories))
            category_color_map = dict(zip(top_categories, category_colors))
            
            # Determine axis limits
            x_min, x_max = pc1.min(), pc1.max()
            y_min, y_max = pc2.min(), pc2.max()
            x_margin = (x_max - x_min) * 0.05
            y_margin = (y_max - y_min) * 0.05

            if labels is not None:
                # Create two subplots: one for each label
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
                unique_labels = np.unique(labels)
                
                # Plot for each class
                for idx, label in enumerate(sorted(unique_labels)):
                    ax = ax1 if int(label) == 0 else ax2
                    
                    # Filter data for current label
                    mask = labels == label
                    pc1_label = pc1[mask]
                    pc2_label = pc2[mask]
                    color_by_label = color_by[mask]
                    
                    if len(pc1_label) == 0:
                        continue
                    
                    marker = 'x'
                    size = 30
                    
                    # Separate points into colored (top categories) and grey (others)
                    colored_mask = np.isin(color_by_label, top_categories)
                    
                    # Plot grey points first (background)
                    if np.any(~colored_mask):
                        ax.scatter(pc1_label[~colored_mask], pc2_label[~colored_mask], 
                                s=size, c=default_color, alpha=0.3, marker=marker,
                                linewidths=0.5)
                    
                    # Plot colored points (top categories) on top
                    for category in top_categories:
                        cat_mask = color_by_label == category
                        if np.any(cat_mask):
                            ax.scatter(pc1_label[cat_mask], pc2_label[cat_mask], 
                                    s=size, c=[category_color_map[category]], 
                                    alpha=0.8, marker=marker, linewidths=1.0,
                                    label=category)
                    
                    # Set subplot title and formatting
                    if int(label) == 0:
                        text = 'Negative'
                    elif int(label) == 1:
                        text = 'Positive'
                    ax.set_title(f'Label = {int(label)} ({text} Crisis Label)', fontsize=16)
                    if idx == 0:  # Only set ylabel for the first subplot
                        ax.set_ylabel(f"{selected_data.columns[1]}", fontsize=16)
                    ax.set_xlabel(f"{selected_data.columns[0]}", fontsize=16)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(x_min - x_margin, x_max + x_margin)
                    ax.set_ylim(y_min - y_margin, y_max + y_margin)

                    # Add bold axis lines at x=0 and y=0
                    ax.axhline(y=0, color='black', linewidth=1, alpha=0.7)
                    ax.axvline(x=0, color='black', linewidth=1, alpha=0.7)
                    
                    # Add legend only to the second subplot (right side)
                    if idx == 1:
                        handles, labels_legend = ax.get_legend_handles_labels()
                        if handles:
                            legend = ax.legend(handles, labels_legend, loc='lower right', fontsize=14)
                            for handle in legend.legend_handles:
                                handle.set_linewidth(2.0)
                
                # Set main title
                fig.suptitle(title, fontsize=16)
                
            else:
                # Single plot when no labels provided
                fig, ax = plt.subplots(figsize=(10, 6))
                
                marker = 'x'
                size = 30
                
                # Separate points into colored (top categories) and grey (others)
                colored_mask = np.isin(color_by, top_categories)
                
                # Plot grey points first (background)
                if np.any(~colored_mask):
                    ax.scatter(pc1[~colored_mask], pc2[~colored_mask], 
                            s=size, c=default_color, alpha=0.3, marker=marker,
                            linewidths=0.5)
                
                # Plot colored points (top categories) on top
                for category in top_categories:
                    cat_mask = color_by == category
                    if np.any(cat_mask):
                        ax.scatter(pc1[cat_mask], pc2[cat_mask], 
                                s=size, c=[category_color_map[category]], 
                                alpha=0.8, marker=marker, linewidths=1.0,
                                label=category)
                
                # Formatting
                ax.set_title(title, fontsize=14)
                ax.set_xlabel(f"{selected_data.columns[0]}", fontsize=16)
                ax.set_ylabel(f"{selected_data.columns[1]}", fontsize=16)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(x_min - x_margin, x_max + x_margin)
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                
                # Add bold axis lines at x=0 and y=0
                ax.axhline(y=0, color='black', linewidth=1, alpha=0.7)
                ax.axvline(x=0, color='black', linewidth=1, alpha=0.7)
            
                # Add legend
                handles, labels_legend = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels_legend, loc='upper right', fontsize=16)
                    
            plt.tight_layout()
            
        else:
            # Fall back to the standard 2D plot when no color_by is provided
            self.plot_2D_feature_space(
                transformed_data=selected_data,
                labels=labels,
                title=title,
                color_by=color_by,
                save_path=save_path,
                show=show
            )
            return idx1, idx2
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
            
        return idx1, idx2

def choose_components(transformed_data: pd.DataFrame, 
                      method: str = 'determinant') -> Tuple[int, int]:
    """
    Choose components based on different spread metrics.
    
    Parameters:
    -----------
    transformed_data : pd.DataFrame
        Transformed data with components as columns
    method : str
        Method for calculating spread: 'determinant', 'trace', 'frobenius', 'range'
        
    Returns:
    --------
    Tuple[int, int]
        Indices of the two components that maximize spread
    """
    if transformed_data.shape[1] < 2:
        raise ValueError("Transformed data must have at least 2 components")
    
    n_components = transformed_data.shape[1]
    max_spread = -1
    best_pair = (0, 1)
    
    for i in range(n_components):
        for j in range(i + 1, n_components):
            comp_i = transformed_data.iloc[:, i].values
            comp_j = transformed_data.iloc[:, j].values
            data_2d = np.column_stack([comp_i, comp_j])
            
            if method == 'determinant':
                # Covariance matrix determinant (area of confidence ellipse)
                cov_matrix = np.cov(data_2d.T)
                spread = np.linalg.det(cov_matrix)
                
            elif method == 'trace':
                # Sum of variances (total variance)
                spread = np.var(comp_i) + np.var(comp_j)
                
            elif method == 'frobenius':
                # Frobenius norm of covariance matrix
                cov_matrix = np.cov(data_2d.T)
                spread = np.linalg.norm(cov_matrix, 'fro')
                
            elif method == 'range':
                # Product of ranges (bounding box area)
                range_i = np.ptp(comp_i)  # peak-to-peak (max - min)
                range_j = np.ptp(comp_j)
                spread = range_i * range_j
                
            else:
                raise ValueError("Method must be 'determinant', 'trace', 'frobenius', or 'range'")
            
            if spread > max_spread:
                max_spread = spread
                best_pair = (i, j)
    return best_pair


def choose_categories(transformed_data: pd.DataFrame,
                      labels: np.array,
                      color_by: np.array):
    
    def mahalanobis_distance(X_pos, X_neg):
        centroid_pos = np.mean(X_pos, axis=0)
        centroid_neg = np.mean(X_neg, axis=0)
        cov = np.cov(np.vstack([X_pos, X_neg]).T)
        cov_inv = np.linalg.pinv(cov)
        diff = centroid_pos - centroid_neg
        return np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))

    # 1. Get unique categories from color_by
    unique_categories = np.unique(color_by)

    # 2. Compute Mahalanobis distance for each category
    distances = []
    for category in unique_categories:
        X_pos = transformed_data[(labels == 1) & (color_by == category)]
        X_neg = transformed_data[(labels == 0) & (color_by == category)]
        if X_pos.shape[0] > 0 and X_neg.shape[0] > 0:
            distance = mahalanobis_distance(X_pos, X_neg)
            distances.append((category, distance))

    # 3. Sort categories by distance
    distances.sort(key=lambda x: x[1], reverse=True)

    # 4. Return top 3 categories
    return [cat for cat, _ in distances[:3]]