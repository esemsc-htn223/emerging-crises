"""
Time Series PCA Analysis Framework

This module provides a comprehensive framework for applying Principal Component Analysis (PCA)
to time series data with three main approaches:
1. Sliding Window PCA: Applies PCA to fixed-length windows of data
2. Dynamic PCA: Continuously adapts PCA to evolving data patterns
3. Frequency-Based PCA: Applies PCA in the frequency domain using Fourier transforms

The framework ensures no look-ahead bias and handles preparation, standardization,
and visualization appropriately for time series data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, List, Union
from sklearn.preprocessing import StandardScaler
from scipy import stats

from data.log_utilities import setup_logging

class TimeSeriesPCABase(ABC):
    """
    Abstract base class for time series PCA implementations.
    
    This class provides common functionality for PCA analysis on time series data,
    including data preparation, standardization without look-ahead bias,
    trend/seasonality removal, and visualization capabilities.
    
    Attributes:
        name (str): Name identifier for the PCA analysis.
        n_components (Union[int, float]): Number of components or variance threshold.
        window_size (int): Size of the analysis window.
        standardize (bool): Whether to standardize the data.
        remove_trend (bool): Whether to remove trends from the data.
        remove_seasonality (bool): Whether to remove seasonality.
        logger: Logger instance for tracking operations.
        scaler (StandardScaler): Scaler for data standardization.
        components_ (np.ndarray): Principal components.
        explained_variance_ratio_ (np.ndarray): Explained variance ratios.
        eigenvalues_ (np.ndarray): Eigenvalues from PCA.
        cumulative_variance_ (np.ndarray): Cumulative explained variance.
    """
    
    def __init__(self,
                 name: str,
                 n_components: Union[int, float] = 0.95,
                 window_size: int = 30,
                 standardize: bool = True,
                 remove_trend: bool = True,
                 remove_seasonality: bool = False,
                 trend_method: str = 'rolling_mean',
                 seasonality_period: int = 12):
        """
        Initialize the Time Series PCA base class.
        
        Args:
            name: Identifier for this PCA analysis.
            n_components: Number of components to keep or variance threshold (0-1).
            window_size: Size of the analysis window (days/periods).
            standardize: Whether to standardize data before PCA.
            remove_trend: Whether to remove trends from data.
            remove_seasonality: Whether to remove seasonal patterns.
            trend_method: Method for trend removal ('rolling_mean', 'differencing').
            seasonality_period: Period for seasonal decomposition.
        """
        self.name = name
        self.n_components = n_components
        self.window_size = window_size
        self.standardize = standardize
        self.remove_trend = remove_trend
        self.remove_seasonality = remove_seasonality
        self.trend_method = trend_method
        self.seasonality_period = seasonality_period
        
        # Initialize logging
        self.logger = setup_logging()
        self.logger.info(f"Initialized {self.__class__.__name__} with name '{name}'")
        
        # Initialize storage for data features
        self.columns = None

        # Initialize storage for PCA results
        # (dictionary with date index as keys)
        self.scaler = None
        self.components = dict()
        self.explained_variance_ratio = dict()
        self.eigenvalues = dict()
        self.cumulative_variance = dict()
        
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate input data format and requirements."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if len(df) < self.window_size:
            raise ValueError(f"DataFrame has {len(df)} rows but window_size is {self.window_size}")
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("DataFrame must contain numeric columns for PCA")

        self.logger.info(f"Data validation passed: {len(df)} rows, {len(numeric_cols)} numeric columns")
    
    def _standardize_data(self, df: pd.DataFrame, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Standardize data without look-ahead bias.
        
        Args:
            df: DataFrame to standardize.
            fit_scaler: Whether to fit the scaler (True for training data).
            
        Returns:
            Standardized DataFrame.
        """
        if not self.standardize:
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_standardized = df.copy()
        
        if fit_scaler:
            self.scaler = StandardScaler()
            scaled_values = self.scaler.fit_transform(df[numeric_cols])
            self.logger.info(f"Fitted scaler on {len(numeric_cols)} numeric columns")
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit_scaler=True first.")
            scaled_values = self.scaler.transform(df[numeric_cols])
        
        df_standardized[numeric_cols] = scaled_values
        return df_standardized
    
    def _remove_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove trends from time series data.
        
        Args:
            df: DataFrame to detrend.
            
        Returns:
            Detrended DataFrame.
        """
        if not self.remove_trend:
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_detrended = df.copy()

        window = min(self.window_size // 3, 12)
        if window == 1 and self.trend_method == 'rolling_mean':
            self.logger.info(f"Data size too small for rolling_mean, using differencing instead.")
            self.trend_method = 'differencing'
        
        if self.trend_method == 'rolling_mean':
            # Remove rolling mean trend
            for col in numeric_cols:
                rolling_mean = df[col].rolling(window=window, min_periods=1).mean()
                df_detrended[col] = df[col] - rolling_mean
                
        elif self.trend_method == 'differencing':
            # First difference to remove trend
            for col in numeric_cols:
                df_detrended[col] = df[col].diff().fillna(0)
                
        elif self.trend_method == 'linear_detrend':
            # Remove linear trend
            for col in numeric_cols:
                x = np.arange(len(df))
                slope, intercept, _, _, _ = stats.linregress(x, df[col].ffill())
                trend = slope * x + intercept
                df_detrended[col] = df[col] - trend
        
        self.logger.info(f"Trend removal completed using {self.trend_method}")
        
        return df_detrended
    
    def _remove_seasonal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove seasonal patterns from time series data.
        
        Args:
            df: DataFrame to deseasonalize.
            
        Returns:
            Deseasonalized DataFrame.
        """
        if not self.remove_seasonality:
            return df
        
        with DataLogger(self.logger, f"remove_seasonality(period={self.seasonality_period})", df_shape=df.shape):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df_deseasoned = df.copy()
            
            for col in numeric_cols:
                series = df[col].ffill()
                
                # Seasonal decomposition using rolling statistics
                if len(series) >= self.seasonality_period * 2:
                    # Calculate seasonal component
                    seasonal = series.groupby(series.index % self.seasonality_period).transform('mean')
                    df_deseasoned[col] = series - seasonal
                else:
                    # If not enough data for seasonal decomposition, skip
                    self.logger.warning(f"Insufficient data for seasonal decomposition of {col}")
                    
            self.logger.info("Seasonal pattern removal completed")
            
        return df_deseasoned
    
    def _prepare_data(self, df: pd.DataFrame, fit_preprocessing: bool = True) -> pd.DataFrame:
        """
        Complete data preparation pipeline for time series data.
        
        Args:
            df: Input DataFrame.
            fit_preprocessing: Whether to fit preprocessing parameters.
            
        Returns:
            DataFrame prepared for PCA.
        """
        with DataLogger(self.logger, "prepare_data", df_shape=df.shape):
            # Store original columns and index
            if fit_preprocessing:
                self.columns = df.select_dtypes(include=[np.number]).columns.tolist()
            df_sorted = df.sort_index(level=1)
            self.index = df_sorted.index

            prepared_df = df_sorted.copy()

            # 1. Handle any remaining NaN values
            for col in self.columns:
                prepared_df[col] = prepared_df[col].ffill()
                prepared_df[col] = prepared_df[col].fillna(0)

            # 2. Standardize the data
            prepared_df = self._standardize_data(prepared_df, fit_scaler=fit_preprocessing)
            
            # 3. Remove trends and seasonality
            prepared_df = self._remove_trends(prepared_df)
            prepared_df = self._remove_seasonal_patterns(prepared_df)
            
            self.logger.info("Data prepared for PCA successfully")
            
        return prepared_df
    
    def _calculate_covariance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the covariance matrix for PCA.
        
        Args:
            data: Preprocessed data array.
            
        Returns:
            Covariance matrix.
        """
        with DataLogger(self.logger, "calculate_covariance_matrix", df_shape=data.shape):
            # Center the data (subtract mean)
            centered_data = data - np.mean(data, axis=0)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(centered_data.T)
            
            self.logger.info(f"Covariance matrix calculated: {cov_matrix.shape}")
            
        return cov_matrix
    
    def _compute_eigenvectors_eigenvalues(self, index: str, cov_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvectors and eigenvalues from covariance matrix.
        
        Args:
            cov_matrix: Covariance matrix.
            
        Returns:
            Tuple of (eigenvalues, eigenvectors).
        """
        with DataLogger(self.logger, "compute_eigenvectors_eigenvalues"):
            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # Sort in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            total_variance = np.sum(eigenvalues)
            
            # Update results
            self.eigenvalues[index] = eigenvalues
            self.components[index] = eigenvectors.T
            self.explained_variance_ratio[index] = eigenvalues / total_variance
            self.cumulative_variance[index] = np.cumsum(self.explained_variance_ratio[index])

            self.logger.info(f"Computed {len(eigenvalues)} eigenvectors and eigenvalues")
        
        return eigenvalues, eigenvectors
    
    def _select_components(self, index: str) -> int:
        """
        Select the optimal number of components based on variance criteria.
        
        Returns:
            Number of components to keep.
        """
        if isinstance(self.n_components, int):
            n_components = min(self.n_components, len(self.eigenvalues[index]))
        else:
            # Find number of components for variance threshold
            n_components = np.argmax(self.cumulative_variance[index] >= self.n_components) + 1
            n_components = max(1, n_components)  # At least 1 component
        
        variance_explained = self.cumulative_variance[index][n_components - 1]
        self.logger.info(f"Selected {n_components} components explaining {variance_explained:.1%} of variance")
        
        return n_components
    
    def _compute_global_components(self) -> None:
        """
        Compute global components by aggregating country-specific components.
        Uses weighted averaging based on explained variance ratios.
        """
        if not self.components:
            raise ValueError("No country-specific components available for global computation")
        
        with DataLogger(self.logger, "compute_global_components"):
            # Collect all components and their weights
            all_components = []
            all_variance_ratios = []
            country_weights = []
            
            for country in self.components.keys():
                components = self.components[country]
                variance_ratios = self.explained_variance_ratio[country]
                
                # Weight by total variance explained by first component
                weight = variance_ratios[0] if len(variance_ratios) > 0 else 0
                
                all_components.append(components)
                all_variance_ratios.append(variance_ratios)
                country_weights.append(weight)
                self.fitted_countries.add(country)
            
            # Normalize weights
            total_weight = sum(country_weights)
            if total_weight > 0:
                country_weights = [w / total_weight for w in country_weights]
            else:
                country_weights = [1.0 / len(country_weights)] * len(country_weights)
            
            # Determine maximum number of components across countries
            max_components = max(len(comp) for comp in all_components)
            n_features = all_components[0].shape[1]
            
            # Initialize global components matrix
            global_components = np.zeros((max_components, n_features))
            global_variance_ratios = np.zeros(max_components)
            
            # Compute weighted average for each component
            for comp_idx in range(max_components):
                weighted_component = np.zeros(n_features)
                weighted_variance = 0
                total_weight_for_comp = 0
                
                for _, (components, variance_ratios, weight) in enumerate(
                    zip(all_components, all_variance_ratios, country_weights)):
                    
                    if comp_idx < len(components):
                        # Ensure consistent orientation (handle sign ambiguity)
                        component = components[comp_idx]
                        if comp_idx == 0 and np.sum(weighted_component) != 0:
                            # Align with existing direction
                            if np.dot(component, weighted_component) < 0:
                                component = -component
                        
                        weighted_component += weight * component
                        weighted_variance += weight * variance_ratios[comp_idx]
                        total_weight_for_comp += weight
                
                if total_weight_for_comp > 0:
                    global_components[comp_idx] = weighted_component / total_weight_for_comp
                    global_variance_ratios[comp_idx] = weighted_variance / total_weight_for_comp
                    
                    # Normalize component to unit length
                    norm = np.linalg.norm(global_components[comp_idx])
                    if norm > 1e-10:
                        global_components[comp_idx] /= norm
            
            # Store global results
            self.global_components = global_components
            self.global_explained_variance_ratio = global_variance_ratios
            self.global_cumulative_variance = np.cumsum(global_variance_ratios)
            
            # Compute global eigenvalues (approximate from variance ratios)
            total_variance = np.sum(global_variance_ratios)
            self.global_eigenvalues = global_variance_ratios * total_variance
            
            self.logger.info(f"Global components computed: {max_components} components, "
                           f"total variance explained: {total_variance:.3f}")
    
    def _select_global_components(self, n_components: Optional[int] = None) -> int:
        """
        Select number of global components based on variance criteria.
        
        Args:
            n_components: Explicit number of components or None for auto-selection
            
        Returns:
            Number of components to use
        """
        if self.global_components is None:
            raise ValueError("Global components not computed. Call fit() first.")
        
        if n_components is not None:
            return min(n_components, len(self.global_components))
        
        if isinstance(self.n_components, int):
            return min(self.n_components, len(self.global_components))
        else:
            # Find number of components for variance threshold
            if self.global_cumulative_variance is None or len(self.global_cumulative_variance) == 0:
                self.logger.warning("Global cumulative variance not available, using first component")
                return 1
                
            # Find first component that exceeds threshold
            exceeds_threshold = self.global_cumulative_variance >= self.n_components
            if not np.any(exceeds_threshold):
                # If no component exceeds threshold, use all components
                n_comp = len(self.global_components)
                self.logger.warning(f"No component exceeds variance threshold {self.n_components}, using all {n_comp} components")
            else:
                n_comp = np.argmax(exceeds_threshold) + 1
                
            n_comp = max(1, min(n_comp, len(self.global_components)))
            
            variance_explained = self.global_cumulative_variance[n_comp - 1]
            self.logger.info(f"Selected {n_comp} global components explaining "
                        f"{variance_explained:.1%} of variance")
            return n_comp
    
    def transform(self, df: pd.DataFrame, n_components: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Transform data using fitted PCA components (country-specific or global).
        
        Args:
            df: DataFrame to transform
            n_components: Number of components to use (default: auto-selected)
            
        Returns:
            Dictionary of transformed data arrays, one per country
        """
        if not self.components and self.global_components is None:
            raise ValueError("PCA not fitted. Call fit() method first.")

        if self.standardize and self.scaler is None:
            raise ValueError("Scaler not fitted. PCA must be fitted before transform.")
        
        # Ensure global components are computed
        if self.global_components is None and self.components:
            self._compute_global_components()
        
        # Prepare data using fitted parameters
        prepared_df = self._prepare_data(df, fit_preprocessing=False)
        
        if self.columns is not None:
            numeric_cols = prepared_df.select_dtypes(include=[np.number]).columns
            if not all(col in numeric_cols for col in self.columns):
                missing_cols = [col for col in self.columns if col not in numeric_cols]
                raise ValueError(f"Missing features in transform data: {missing_cols}")
            # Use same column order as during fitting
            numeric_cols = [col for col in self.columns if col in numeric_cols]
        
        # Determine number of components to use
        if n_components is None:
            n_components = self._select_global_components()
        else:
            n_components = self._select_global_components(n_components)
        
        # Transform data for each country
        transformed_data = {}
        
        for country in prepared_df.index.get_level_values('Country').unique():
            try:
                # Extract country data
                country_data = prepared_df.loc[(country, slice(None)), numeric_cols]
                data_array = country_data.values
                
                if country in self.components:
                    # Use country-specific components for fitted countries
                    components = self.components[country][:n_components]
                    self.logger.debug(f"Using country-specific components for {country}")
                else:
                    # Use global components for unseen countries
                    components = self.global_components[:n_components]
                    self.logger.debug(f"Using global components for unseen country {country}")
                
                # Ensure data and components are compatible
                if data_array.shape[1] != components.shape[1]:
                    self.logger.warning(f"Dimension mismatch for {country}: "
                                      f"data {data_array.shape[1]} vs components {components.shape[1]}")
                    min_features = min(data_array.shape[1], components.shape[1])
                    data_array = data_array[:, :min_features]
                    components = components[:, :min_features]
                
                # Transform using selected components
                transformed_data[country] = np.dot(data_array, components.T)
                
            except Exception as e:
                self.logger.error(f"Failed to transform data for {country}: {str(e)}")
                continue
        
        self.logger.info(f"Transformed data for {len(transformed_data)} countries using "
                        f"{n_components} components")
        return transformed_data
    
    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs) -> 'TimeSeriesPCABase':
        """
        Fit the PCA model to the data.
        
        Args:
            df: Input DataFrame for fitting.
            **kwargs: Additional arguments specific to implementation.
            
        Returns:
            Self for method chaining.
        """
        pass
    
    @abstractmethod
    def fit_transform(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Fit the PCA model and transform the data.
        
        Args:
            df: Input DataFrame.
            **kwargs: Additional arguments specific to implementation.
            
        Returns:
            Transformed data array.
        """
        pass


class SlidingWindowPCA(TimeSeriesPCABase):
    """
    Sliding Window PCA implementation for time series analysis.
    
    This class applies PCA to fixed-length windows of time series data,
    sliding the window forward step by step to capture evolving patterns
    while maintaining temporal structure.
    
    The sliding window approach is useful for:
    - Detecting gradual changes in data patterns
    - Maintaining temporal locality of analysis
    - Identifying structural breaks or regime changes
    """
    
    def __init__(self, 
                 name: str,
                 window_size: int,
                 step_size: int,
                 **kwargs):
        """
        Initialize Sliding Window PCA.
        
        Args:
            name: Identifier for this analysis.
            window_size: Size of each analysis window.
            step_size: Step size for sliding the window.
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(name=name, window_size=window_size, **kwargs)
        self.step_size = step_size
        self.window_results = dict()
        self.window_dates = dict()

        self.title = f"[Window: {self.window_size}, Step: {self.step_size}]"
    
    def fit(self, df: pd.DataFrame) -> 'SlidingWindowPCA':
        """
        Fit sliding window PCA to the time series data.
        
        Args:
            df: Input DataFrame with MultiIndex (Country, DateTime) and time series data.
            
        Returns:
            Self for method chaining.
        """
        with DataLogger(self.logger, f"fit_sliding_window_pca(window_size={self.window_size}, step_size={self.step_size})", df_shape=df.shape):
            self._validate_data(df)
            
            # Get date information from second index level (DateTime)
            df_sorted = df.sort_index(level=1)
            
            # For each country
            for country in df_sorted.index.get_level_values('Country').unique():
                country_df = df_sorted.loc[(country, slice(None)), :]
                dates = pd.to_datetime(country_df.index.get_level_values(1))
                
                _window_results = dict()
                _window_dates = []

                # Slide window through the data
                for start_idx in range(0, len(country_df) - self.window_size + 1, self.step_size):
                    end_idx = start_idx + self.window_size
                    
                    # Extract window data
                    window_data = country_df.iloc[start_idx:end_idx]
                    window_date = dates[end_idx - 1]  # Use end date of window
                    
                    # Prepare window data
                    processed_window = self._prepare_data(window_data, fit_preprocessing=True)
                    numeric_cols = processed_window.select_dtypes(include=[np.number]).columns
                    data_array = processed_window[numeric_cols].values
                    
                    # Apply PCA to window
                    if data_array.shape[0] > 1 and data_array.shape[1] > 1:
                        cov_matrix = self._calculate_covariance_matrix(data_array)
                        eigenvalues, eigenvectors = self._compute_eigenvectors_eigenvalues(country, cov_matrix)
                        
                        # Append window results
                        # (Note: sliding window stores results from multiple windows per country.
                        # This is stored as nested dictionary within window_results)
                        _window_results[start_idx] = {
                            'eigenvalues': eigenvalues.copy(),
                            'components': eigenvectors.T.copy(),
                            'explained_variance_ratio': self.explained_variance_ratio[country].copy(),
                            'cumulative_variance': self.cumulative_variance[country].copy()
                        }
                        _window_dates.append(window_date)
                
                self.window_results[country] = _window_results
                self.window_dates[country] = _window_dates

            # After processing all windows, compute global components
            # Aggregate the final window results for each country
            for country in self.window_results.keys():
                if self.window_results[country]:
                    # Use the last window's results as the country's representative components
                    last_window_key = max(self.window_results[country].keys())
                    last_results = self.window_results[country][last_window_key]
                    
                    # Store in the base class format for global computation
                    self.components[country] = last_results['components']
                    self.explained_variance_ratio[country] = last_results['explained_variance_ratio']
                    self.eigenvalues[country] = last_results['eigenvalues']
                    self.cumulative_variance[country] = last_results['cumulative_variance']
            
            # Compute global components
            if self.components:
                self._compute_global_components()

                self.logger.info(f"Sliding window PCA completed: {len(_window_dates)} windows processed")
            
        return self
    
    def fit_transform(self, df: pd.DataFrame) -> List[np.ndarray]:
        """
        Fit sliding window PCA and return transformed data for each window.
        
        Args:
            df: Input DataFrame with MultiIndex (Country, DateTime).
            
        Returns:
            List of transformed data arrays, one per window.
        """
        self.fit(df)
        return self.transform(df)

    def plot_scree_plot(self, show: bool = True,
                        aggregate_by: Optional[str] = "last",
                        countries: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Create a scree plot showing eigenvalues and explained variance.
        
        Args:
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
        """
        if self.window_results is None:
            raise ValueError("PCA not fitted. Call fit() method first.")

        if aggregate_by not in ["mean", "min", "max", "last"]:
            raise ValueError(f"Invalid aggregate_by value: {aggregate_by}. Must be one of ['mean', 'min', 'max', 'last']")

        if countries is None or any(country not in list(self.window_results.keys()) for country in countries):
            countries = list(self.window_results.keys())

        with DataLogger(self.logger, "plot_scree_plot"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            threshold1 = 1e-6  # Plotting threshold for eigenvalues to consider
            threshold2 = 0.98 # Plotting threshold for cumulative variance
        
            # Create loading matrix for each country x window
            for country in countries:
                eigenvalues, cumulative_variance = [], []
                for _, result in self.window_results[country].items():
                    eigenvalues.append(result['eigenvalues'])
                    cumulative_variance.append(result['cumulative_variance'])

                if aggregate_by == "last":
                    _eigenvalues = eigenvalues[-1]
                    _cumulative_variance = cumulative_variance[-1]
                else:
                    eigenvalues = np.array(eigenvalues)
                    cumulative_variance = np.array(cumulative_variance)
                    if aggregate_by == "mean":
                        _eigenvalues = np.mean(eigenvalues, axis=0)
                        _cumulative_variance = np.mean(cumulative_variance, axis=0)
                    elif aggregate_by == "min":
                        _eigenvalues = np.min(eigenvalues, axis=0)
                        _cumulative_variance = np.min(cumulative_variance, axis=0)
                    else:
                        _eigenvalues = np.max(eigenvalues, axis=0)
                        _cumulative_variance = np.max(cumulative_variance, axis=0)

                # Scree plot (eigenvalues)
                if country == "United States":
                    ax1.plot(range(1, sum(_eigenvalues > threshold1) + 1),
                             _eigenvalues[_eigenvalues > threshold1],
                             'ro-', linewidth=2, markersize=4, label=f"{country}")
                elif country == "Japan":
                    ax1.plot(range(1, sum(_eigenvalues > threshold1) + 1),
                             _eigenvalues[_eigenvalues > threshold1],
                             'bo-', linewidth=2, markersize=4, label=f"{country}")
                else:
                    ax1.plot(range(1, sum(_eigenvalues > threshold1) + 1),
                             _eigenvalues[_eigenvalues > threshold1],
                             'ko-', linewidth=0.25, markersize=1)
                ax1.set_xlabel('Principal Component')
                ax1.set_ylabel('Eigenvalue')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Cumulative variance plot
                if country == "United States":
                    ax2.plot(range(1, sum(_cumulative_variance < threshold2) + 1),
                             _cumulative_variance[_cumulative_variance < threshold2],
                             'ro-', linewidth=2, markersize=4, label=f"{country}")
                elif country == "Japan":
                    ax2.plot(range(1, sum(_cumulative_variance < threshold2) + 1),
                             _cumulative_variance[_cumulative_variance < threshold2],
                             'bo-', linewidth=2, markersize=4, label=f"{country}")
                else:
                    ax2.plot(range(1, sum(_cumulative_variance < threshold2) + 1),
                             _cumulative_variance[_cumulative_variance < threshold2],
                             'ko-', linewidth=0.25, markersize=1)
                ax2.set_xlabel('Principal Component')
                ax2.set_ylabel('Cumulative Explained Variance Ratio')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                ax2.set_ylim(0, 1)
            
            plt.suptitle(f'Scree Plot - {self.name}\n{self.title}')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Scree plot saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()

    def plot_component_loadings(self, show: bool = True,
                                n_components: int = 3,
                                color_by: Optional[str] = "max",
                                countries: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Plot component loadings showing how original variables contribute to components.
        Overrides base class method to handle multiple windows.

        Args:
            n_components: Number of components to plot.
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
        """
        if self.window_results is None:
            raise ValueError("PCA not fitted. Call fit() method first.")
        
        if countries is None or any(country not in list(self.window_results.keys()) for country in countries):
                countries = list(self.window_results.keys())

        if color_by not in ["mean", "min", "max"]:
            raise ValueError(f"Invalid color_by value: {color_by}. Must be one of ['mean', 'min', 'max']")

        with DataLogger(self.logger, f"plot_component_loadings(n_components={n_components})"):
            loadings = []
            # Create loading matrix for each country x window
            for country in countries:
                n_components = min(n_components, len(self.window_results[country][0]['components']))
                for _, result in self.window_results[country].items():
                    loadings.append(result['components'][:n_components].T)

            loadings = np.array(loadings)
            if color_by == "mean":
                _loadings = np.mean(loadings, axis=0)
            elif color_by == "min":
                _loadings = np.min(loadings, axis=0)
            else:
                _loadings = np.max(loadings, axis=0)
            
            # Create heatmap
            plt.figure(figsize=(12, max(8, len(self.columns) * 0.3)))

            sns.heatmap(_loadings, 
                        xticklabels=[f'PC{i+1}' for i in range(n_components)],
                        yticklabels=self.columns,
                        annot=True, 
                        cmap='RdBu_r', 
                        center=0,
                        fmt='.3f')
            
            plt.title(f'{color_by.capitalize()} Component Loadings - {self.name}')
            plt.xlabel('Principal Components')
            plt.ylabel('Original Variables')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Component loadings plot saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()

    def plot_evolving_variance(self, show: bool = True,
                               save_path: Optional[str] = None,
                               label_file: Optional[str] = "crisis_labels.csv") -> None:
        """
        Plot how explained variance evolves over time across windows.
        
        Args:
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
        """
        if not self.window_results:
            raise ValueError("No window results available. Call fit() first.")
        
        with DataLogger(self.logger, "plot_evolving_variance"):
            labels = None
            try:
                labels = pd.read_csv(label_file, index_col=0)
            except FileNotFoundError:
                self.logger.warning(f"Label file {label_file} not found.")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            for country in self.window_results.keys():
                results = self.window_results[country]
                
                # Extract variance data over time
                dates = self.window_dates[country]
                first_pc_variance = [result['explained_variance_ratio'][0] for _, result in results.items()]
                total_variance_3pc = [np.sum(result['explained_variance_ratio'][:3]) for _, result in results.items()]
                
                # Extract crisis labels if available
                crisis_idx = []
                if labels is not None and country in labels.index:
                    crisis_years = labels.loc[country, 'Year'].values
                    crisis_idx = [i for i, d in enumerate(dates) if d.year in crisis_years]
                
                # Plot first PC variance over time
                if country == "United States":
                    ax1.plot(dates, first_pc_variance, 'r-', linewidth=2, label=f"{country}")
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [first_pc_variance[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax1.plot(dates, first_pc_variance, 'b-', linewidth=2, label=f"{country}")
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [first_pc_variance[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax1.plot(dates, first_pc_variance, 'k-', linewidth=0.25)
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [first_pc_variance[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
                ax1.set_ylabel('Explained Variance')
                ax1.set_title(f'Evolution of Variance of First PC - {self.name}\n{self.title}')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
            
                # Plot cumulative variance of first 3 PCs
                if country == "United States":
                    ax2.plot(dates, total_variance_3pc, 'r-', linewidth=2, label=f"{country}")
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [total_variance_3pc[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax2.plot(dates, total_variance_3pc, 'b-', linewidth=2, label=f"{country}")
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [total_variance_3pc[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax2.plot(dates, total_variance_3pc, 'k-', linewidth=0.25)
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [total_variance_3pc[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Cumulative Explained Variance')
                ax2.set_title('Evolution of Cumulative Variance of First 3 PCs')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Evolving variance plot saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()

    def plot_eigenvalue_features(self, show: bool = True,
                                 save_path: Optional[str] = None,
                                 label_file: Optional[str] = "crisis_labels.csv") -> None:
        
        if not self.window_results:
            raise ValueError("No window results available. Call fit() first.")
        
        with DataLogger(self.logger, "plot_spectral_entropy"):
            labels = None
            try:
                labels = pd.read_csv(label_file, index_col=0)
            except FileNotFoundError:
                self.logger.warning(f"Label file {label_file} not found.")
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

            for country in self.window_results.keys():
                results = self.window_results[country]
                
                # Extract features over time
                dates = self.window_dates[country]
                num_eigenvalues, eigenvalue_ratio, spectral_entropy = [], [], []
                
                for _, result in results.items():
                    eigenvalues = result['eigenvalues']
                    sum_eigenvalues = np.sum(eigenvalues)
                    # Ratio of first eigenvalue to sum of all eigenvalues
                    eigenvalue_ratio.append(eigenvalues[0] / sum_eigenvalues)
                    eigenvalues_normalized = eigenvalues / sum_eigenvalues
                    # Spectral entropy of eigenvalues
                    entropy = -np.sum(eigenvalues_normalized * np.log2(eigenvalues_normalized + 1e-10))
                    spectral_entropy.append(entropy)
                    # Count number of eigenvalues explaining at least 95% variance
                    num_eigenvalues.append(np.argmax(result['cumulative_variance'] >= 0.95) + 1)
                
                # Extract crisis labels if available
                crisis_idx = []
                if labels is not None and country in labels.index:
                    crisis_years = labels.loc[country, 'Year'].values
                    crisis_idx = [i for i, d in enumerate(dates) if d.year in crisis_years]
                
                # Plot number of eigenvalues required to explain at least 95% variance
                if country == "United States":
                    ax1.plot(dates, num_eigenvalues, 'r-', linewidth=2, label=f"{country}")
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [num_eigenvalues[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax1.plot(dates, num_eigenvalues, 'b-', linewidth=2, label=f"{country}")
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [num_eigenvalues[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax1.plot(dates, num_eigenvalues, 'k-', linewidth=0.25)
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [num_eigenvalues[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
                ax1.set_ylabel('Number of Eigenvalues')
                ax1.set_title(f'Evolution of Number of Eigenvalues Required to Explain at least 95% of Variance - {self.name}\n{self.title}')
                ax1.grid(True, alpha=0.3)
                ax1.legend()

                # Plot ratio of first eigenvalue to sum of all eigenvalues
                if country == "United States":
                    ax2.plot(dates, eigenvalue_ratio, 'r-', linewidth=2, label=f"{country}")
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [eigenvalue_ratio[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax2.plot(dates, eigenvalue_ratio, 'b-', linewidth=2, label=f"{country}")
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [eigenvalue_ratio[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax2.plot(dates, eigenvalue_ratio, 'k-', linewidth=0.25)
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [eigenvalue_ratio[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
                ax2.set_ylabel(r'$\lambda_1 : \sum \lambda_i$')
                ax2.set_title(f'Evolution of Ratio of First Eigenvalue to Sum of All Eigenvalues')
                ax2.grid(True, alpha=0.3)

                # Plot spectral entropy of eigenvalues over time
                if country == "United States":
                    ax3.plot(dates, spectral_entropy, 'r-', linewidth=2, label=f"{country}")
                    ax3.scatter([dates[i] for i in crisis_idx],
                                [spectral_entropy[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax3.plot(dates, spectral_entropy, 'b-', linewidth=2, label=f"{country}")
                    ax3.scatter([dates[i] for i in crisis_idx],
                                [spectral_entropy[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax3.plot(dates, spectral_entropy, 'k-', linewidth=0.25)
                    ax3.scatter([dates[i] for i in crisis_idx],
                                [spectral_entropy[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
                ax3.set_xlabel('Date')
                ax3.set_ylabel('Spectral Entropy')
                ax3.set_title(f'Evolution of Spectral Entropy of Eigenvalues')
                ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Evolving variance plot saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()


class DynamicPCA(TimeSeriesPCABase):
    """
    Dynamic PCA implementation with adaptive learning capabilities.
    
    This class implements dynamic PCA that continuously adapts to new data,
    using techniques like recursive updates and exponential forgetting to
    maintain relevance to recent patterns while learning from historical data.
    
    Each country is processed independently to maintain country-specific patterns.
    """
    
    def __init__(self,
                 name: str,
                 forgetting_factor: float = 0.95,
                 min_samples: int = 10,
                 update_frequency: int = 1,
                 **kwargs):
        """
        Initialize Dynamic PCA.
        
        Args:
            name: Identifier for this analysis.
            forgetting_factor: Exponential forgetting factor (0-1).
            min_samples: Minimum samples before starting PCA.
            update_frequency: How often to update the PCA (every N samples).
            **kwargs: Additional arguments passed to base class.
        """
        super().__init__(name=name, **kwargs)
        self.forgetting_factor = forgetting_factor
        self.min_samples = min_samples
        self.update_frequency = update_frequency
        
        self.update_history = dict()  # {country: [history_entries]}
        self.update_dates = dict()    # {country: [dates]}
        self.country_state = dict()   # {country: {running_mean, running_cov, sample_count}}
        
        self.title = f"[Forgetting: {self.forgetting_factor}, Min Samples: {self.min_samples}, Update Freq: {self.update_frequency}]"
    
    def _update_running_statistics(self, country: str, new_data: np.ndarray) -> None:
        """
        Update running mean and covariance with exponential forgetting for a specific country.
        
        Args:
            country: Country identifier.
            new_data: New data point to incorporate.
        """
        # Validate input data
        if np.any(np.isnan(new_data)):
            self.logger.warning(f"NaN values detected in update for {country} - skipping update")
            return
        
        if np.any(np.isinf(new_data)):
            self.logger.warning(f"Infinite values detected in update for {country} - skipping update")
            return
    
        if country not in self.country_state:
            # Initialize country state
            self.country_state[country] = {
                'running_mean': new_data.copy(),
                'running_cov': np.zeros((len(new_data), len(new_data))),
                'sample_count': 1
            }
        else:
            state = self.country_state[country]
            
            if state['running_mean'] is None:
                # Initialize if not yet set
                state['running_mean'] = new_data.copy()
                state['running_cov'] = np.zeros((len(new_data), len(new_data)))
                state['sample_count'] = 1
            else:
                # Update with exponential forgetting
                alpha = 1 - self.forgetting_factor
                
                # Update mean
                old_mean = state['running_mean'].copy()
                state['running_mean'] = self.forgetting_factor * state['running_mean'] + alpha * new_data
                
                # Update covariance (Welford's online algorithm with forgetting)
                delta1 = new_data - old_mean
                delta2 = new_data - state['running_mean']
                state['running_cov'] = self.forgetting_factor * state['running_cov'] + alpha * np.outer(delta1, delta2)
                
                state['sample_count'] += 1
    
    def _recursive_pca_update(self, country: str, current_date: pd.Timestamp) -> None:
        """
        Update PCA components using current running statistics for a specific country.
        
        Args:
            country: Country identifier.
            current_date: Current date for the update.
        """
        if country not in self.country_state:
            return
            
        state = self.country_state[country]
        if state['running_cov'] is not None:
            eigenvalues, eigenvectors = self._compute_eigenvectors_eigenvalues(country, state['running_cov'])
            
            # Store update history for this country
            if country not in self.update_history:
                self.update_history[country] = []
                self.update_dates[country] = []
                
            update_info = {
                'sample_count': state['sample_count'],
                'eigenvalues': eigenvalues.copy(),
                'components': eigenvectors.T.copy(),
                'explained_variance_ratio': self.explained_variance_ratio[country].copy(),
                'cumulative_variance': self.cumulative_variance[country].copy()
            }
            self.update_history[country].append(update_info)
            self.update_dates[country].append(current_date)
    
    def fit(self, df: pd.DataFrame) -> 'DynamicPCA':
        """
        Fit dynamic PCA to time series data with incremental updates per country.
        
        Args:
            df: Input DataFrame with MultiIndex (Country, DateTime) and time series data.
            
        Returns:
            Self for method chaining.
        """
        with DataLogger(self.logger, f"fit_dynamic_pca(forgetting_factor={self.forgetting_factor})", df_shape=df.shape):
            self._validate_data(df)
            
            # Sort by DateTime (level 1)
            df_sorted = df.sort_index(level=1)
            
            # Process each country independently
            for country in df_sorted.index.get_level_values('Country').unique():
                country_df = df_sorted.loc[(country, slice(None)), :]
                dates = pd.to_datetime(country_df.index.get_level_values(1))
                
                # Initialize country state
                if country not in self.country_state:
                    self.country_state[country] = {
                        'running_mean': None,
                        'running_cov': None,
                        'sample_count': 0
                    }
                
                # Prepare data for this country
                processed_df = self._prepare_data(country_df, fit_preprocessing=True)
                numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                
                # Process data incrementally for this country
                for i, (idx, row) in enumerate(processed_df.iterrows()):
                    data_point = row[numeric_cols].values
                    current_date = dates[i]
                    
                    # Skip if data point contains NaN
                    if np.any(np.isnan(data_point)):
                        continue
                    
                    # Update running statistics for this country
                    self._update_running_statistics(country, data_point)
                    
                    # Update PCA if conditions are met
                    sample_count = self.country_state[country]['sample_count']
                    if (sample_count >= self.min_samples and 
                        sample_count % self.update_frequency == 0):
                        self._recursive_pca_update(country, current_date)
            
            # After processing all countries, compute global components
            # Use the latest results from each country
            for country in self.update_history.keys():
                if self.update_history[country]:
                    # Use the last update's results as the country's representative components
                    last_update = self.update_history[country][-1]
                    
                    # Store in the base class format for global computation
                    self.components[country] = last_update['components']
                    self.explained_variance_ratio[country] = last_update['explained_variance_ratio']
                    self.eigenvalues[country] = last_update['eigenvalues']
                    self.cumulative_variance[country] = last_update['cumulative_variance']
            
            # Compute global components
            if self.components:
                self._compute_global_components()
                
            total_updates = sum(len(history) for history in self.update_history.values())
            self.logger.info(f"Dynamic PCA completed: {total_updates} total updates across {len(self.update_history)} countries")
            
        return self
    
    def fit_transform(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Fit dynamic PCA and return transformed data for each country.
        
        Args:
            df: Input DataFrame with MultiIndex (Country, DateTime).
            
        Returns:
            Dictionary mapping country names to transformed data arrays.
        """
        self.fit(df)
        return self.transform(df)
    
    def plot_scree_plot(self, show: bool = True,
                        aggregate_by: Optional[str] = "last",
                        countries: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Create a scree plot showing eigenvalues and explained variance.
        
        Args:
            show: Whether to display the plot.
            aggregate_by: How to aggregate across updates ("last", "mean", "min", "max").
            countries: List of countries to include (if None, includes all).
            save_path: Optional path to save the plot.
        """
        if not self.update_history:
            raise ValueError("PCA not fitted. Call fit() method first.")

        if aggregate_by not in ["mean", "min", "max", "last"]:
            raise ValueError(f"Invalid aggregate_by value: {aggregate_by}. Must be one of ['mean', 'min', 'max', 'last']")

        if countries is None or any(country not in list(self.update_history.keys()) for country in countries):
            countries = list(self.update_history.keys())

        with DataLogger(self.logger, "plot_scree_plot"):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            threshold1 = 1e-6  # Plotting threshold for eigenvalues to consider
            threshold2 = 0.98  # Plotting threshold for cumulative variance
        
            # Create loading matrix for each country
            for country in countries:
                updates = self.update_history[country]
                if not updates:
                    continue
                    
                eigenvalues, cumulative_variance = [], []
                for update in updates:
                    eigenvalues.append(update['eigenvalues'])
                    cumulative_variance.append(update['cumulative_variance'])

                if aggregate_by == "last":
                    _eigenvalues = eigenvalues[-1]
                    _cumulative_variance = cumulative_variance[-1]
                else:
                    eigenvalues = np.array(eigenvalues)
                    cumulative_variance = np.array(cumulative_variance)
                    if aggregate_by == "mean":
                        _eigenvalues = np.mean(eigenvalues, axis=0)
                        _cumulative_variance = np.mean(cumulative_variance, axis=0)
                    elif aggregate_by == "min":
                        _eigenvalues = np.min(eigenvalues, axis=0)
                        _cumulative_variance = np.min(cumulative_variance, axis=0)
                    else:
                        _eigenvalues = np.max(eigenvalues, axis=0)
                        _cumulative_variance = np.max(cumulative_variance, axis=0)

                # Scree plot (eigenvalues)
                if country == "United States":
                    ax1.plot(range(1, sum(_eigenvalues > threshold1) + 1),
                             _eigenvalues[_eigenvalues > threshold1],
                             'ro-', linewidth=2, markersize=4, label=f"{country}")
                elif country == "Japan":
                    ax1.plot(range(1, sum(_eigenvalues > threshold1) + 1),
                             _eigenvalues[_eigenvalues > threshold1],
                             'bo-', linewidth=2, markersize=4, label=f"{country}")
                else:
                    ax1.plot(range(1, sum(_eigenvalues > threshold1) + 1),
                             _eigenvalues[_eigenvalues > threshold1],
                             'ko-', linewidth=0.5, markersize=2, alpha=0.7)
                
                # Cumulative variance plot
                if country == "United States":
                    ax2.plot(range(1, sum(_cumulative_variance < threshold2) + 1),
                             _cumulative_variance[_cumulative_variance < threshold2],
                             'ro-', linewidth=2, markersize=4, label=f"{country}")
                elif country == "Japan":
                    ax2.plot(range(1, sum(_cumulative_variance < threshold2) + 1),
                             _cumulative_variance[_cumulative_variance < threshold2],
                             'bo-', linewidth=2, markersize=4, label=f"{country}")
                else:
                    ax2.plot(range(1, sum(_cumulative_variance < threshold2) + 1),
                             _cumulative_variance[_cumulative_variance < threshold2],
                             'ko-', linewidth=0.5, markersize=2, alpha=0.7)

            ax1.set_xlabel('Component Number')
            ax1.set_ylabel('Eigenvalue')
            ax1.set_title(f'Scree Plot - {self.name}\n{self.title}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.set_xlabel('Component Number')
            ax2.set_ylabel('Cumulative Explained Variance')
            ax2.set_title('Cumulative Explained Variance')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Scree plot saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()

    # TODO: come up with better visualization for component loadings
    def plot_component_loadings(self, show: bool = True,
                                n_components: int = 3,
                                color_by: Optional[str] = "max",
                                countries: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Plot component loadings showing how original variables contribute to components.

        Args:
            n_components: Number of components to plot.
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
        """
        if self.update_history is None:
            raise ValueError("PCA not fitted. Call fit() method first.")
        
        if countries is None or any(country not in list(self.update_history.keys()) for country in countries):
                countries = list(self.update_history.keys())

        if color_by not in ["mean", "min", "max"]:
            raise ValueError(f"Invalid color_by value: {color_by}. Must be one of ['mean', 'min', 'max']")

        with DataLogger(self.logger, f"plot_component_loadings(n_components={n_components})"):
            loadings = []
            # Create loading matrix for each country x window
            for country in countries:
                n_components = min(n_components, len(self.update_history[country][-1]['components']))
                loadings.append(self.update_history[country][-1]['components'][:n_components].T)

            loadings = np.array(loadings)
            if color_by == "mean":
                _loadings = np.mean(loadings, axis=0)
            elif color_by == "min":
                _loadings = np.min(loadings, axis=0)
            else:
                _loadings = np.max(loadings, axis=0)
            
            # Create heatmap
            plt.figure(figsize=(12, max(8, len(self.columns) * 0.3)))

            sns.heatmap(_loadings, 
                        xticklabels=[f'PC{i+1}' for i in range(n_components)],
                        yticklabels=self.columns,
                        annot=True, 
                        cmap='RdBu_r', 
                        center=0,
                        fmt='.3f')
            
            plt.title(f'{color_by.capitalize()} Component Loadings - {self.name}')
            plt.xlabel('Principal Components')
            plt.ylabel('Original Variables')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Component loadings plot saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()

    def plot_evolving_variance(self, show: bool = True,
                               save_path: Optional[str] = None,
                               label_file: Optional[str] = "crisis_labels.csv") -> None:
        """
        Plot how explained variance evolves over time across updates.
        
        Args:
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
            label_file: Path to crisis labels file.
        """
        if not self.update_history:
            raise ValueError("No update history available. Call fit() first.")
        
        with DataLogger(self.logger, "plot_evolving_variance"):
            labels = None
            try:
                labels = pd.read_csv(label_file, index_col=0)
            except FileNotFoundError:
                self.logger.warning(f"Label file {label_file} not found.")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

            for country in self.update_history.keys():
                updates = self.update_history[country]
                dates = self.update_dates[country]
                
                if not updates:
                    continue
                
                # Extract variance data over time
                first_pc_variance = [update['explained_variance_ratio'][0] for update in updates]
                total_variance_3pc = [np.sum(update['explained_variance_ratio'][:3]) for update in updates]
                
                # Extract crisis labels if available
                crisis_idx = []
                if labels is not None and country in labels.index:
                    crisis_years = labels.loc[country, 'Year'].values
                    crisis_idx = [i for i, d in enumerate(dates) if d.year in crisis_years]
                
                # Plot first PC variance over time
                if country == "United States":
                    ax1.plot(dates, first_pc_variance, 'r-', linewidth=2, label=f"{country}")
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [first_pc_variance[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax1.plot(dates, first_pc_variance, 'b-', linewidth=2, label=f"{country}")
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [first_pc_variance[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax1.plot(dates, first_pc_variance, 'k-', linewidth=0.25)
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [first_pc_variance[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
                
                # Plot cumulative variance of first 3 PCs
                if country == "United States":
                    ax2.plot(dates, total_variance_3pc, 'r-', linewidth=2, label=f"{country}")
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [total_variance_3pc[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax2.plot(dates, total_variance_3pc, 'b-', linewidth=2, label=f"{country}")
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [total_variance_3pc[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax2.plot(dates, total_variance_3pc, 'k-', linewidth=0.25)
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [total_variance_3pc[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
            
            ax1.set_ylabel('Explained Variance')
            ax1.set_title(f'Evolution of Variance of First PC - {self.name}\n{self.title}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Cumulative Explained Variance')
            ax2.set_title('Evolution of Cumulative Variance of First 3 PCs')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Evolving variance plot saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()

    def plot_eigenvalue_features(self, show: bool = True,
                                 save_path: Optional[str] = None,
                                 label_file: Optional[str] = "crisis_labels.csv") -> None:
        """
        Plot eigenvalue-based features evolution over time.
        
        Args:
            save_path: Optional path to save the plot.
            show: Whether to display the plot.
            label_file: Path to crisis labels file.
        """
        if not self.update_history:
            raise ValueError("No update history available. Call fit() first.")
        
        with DataLogger(self.logger, "plot_eigenvalue_features"):
            labels = None
            try:
                labels = pd.read_csv(label_file, index_col=0)
            except FileNotFoundError:
                self.logger.warning(f"Label file {label_file} not found.")
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

            for country in self.update_history.keys():
                updates = self.update_history[country]
                dates = self.update_dates[country]
                
                if not updates:
                    continue
                
                # Calculate eigenvalue features
                num_eigenvalues = []
                eigenvalue_ratio = []
                spectral_entropy = []
                
                for update in updates:
                    eigenvals = update['eigenvalues']
                    cumulative_var = update['cumulative_variance']
                    
                    # Number of eigenvalues needed for 95% variance
                    num_95 = np.sum(cumulative_var < 0.95) + 1
                    num_eigenvalues.append(min(num_95, len(eigenvals)))
                    
                    # Ratio of first eigenvalue to sum of all eigenvalues
                    ratio = eigenvals[0] / np.sum(eigenvals) if np.sum(eigenvals) > 0 else 0
                    eigenvalue_ratio.append(ratio)
                    
                    # Spectral entropy
                    normalized_eigenvals = eigenvals / np.sum(eigenvals) if np.sum(eigenvals) > 0 else eigenvals
                    entropy = -np.sum(normalized_eigenvals * np.log(normalized_eigenvals + 1e-10))
                    spectral_entropy.append(entropy)
                
                # Extract crisis labels if available
                crisis_idx = []
                if labels is not None and country in labels.index:
                    crisis_years = labels.loc[country, 'Year'].values
                    crisis_idx = [i for i, d in enumerate(dates) if d.year in crisis_years]
                
                # Plot number of eigenvalues for 95% variance
                if country == "United States":
                    ax1.plot(dates, num_eigenvalues, 'r-', linewidth=2, label=f"{country}")
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [num_eigenvalues[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax1.plot(dates, num_eigenvalues, 'b-', linewidth=2, label=f"{country}")
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [num_eigenvalues[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax1.plot(dates, num_eigenvalues, 'k-', linewidth=0.25)
                    ax1.scatter([dates[i] for i in crisis_idx],
                                [num_eigenvalues[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
                
                # Plot ratio of first eigenvalue to sum of all eigenvalues
                if country == "United States":
                    ax2.plot(dates, eigenvalue_ratio, 'r-', linewidth=2, label=f"{country}")
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [eigenvalue_ratio[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax2.plot(dates, eigenvalue_ratio, 'b-', linewidth=2, label=f"{country}")
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [eigenvalue_ratio[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax2.plot(dates, eigenvalue_ratio, 'k-', linewidth=0.25)
                    ax2.scatter([dates[i] for i in crisis_idx],
                                [eigenvalue_ratio[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
                
                # Plot spectral entropy of eigenvalues over time
                if country == "United States":
                    ax3.plot(dates, spectral_entropy, 'r-', linewidth=2, label=f"{country}")
                    ax3.scatter([dates[i] for i in crisis_idx],
                                [spectral_entropy[i] for i in crisis_idx],
                                color='r', s=50, marker='o', zorder=5)
                elif country == "Japan":
                    ax3.plot(dates, spectral_entropy, 'b-', linewidth=2, label=f"{country}")
                    ax3.scatter([dates[i] for i in crisis_idx],
                                [spectral_entropy[i] for i in crisis_idx],
                                color='b', s=50, marker='o', zorder=5)
                else:
                    ax3.plot(dates, spectral_entropy, 'k-', linewidth=0.25)
                    ax3.scatter([dates[i] for i in crisis_idx],
                                [spectral_entropy[i] for i in crisis_idx],
                                color='k', s=25, marker='o', zorder=5, alpha=0.5)
            
            ax1.set_ylabel('Number of Eigenvalues')
            ax1.set_title(f'Evolution of Number of Eigenvalues Required to Explain at least 95% of Variance - {self.name}\n{self.title}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.set_ylabel(r'$\lambda_1 : \sum \lambda_i$')
            ax2.set_title(f'Evolution of Ratio of First Eigenvalue to Sum of All Eigenvalues')
            ax2.grid(True, alpha=0.3)
            
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Spectral Entropy')
            ax3.set_title(f'Evolution of Spectral Entropy of Eigenvalues')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Eigenvalue features plot saved to {save_path}")
            
            if show:
                plt.show()
            else:
                plt.close()