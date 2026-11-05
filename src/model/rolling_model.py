import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from data.data_utilities import build_labels

class BaseModel:
    def __init__(self, predictions, predictions_proba):
        self.predictions = predictions
        self.predictions_proba = predictions_proba
    
    def predict(self, X):
        return self.predictions
    
    def predict_proba(self, X):
        return np.column_stack([1 - self.predictions_proba, self.predictions_proba])
    
class RollingWindowModel:
    """
    Rolling window model training and testing framework for financial crisis prediction.
    
    This class implements a time-series cross-validation approach where:
    1. Train on historical data from train_df
    2. Test on next 12 months from test_df
    3. Retrain with updated data (original train_df + 1 year from test_df)
    4. Test on next 12 months, and so on...
    
    Ensures no data leakage by strictly maintaining temporal order.
    """
    
    def __init__(self, 
                 model_class: type,
                 model_params: Dict[str, Any],
                 test_window_months: int = 12,
                 retrain_window_months: int = 12):
        """
        Initialize the rolling window model.
        
        Args:
            model_class: The sklearn model class (e.g., XGBClassifier)
            model_params: Parameters to pass to the model constructor
            test_window_months: Number of months to test on in each window (default: 12)
            retrain_window_months: Number of months to add to training data after each test (default: 12)
        """
        self.model_class = model_class
        self.model_params = model_params
        self.test_window_months = test_window_months
        self.retrain_window_months = retrain_window_months
        
        # Store results and models from each window
        self.window_results = []
        self.window_models = []
        self.feature_importances_history = []
        self.current_model = None
        
        # Combined results for plotting
        self.all_predictions = []
        self.all_true_labels = []
        self.all_prediction_data = []
        
    def _prepare_data(self, df: pd.DataFrame, y_labels: np.ndarray = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data by removing non-feature columns and using provided labels or building them.
        
        Args:
            df: DataFrame with Country, Date, and feature columns
            y_labels: Optional pre-computed labels array. If None, will try to build labels.
            
        Returns:
            Tuple of (features_df, labels_array)
        """
        # Use provided labels or build them
        if y_labels is not None:
            y = y_labels
        else:
            y = build_labels(df)
        
        # Remove non-feature columns
        feature_columns = [col for col in df.columns if col not in ['Country', 'Date']]
        X = df[feature_columns].copy()
        
        return X, y
    
    def _get_date_windows(self, test_df: pd.DataFrame) -> List[Tuple[datetime, datetime]]:
        """
        Generate date windows for rolling validation.
        
        Args:
            test_df: Test dataframe to determine date range
            
        Returns:
            List of (start_date, end_date) tuples for each test window
        """
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        min_date = test_df['Date'].min()
        max_date = test_df['Date'].max()
        
        windows = []
        current_start = min_date
        
        while current_start < max_date:
            current_end = current_start + relativedelta(months=self.test_window_months)
            if current_end > max_date:
                current_end = max_date
            
            windows.append((current_start, current_end))
            current_start += relativedelta(months=self.retrain_window_months)
            
        return windows
    
    def _filter_by_date_range(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Filter dataframe by date range.
        
        Args:
            df: DataFrame with Date column
            start_date: Start date (inclusive)
            end_date: End date (exclusive)
            
        Returns:
            Filtered DataFrame
        """
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        mask = (df_copy['Date'] >= start_date) & (df_copy['Date'] < end_date)
        return df_copy[mask]
    
    def fit_predict_rolling(self, 
                           train_df: pd.DataFrame, 
                           test_df: pd.DataFrame,
                           train_labels: np.ndarray = None,
                           test_labels: np.ndarray = None,
                           verbose: bool = True) -> Dict[str, Any]:
        """
        Perform rolling window training and testing.
        
        Args:
            train_df: Initial training data
            test_df: Data to use for rolling testing and retraining
            train_labels: Optional pre-computed labels for train_df
            test_labels: Optional pre-computed labels for test_df
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing aggregated results and predictions
        """
        # Reset results
        self.window_results = []
        self.window_models = []
        self.feature_importances_history = []
        self.all_predictions = []
        self.all_true_labels = []
        self.all_prediction_data = []
        
        # Ensure Date columns are datetime
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['Date'] = pd.to_datetime(train_df['Date'])
        test_df['Date'] = pd.to_datetime(test_df['Date'])
        
        # Get date windows
        date_windows = self._get_date_windows(test_df)
        
        if verbose:
            print(f"Starting rolling window validation with {len(date_windows)} windows")
            print(f"Test window: {self.test_window_months} months")
            print(f"Retrain window: {self.retrain_window_months} months")
        
        # Current training data starts with the original train_df
        current_train_df = train_df.copy()
        current_train_labels = train_labels.copy() if train_labels is not None else None
        
        for window_idx, (window_start, window_end) in enumerate(date_windows):
            if verbose:
                print(f"\nWindow {window_idx + 1}/{len(date_windows)}: {window_start.date()} to {window_end.date()}")
            
            # Get test data for this window
            window_test_df = self._filter_by_date_range(test_df, window_start, window_end)
            
            if len(window_test_df) == 0:
                if verbose:
                    print(f"No data in window {window_idx + 1}, skipping...")
                continue
            
            # Get corresponding test labels for this window
            if test_labels is not None:
                # Find the indices of the window_test_df in the original test_df
                test_df_reset = test_df.reset_index(drop=True)
                window_test_df_reset = window_test_df.reset_index(drop=True)
                
                # Merge to find matching indices
                merged = test_df_reset.merge(window_test_df_reset[['Country', 'Date']], 
                                           on=['Country', 'Date'], how='inner')
                window_test_indices = merged.index
                window_test_labels = test_labels[window_test_indices]
            else:
                window_test_labels = None
            
            # Prepare training data
            X_train, y_train = self._prepare_data(current_train_df, current_train_labels)
            
            # Prepare test data
            X_test, y_test = self._prepare_data(window_test_df, window_test_labels)
            
            if verbose:
                print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
                print(f"Training crisis rate: {y_train.mean():.2%}")
                print(f"Test crisis rate: {y_test.mean():.2%}")
            
            # Train model
            model = self.model_class(**self.model_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'window': window_idx + 1,
                'start_date': window_start,
                'end_date': window_end,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'train_crisis_rate': y_train.mean(),
                'test_crisis_rate': y_test.mean(),
                'f1_score': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
            }
            
            # Store results
            self.window_results.append(metrics)
            self.window_models.append(model)
            
            # Store feature importances if available
            if hasattr(model, 'feature_importances_'):
                feature_importance_dict = dict(zip(X_train.columns, model.feature_importances_))
                self.feature_importances_history.append(feature_importance_dict)
            
            # Store predictions for later plotting
            window_test_df_copy = window_test_df.copy()
            window_test_df_copy['y_true'] = y_test
            window_test_df_copy['y_pred'] = y_pred
            window_test_df_copy['y_pred_proba'] = y_pred_proba
            window_test_df_copy['window'] = window_idx + 1
            
            self.all_prediction_data.append(window_test_df_copy)
            self.all_predictions.extend(y_pred)
            self.all_true_labels.extend(y_test)
            
            if verbose:
                print(f"F1: {metrics['f1_score']:.3f}, Precision: {metrics['precision']:.3f}, "
                      f"Recall: {metrics['recall']:.3f}, ROC-AUC: {metrics['roc_auc']:.3f}")
            
            # Update training data for next window by adding retrain_window_months of data
            retrain_start = window_start
            retrain_end = window_start + relativedelta(months=self.retrain_window_months)
            retrain_data = self._filter_by_date_range(test_df, retrain_start, retrain_end)
            
            if len(retrain_data) > 0:
                # Get corresponding labels for retrain data
                if test_labels is not None:
                    retrain_merged = test_df_reset.merge(retrain_data[['Country', 'Date']], 
                                                       on=['Country', 'Date'], how='inner')
                    retrain_indices = retrain_merged.index
                    retrain_labels = test_labels[retrain_indices]
                    
                    # Combine training data and labels
                    current_train_df = pd.concat([current_train_df, retrain_data], ignore_index=True)
                    if current_train_labels is not None:
                        current_train_labels = np.concatenate([current_train_labels, retrain_labels])
                    else:
                        current_train_labels = retrain_labels
                else:
                    current_train_df = pd.concat([current_train_df, retrain_data], ignore_index=True)
                
                # Remove duplicates based on Country and Date
                if current_train_labels is not None:
                    # Create a combined dataframe for deduplication
                    train_with_labels = current_train_df.copy()
                    train_with_labels['_labels'] = current_train_labels
                    train_with_labels = train_with_labels.drop_duplicates(subset=['Country', 'Date'], keep='last')
                    train_with_labels = train_with_labels.sort_values(['Country', 'Date']).reset_index(drop=True)
                    
                    current_train_df = train_with_labels.drop(columns=['_labels'])
                    current_train_labels = train_with_labels['_labels'].values
                else:
                    current_train_df = current_train_df.drop_duplicates(subset=['Country', 'Date'], keep='last')
                    current_train_df = current_train_df.sort_values(['Country', 'Date']).reset_index(drop=True)
                
                if verbose:
                    print(f"Added {len(retrain_data)} samples to training data (total: {len(current_train_df)})")
        
        # Set the last trained model as current
        if self.window_models:
            self.current_model = self.window_models[-1]
        
        # Combine all prediction data
        if self.all_prediction_data:
            self.combined_predictions_df = pd.concat(self.all_prediction_data, ignore_index=True)
        else:
            self.combined_predictions_df = pd.DataFrame()
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics()
        
        return {
            'window_results': self.window_results,
            'overall_metrics': overall_metrics,
            'predictions_df': self.combined_predictions_df,
            'feature_importances_history': self.feature_importances_history
        }
    
    def _calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall metrics across all windows."""
        if not self.all_true_labels:
            return {}
        
        y_true = np.array(self.all_true_labels)
        y_pred = np.array(self.all_predictions)
        
        return {
            'overall_f1': f1_score(y_true, y_pred),
            'overall_precision': precision_score(y_true, y_pred, zero_division=0),
            'overall_recall': recall_score(y_true, y_pred, zero_division=0),
            'overall_samples': len(y_true),
            'overall_crisis_rate': y_true.mean(),
            'overall_predicted_crisis_rate': y_pred.mean()
        }
    
    def get_feature_importances(self, model_idx: int = -1, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get feature importances from a model in the rolling window framework.
        
        Args:
            model_idx: Index of the model to get feature importances from
            top_n: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        if model_idx < -1 or model_idx > len(self.feature_importances_history):
            raise ValueError(f"Model index must be > -1 and < number of windows trained. Received {model_idx}.")
        if not self.feature_importances_history:
            return []
        
        latest_importances = self.feature_importances_history[model_idx]
        sorted_features = sorted(latest_importances.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features[:top_n]
    
    def predict(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the most recent model.
        
        Args:
            X_test: Test features (should contain same columns as training data)
            
        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if self.current_model is None:
            raise ValueError("No model has been trained yet. Call fit_predict_rolling first.")
        
        # Prepare features (remove non-feature columns if present)
        feature_columns = [col for col in X_test.columns if col not in ['Country', 'Date', 'y_true', 'y_pred', 'y_pred_proba', 'window']]
        X_features = X_test[feature_columns]
        
        y_pred = self.current_model.predict(X_features)
        y_pred_proba = self.current_model.predict_proba(X_features)[:, 1]
        
        return y_pred, y_pred_proba
    
    def get_predictions_for_plotting(self) -> pd.DataFrame:
        """
        Get predictions DataFrame formatted for use with plot_metrics and plot_prediction_timeline.
        
        Returns:
            DataFrame with predictions that can be used with existing plotting functions
        """
        return self.combined_predictions_df
    
    def print_summary(self) -> None:
        """Print a summary of the rolling window results."""
        if not self.window_results:
            print("No results available. Run fit_predict_rolling first.")
            return
        
        print("\n" + "="*80)
        print("ROLLING WINDOW MODEL SUMMARY")
        print("="*80)
        
        # Overall metrics
        overall_metrics = self._calculate_overall_metrics()
        if overall_metrics:
            print(f"\nOverall Performance:")
            print(f"  Total samples: {overall_metrics['overall_samples']}")
            print(f"  Crisis rate: {overall_metrics['overall_crisis_rate']:.2%}")
            print(f"  Predicted crisis rate: {overall_metrics['overall_predicted_crisis_rate']:.2%}")
            print(f"  F1 Score: {overall_metrics['overall_f1']:.3f}")
            print(f"  Precision: {overall_metrics['overall_precision']:.3f}")
            print(f"  Recall: {overall_metrics['overall_recall']:.3f}")
        
        # Window-by-window results
        print(f"\nWindow-by-Window Results:")
        print("-" * 120)
        print(f"{'Window':<8} {'Period':<25} {'Train Samples':<12} {'Test Samples':<12} {'F1':<8} {'Precision':<10} {'Recall':<8} {'ROC-AUC':<8}")
        print("-" * 120)
        
        for result in self.window_results:
            period = f"{result['start_date'].strftime('%Y-%m')} to {result['end_date'].strftime('%Y-%m')}"
            print(f"{result['window']:<8} {period:<25} {result['train_samples']:<12} {result['test_samples']:<12} "
                  f"{result['f1_score']:<8.3f} {result['precision']:<10.3f} {result['recall']:<8.3f} {result['roc_auc']:<8.3f}")
        
        # Top features from first and last model
        top_features = self.get_feature_importances(model_idx=0, top_n=10)
        if top_features:
            print(f"\nTop 10 Features (from first model):")
            print("-" * 50)
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"{i:2d}. {feature:<35} {importance:.4f}")
        top_features = self.get_feature_importances(model_idx=-1, top_n=10)
        if top_features:
            print(f"\nTop 10 Features (from last model):")
            print("-" * 50)
            for i, (feature, importance) in enumerate(top_features, 1):
                print(f"{i:2d}. {feature:<35} {importance:.4f}")
        
        print("="*80)


class RollingWindowModelAdapter:
    """
    Adapter class to make RollingWindowModel compatible with plot_metrics and plot_prediction_timeline.
    This class wraps the RollingWindowModel to provide sklearn-like interface for the plotting functions.
    """
    
    def __init__(self, rolling_model: RollingWindowModel):
        """
        Initialize adapter with a trained RollingWindowModel.
        
        Args:
            rolling_model: A trained RollingWindowModel instance
        """
        self.rolling_model = rolling_model
        self._predictions_df = rolling_model.get_predictions_for_plotting()
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict method for compatibility with plot_metrics.
        
        For rolling window models, we return the stored predictions rather than
        making new predictions, since the test data should match what was used
        in the rolling validation.
        """
        # If X_test contains the same data as stored predictions, return those
        if not self._predictions_df.empty:
            # Try to match based on Country and Date
            if 'Country' in X_test.columns and 'Date' in X_test.columns:
                # Check if the y_pred column exists in predictions_df
                if 'y_pred' in self._predictions_df.columns:
                    # Merge to get predictions for the provided test data
                    merged = X_test.merge(
                        self._predictions_df[['Country', 'Date', 'y_pred']], 
                        on=['Country', 'Date'], 
                        how='left'
                    )
                    if 'y_pred' in merged.columns and not merged['y_pred'].isna().all():
                        return merged['y_pred'].fillna(0).values
                
                # If we have direct y_pred column in X_test
                if 'y_pred' in X_test.columns:
                    return X_test['y_pred'].values
        
        # Fall back to using the current model
        return self.rolling_model.predict(X_test)[0]
    
    def predict_proba(self, X_test: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities method for compatibility with plot_metrics.
        """
        # If X_test contains the same data as stored predictions, return those
        if not self._predictions_df.empty:
            # Try to match based on Country and Date
            if 'Country' in X_test.columns and 'Date' in X_test.columns:
                # Check if the y_pred_proba column exists in predictions_df
                if 'y_pred_proba' in self._predictions_df.columns:
                    # Merge to get predictions for the provided test data
                    merged = X_test.merge(
                        self._predictions_df[['Country', 'Date', 'y_pred_proba']], 
                        on=['Country', 'Date'], 
                        how='left'
                    )
                    if 'y_pred_proba' in merged.columns and not merged['y_pred_proba'].isna().all():
                        proba_positive = merged['y_pred_proba'].fillna(0.5).values
                        # Return in sklearn format: [[prob_negative, prob_positive], ...]
                        return np.column_stack([1 - proba_positive, proba_positive])
                
                # If we have direct y_pred_proba column in X_test
                if 'y_pred_proba' in X_test.columns:
                    proba_positive = X_test['y_pred_proba'].values
                    return np.column_stack([1 - proba_positive, proba_positive])
        
        # Fall back to using the current model
        _, proba_positive = self.rolling_model.predict(X_test)
        return np.column_stack([1 - proba_positive, proba_positive])
