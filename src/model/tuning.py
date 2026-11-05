import numpy as np
import pandas as pd
from typing import Dict, Any
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from features.utilities import load_or_train_model

def stratified_gridsearch(param_grid: dict,
                          base_model,
                          X_train, y_train,
                          scoring='f1_weighted') -> GridSearchCV:
    
    X, _, y, _ = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
        )
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv_folds, scoring=scoring, verbose=1
        )
    grid_search.fit(X, y)
    return grid_search

def objective_function(embed_model, classify_model, train_data: pd.DataFrame, train_labels: np.ndarray, 
                       **kwargs) -> float:
    """
    Objective function for Bayesian optimization.
    
    Args:
        embed_model: The embedding model class to optimize (e.g., VAE, TimeSeriesVAE, TimeSeriesUNET)
        classify_model: The classification model class to optimize (e.g. Random Forest, XGBoost, LightGBM)
        train_data: Training data
        train_labels: Training labels
        test_data: Test data
        test_labels: Test labels
        **kwargs: Hyperparameters to optimize
    
    Returns:
        Negative total loss (to maximize)
    """
    # Extract embedding model hyperparamters
    params = {
        'n_components': int(kwargs['n_components']),
        'batch_size': int(kwargs['batch_size']),
        'beta': kwargs['beta'],
        'learning_rate': kwargs['learning_rate'],
        'n_epochs': 25, # Fixed for faster optimization
    }
    if 'hidden_size' in kwargs:
        params['hidden_size'] = int(kwargs['hidden_size'])
    
    if 'num_layers' in kwargs:
        params['num_layers'] = int(kwargs['num_layers'])

    if 'hidden_dim1' in kwargs:
        hidden_dims = [int(kwargs['hidden_dim1'])]
        if 'hidden_dim2' in kwargs:
            hidden_dims.append(int(kwargs['hidden_dim2']))
        if 'hidden_dim3' in kwargs:
            hidden_dims.append(int(kwargs['hidden_dim3']))
        params['hidden_dims'] = hidden_dims
    
    if 'sequence_length' in kwargs:
        params['sequence_length'] = int(kwargs['sequence_length'])
    
    if 'KL_annealing_int' in kwargs:
        params['KL_annealing'] = bool(int(kwargs['KL_annealing_int']))
        params['warmup_epochs'] = int(kwargs['warmup_epochs']) if params['KL_annealing'] else 10
        annealing_modes = ['linear', 'cosine', 'cyclical']
        params['annealing_mode'] = annealing_modes[int(kwargs['annealing_mode_int']) % len(annealing_modes)]
    
    if 'block_int' in kwargs:
        blocks = ['LSTM', 'GRU']
        params['block'] = blocks[int(kwargs['block_int']) % len(blocks)]
    
    # Extract classification model parameter grid
    if "RandomForestClassifier" in str(type(classify_model)):
        param_grid = {
            'n_estimators': np.random.randint(50, 200, 5),
            'max_depth': np.random.randint(3, 10, 3),
            # 'min_samples_split': np.random.uniform(0.0, 1.0, 3),
            # 'min_samples_leaf': np.random.uniform(0.1, 1.0, 3),
            # 'max_features': np.random.choice(["sqrt", "log2"], 2),
        }
    elif "XGBClassifier" in str(type(classify_model)):
        param_grid = {
            'n_estimators': np.random.randint(50, 200, 5),
            'max_depth': np.random.randint(3, 10, 3),
        }
    elif "LGBMClassifier" in str(type(classify_model)):
        param_grid = {
            'n_estimators': np.random.randint(50, 200, 5),
            'max_depth': np.random.randint(3, 10, 3),
        }
    else:
        raise ValueError(f"Unsupported classifier type: {type(classify_model)}")
    
    try:
        # Train embedder
        embedder = load_or_train_model(
            model_class=embed_model,
            params=params,
            train_data=train_data,
            train_labels=train_labels,
            force_retrain=True
        )
        # Train classifier
        grid_search = stratified_gridsearch(
            param_grid=param_grid,
            base_model=classify_model,
            X_train=embedder.transform(train_data),
            y_train=train_labels,
            scoring='roc_auc'
        )
        
        print(f"Best classifier parameters: {grid_search.best_params_}")
        print(f"Best classifier score: {grid_search.best_score_:.4f}")

        return -grid_search.best_score_
        
    except Exception as e:
        print(f"Error during model training: {e}")
        return -1e6  # Return very bad score for failed runs

def run_bayesian_search(embed_model, classify_model, train_data: pd.DataFrame, train_labels: np.ndarray, 
                        n_iter: int = 25, init_points: int = 5, seed: int = 42) -> Dict[str, Any]:
    """
    Run Bayesian optimization for hyperparameter tuning.
    
    Args:
        embed_model: The embedding model class to optimize (e.g., VAE, TimeSeriesVAE, TimeSeriesUNET)
        classify_model: The classification model class to optimize (e.g. Random Forest, XGBoost, LightGBM)
        train_data: Training data
        train_labels: Training labels
        n_iter: Number of optimization iterations
        init_points: Number of random initialization points
        seed: Random seed
    
    Returns:
        Dictionary with best parameters and results
    """
    
    # TimeSeriesVAE and TimeSeriesUNET search space
    if embed_model.__name__ in ['TimeSeriesVAE', 'TimeSeriesUNET']:
        pbounds = {
            'hidden_size': (32, 256),
            'n_components': (2, 32),
            'num_layers': (1, 3),
            'batch_size': (8, 32),
            'beta': (1.0, 1.0),
            'learning_rate': (1e-3, 1e-3),
            'sequence_length': (12, 48),
            'KL_annealing_int': (0, 0),  # 0=False, 1=True
            'warmup_epochs': (0, 0),
            'annealing_mode_int': (0, 0),  # 0=linear, 1=cosine, 2=cyclical
            'block_int': (0, 1)  # 0=LSTM, 1=GRU
        }
    elif embed_model.__name__ == 'BaseVAE':
        # VAE search space
        pbounds = {
            'hidden_dim1': (32, 128),   # First hidden layer
            'hidden_dim2': (16, 96),    # Second hidden layer
            'hidden_dim3': (8, 64),     # Third hidden layer
            'n_components': (2, 10),
            'num_layers': (1, 3),
            'batch_size': (16, 128),
            'beta': (0.5, 3.0),
            'learning_rate': (1e-5, 1e-3),
        }
    elif embed_model.__name__ == 'LSTMClassifier':
        # # LSTMClassifier search space
        # pbounds = {
        #     'hidden_size': (32, 256),
        #     'n_components': (8, 48),
        #     'num_layers': (1, 3),
        #     'batch_size': (16, 128),
        #     'beta': (1.0, 50.0),
        #     'learning_rate': (1e-5, 1e-3),
        #     'sequence_length': (6, 48),
        #     'KL_annealing_int': (0, 1),  # 0=False, 1=True
        #     'warmup_epochs': (0, 20),
        #     'annealing_mode_int': (0, 2),  # 0=linear, 1=cosine, 2=cyclical
        #     'block_int': (0, 1)  # 0=LSTM, 1=GRU
        # }
        raise NotImplementedError("LSTMClassifier optimization not implemented yet")
    else:
        raise ValueError(f"Unsupported model class: {embed_model.__name__}")
    
    # Create objective function with fixed arguments
    def objective(**kwargs):
        return objective_function(embed_model, classify_model, train_data, train_labels, **kwargs)
    
    # Create a BayesianOptimization object
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=seed,
        allow_duplicate_points=True
    )
    
    # Run the optimization
    optimizer.maximize(
        init_points=init_points,
        n_iter=n_iter,
    )
    
    # Process results
    best_params = optimizer.max['params'].copy()
    
    # Convert back to proper types and handle categorical variables
    if 'hidden_size' in best_params:
        best_params['hidden_size'] = int(best_params['hidden_size'])
    if 'n_components' in best_params:
        best_params['n_components'] = int(best_params['n_components'])
    if 'num_layers' in best_params:
        best_params['num_layers'] = int(best_params['num_layers'])
    if 'batch_size' in best_params:
        best_params['batch_size'] = int(best_params['batch_size'])
    if 'sequence_length' in best_params:
        best_params['sequence_length'] = int(best_params['sequence_length'])
    
    if 'hidden_dim1' in best_params:
        hidden_dims = [int(best_params['hidden_dim1'])]
        if 'hidden_dim2' in best_params:
            hidden_dims.append(int(best_params['hidden_dim2']))
        if 'hidden_dim3' in best_params:
            hidden_dims.append(int(best_params['hidden_dim3']))
        best_params['hidden_dims'] = hidden_dims
        
        # Remove individual dimension parameters
        del best_params['hidden_dim1']
        if 'hidden_dim2' in best_params:
            del best_params['hidden_dim2']
        if 'hidden_dim3' in best_params:
            del best_params['hidden_dim3']
    
    # Handle categorical parameters
    if 'KL_annealing_int' in best_params:
        best_params['KL_annealing'] = bool(int(best_params['KL_annealing_int']))
        best_params['warmup_epochs'] = int(best_params['warmup_epochs'])
        
        annealing_modes = ['linear', 'cosine', 'cyclical']
        best_params['annealing_mode'] = annealing_modes[int(best_params['annealing_mode_int']) % len(annealing_modes)]
        
        # Remove the integer versions
        del best_params['KL_annealing_int']
        del best_params['annealing_mode_int']
    
    if 'block_int' in best_params:
        blocks = ['LSTM', 'GRU']
        best_params['block'] = blocks[int(best_params['block_int']) % len(blocks)]
        del best_params['block_int']
    
    # Log the results
    results_df = pd.DataFrame(optimizer.res)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'hyperparameter_results_{embed_model.__name__}_{timestamp}.csv'
    results_df.to_csv(results_filename, index=False)
    
    print(f"Optimization complete! Results saved to {results_filename}")
    print(f"Best score: {optimizer.max['target']:.4f}")
    print(f"Best parameters: {best_params}")
    
    return {
        'best_params': best_params,
        'best_score': optimizer.max['target'],
        'optimizer': optimizer,
        'results_df': results_df
    }