import os
import pickle
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Union
import hashlib
import json
from pathlib import Path

from data.log_utilities import setup_logging, _find_project_root
from model.utilities import set_seed, set_device


def get_artifacts_path() -> Path:
    """
    Get the absolute path to the artifacts directory.
    
    Returns:
        Path object pointing to the artifacts directory
    """
    project_root = _find_project_root()
    return project_root / "artifacts"

def model_exists(model_name: str, model_dir: str = "artifacts") -> bool:
    """
    Check if a model with the given name already exists.
    
    Args:
        model_name: Name of the model to check
        model_dir: Directory to check (relative to project root)
        
    Returns:
        True if model exists, False otherwise
    """
    # Resolve path relative to project root
    if not os.path.isabs(model_dir):
        project_root = _find_project_root()
        model_path = project_root / model_dir
    else:
        model_path = Path(model_dir)
    
    model_file = model_path / f"{model_name}.pkl"
    return model_file.exists()

def load_or_train_model(model_class, params: Dict[str, Any], 
                        train_data: pd.DataFrame, train_labels: np.array,
                        model_dir: str = "artifacts", 
                        force_retrain: bool = False) -> Any:
    """
    Load an existing model if it exists, otherwise train a new one.
    
    Args:
        model_class: Class of the model to train
        params: Parameters for the model
        train_data: Training data
        train_labels: Training labels
        model_dir: Directory to save/load models
        force_retrain: If True, retrain even if model exists
        
    Returns:
        Trained model object
    """
    logger = setup_logging()
    
    # Generate model name from parameters
    model_type = model_class.__name__
    model_name = generate_model_name(model_type, params)
    
    # Check if model exists and load it
    if not force_retrain and model_exists(model_name, model_dir):
        logger.info(f"Loading existing model: {model_name}")
        try:
            # Resolve path relative to project root
            if not os.path.isabs(model_dir):
                project_root = _find_project_root()
                model_path = project_root / model_dir / f"{model_name}.pkl"
            else:
                model_path = Path(model_dir) / f"{model_name}.pkl"
            
            model, metadata = load_trained_model(str(model_path), load_metadata=True)
            logger.info(f"Successfully loaded existing model from {model_path}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load existing model: {e}. Training new model.")
    
    # Train new model
    logger.info(f"Training new model: {model_name}")
    
    # Handle different model initialization patterns
    try:
        # First try: TimeSeriesVAE and TimeSeriesUNET pattern (takes params dict)
        if model_type in ['TimeSeriesVAE', 'TimeSeriesUNET']:
            seed = set_seed()
            device = set_device()
            model = model_class(params=params, seed=seed, device=device)
        # Second try: Standard pattern (unpacked parameters)
        else:
            model = model_class(**params)
    except TypeError as e:
        # Third try: If unpacking fails, try with params as dict
        if "unexpected keyword argument" in str(e):
            try:
                model = model_class(params=params)
            except TypeError:
                # Fourth try: For models that need specific argument patterns
                if model_type == 'BaseVAE':
                    # Extract specific parameters for BaseVAE
                    vae_params = {k: v for k, v in params.items() 
                                 if k in ['n_components', 'hidden_dims', 'learning_rate', 
                                         'batch_size', 'n_epochs', 'beta', 'seed', 'device']}
                    model = model_class(**vae_params)
                else:
                    raise e
        else:
            raise e
    
    model.fit(train_data, train_labels)
    
    # Save the trained model
    save_trained_model(
        model=model, 
        model_name=model_name, 
        save_dir=model_dir,
        metadata={'training_params': params}
    )
    
    logger.info(f"Successfully trained and saved new model: {model_name}")
    return model

def generate_model_name(model_type: str, params: Dict[str, Any], 
                       timestamp: bool = False, hash_params: bool = False) -> str:
    """
    Generate a standardized model name based on model type and parameters.
    
    Args:
        model_type: Type of model (e.g., 'VAE', 'RecurrentVAE', 'UNET', 'UMAP')
        params: Dictionary of model parameters
        timestamp: Whether to include timestamp in the name
        hash_params: Whether to include parameter hash for uniqueness
        
    Returns:
        Generated model name string
    """
    # Base name with model type
    name_parts = [model_type]
    
    # Add key parameters to name
    key_params = []
    if 'n_components' in params:
        key_params.append(f"comp{params['n_components']}")
    if 'sequence_length' in params:
        key_params.append(f"seq{params['sequence_length']}")
    if 'hidden_dim' in params or 'hidden_size' in params:
        hidden = params.get('hidden_dim', params.get('hidden_size'))
        key_params.append(f"hid{hidden}")
    if 'num_layers' in params:
        key_params.append(f"lay{params['num_layers']}")
    if 'cell_type' in params or 'block' in params:
        cell = params.get('cell_type', params.get('block', ''))
        key_params.append(f"{cell.lower()}")
    if 'learning_rate' in params:
        lr = params['learning_rate']
        key_params.append(f"lr{lr:.0e}".replace('e-0', 'e-'))
    if 'batch_size' in params:
        key_params.append(f"bs{params['batch_size']}")
    if 'n_epochs' in params:
        key_params.append(f"ep{params['n_epochs']}")
    if 'beta' in params and params['beta'] != 1.0:
        key_params.append(f"beta{params['beta']}")
    if 'dropout_rate' in params:
        dropout = params['dropout_rate']
        key_params.append(f"drop{dropout:.2f}".replace('.', 'p'))
    if 'lstm_units' in params:
        key_params.append(f"lstm{params['lstm_units']}")
    if 'num_lstm_layers' in params:
        key_params.append(f"lstm_layers{params['num_lstm_layers']}")
    
    if key_params:
        name_parts.append('_'.join(key_params))
    
    # Add parameter hash for uniqueness
    if hash_params:
        param_str = json.dumps(params, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        name_parts.append(f"hash{param_hash}")
    
    # Add timestamp
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts.append(timestamp_str)
    
    return '_'.join(name_parts)

def save_trained_model(model: Any, model_name: str, save_dir: str = "artifacts",
                       metadata: Optional[Dict[str, Any]] = None,
                       save_state_dict_only: bool = False) -> str:
    """
    Save a trained deep learning model with metadata.
    
    Args:
        model: Trained model object (VAE, RecurrentVAE, etc.)
        model_name: Name for the saved model
        save_dir: Directory to save the model (relative to project root)
        metadata: Additional metadata to save with the model
        save_state_dict_only: Whether to save only state dict (for PyTorch models)
        
    Returns:
        Path to the saved model file
    """
    logger = setup_logging()
    
    # Resolve path relative to project root (assuming this script is in src/features/)
    if not os.path.isabs(save_dir):
        project_root = _find_project_root()
        save_path = project_root / save_dir
    else:
        save_path = Path(save_dir)
    
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    metadata.update({
        'model_name': model_name,
        'save_timestamp': datetime.now().isoformat(),
        'model_type': model.__class__.__name__,
    })
    
    # Add model-specific metadata
    if hasattr(model, 'get_component_names'):
        try:
            metadata['component_names'] = model.get_component_names()
        except:
            pass
    
    if hasattr(model, 'n_components'):
        metadata['n_components'] = model.n_components
    
    if hasattr(model, 'sequence_length'):
        metadata['sequence_length'] = model.sequence_length
    
    if hasattr(model, 'scaler') and model.scaler is not None:
        metadata['has_scaler'] = True
    
    if hasattr(model, 'feature_cols'):
        metadata['feature_columns'] = getattr(model, 'feature_cols', None)
    
    if hasattr(model, 'numeric_columns'):
        metadata['numeric_columns'] = getattr(model, 'numeric_columns', None)
    
    # Save the model
    model_file = save_path / f"{model_name}.pkl"
    
    try:
        # Handle PyTorch models differently
        if hasattr(model, 'state_dict') and save_state_dict_only:
            # Save state dict for PyTorch models
            torch_file = save_path / f"{model_name}_state_dict.pth"
            torch.save(model.state_dict(), torch_file)
            metadata['torch_state_dict_file'] = str(torch_file)
            logger.info(f"Saved PyTorch model state dict to: {torch_file}")
        
        # Save the complete model object
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata separately
        metadata_file = save_path / f"{model_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Successfully saved model to: {model_file}")
        logger.info(f"Saved metadata to: {metadata_file}")
        
        return str(model_file)
        
    except Exception as e:
        logger.error(f"Error saving model {model_name}: {str(e)}")
        raise

def load_trained_model(model_path: str, load_metadata: bool = True,
                      device: Optional[str] = None) -> Union[Any, tuple]:
    """
    Load a trained deep learning model with optional metadata.
    
    Args:
        model_path: Path to the saved model file
        load_metadata: Whether to also load and return metadata
        device: Device to load PyTorch models on
        
    Returns:
        Loaded model object, or tuple of (model, metadata) if load_metadata=True
    """
    logger = setup_logging()
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Set device for PyTorch models
        if hasattr(model, 'device') and device is not None:
            model.device = torch.device(device)
            if hasattr(model, 'to'):
                model = model.to(device)
        
        logger.info(f"Successfully loaded model from: {model_path}")
        
        # Load metadata if requested
        metadata = None
        if load_metadata:
            metadata_file = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata from: {metadata_file}")
            else:
                logger.warning(f"Metadata file not found: {metadata_file}")
                metadata = {}
        
        if load_metadata:
            return model, metadata
        else:
            return model
            
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {str(e)}")
        raise

def save_model_results(model: Any, train_data: pd.DataFrame, test_data: pd.DataFrame,
                      train_labels: np.array, test_labels: np.array,
                      results_dir: str = "artifacts") -> Dict[str, str]:
    """
    Save model training results including transformed data and training history.
    
    Args:
        model: Trained model object
        train_data: Training data DataFrame
        test_data: Test data DataFrame  
        train_labels: Training labels
        test_labels: Test labels
        results_dir: Directory to save results (relative to project root)
        
    Returns:
        Dictionary with paths to saved files
    """
    logger = setup_logging()
    
    # Resolve path relative to project root (assuming this script is in src/features/)
    if not os.path.isabs(results_dir):
        project_root = _find_project_root()
        results_path = project_root / results_dir
    else:
        results_path = Path(results_dir)
    
    results_path.mkdir(parents=True, exist_ok=True)
    
    model_name = getattr(model, 'model_name', model.__class__.__name__)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    saved_files = {}
    
    try:
        # Transform and save training data
        train_transformed = model.transform(train_data, train_labels)
        train_file = results_path / f"{model_name}_train_transformed_{timestamp}.csv"
        train_transformed.to_csv(train_file)
        saved_files['train_transformed'] = str(train_file)
        
        # Transform and save test data
        test_transformed = model.transform(test_data, test_labels)
        test_file = results_path / f"{model_name}_test_transformed_{timestamp}.csv"
        test_transformed.to_csv(test_file)
        saved_files['test_transformed'] = str(test_file)
        
        # Save training history if available
        if hasattr(model, 'train_losses') and model.train_losses:
            history_data = {
                'train_losses': model.train_losses,
                'recon_losses': getattr(model, 'recon_losses', []),
                'kl_losses': getattr(model, 'kl_losses', [])
            }
            history_file = results_path / f"{model_name}_training_history_{timestamp}.json"
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            saved_files['training_history'] = str(history_file)
        
        # Save model parameters
        if hasattr(model, '__dict__'):
            params = {}
            for key, value in model.__dict__.items():
                if not key.startswith('_') and not callable(value):
                    try:
                        # Only save serializable parameters
                        json.dumps(value, default=str)
                        params[key] = value
                    except:
                        params[key] = str(value)
            
            params_file = results_path / f"{model_name}_parameters_{timestamp}.json"
            with open(params_file, 'w') as f:
                json.dump(params, f, indent=2, default=str)
            saved_files['parameters'] = str(params_file)
        
        logger.info(f"Saved model results to {results_dir}")
        return saved_files
        
    except Exception as e:
        logger.error(f"Error saving model results: {str(e)}")
        raise

def load_model_results(results_dir: str, model_name: str, 
                       timestamp: Optional[str] = None) -> Dict[str, Any]:
    """
    Load previously saved model results.
    
    Args:
        results_dir: Directory containing saved results (relative to project root)
        model_name: Name of the model
        timestamp: Specific timestamp to load (optional, loads latest if None)
        
    Returns:
        Dictionary containing loaded results
    """
    logger = setup_logging()
    
    # Resolve path relative to project root (assuming this script is in src/features/)
    if not os.path.isabs(results_dir):
        project_root = _find_project_root()
        results_path = project_root / results_dir
    else:
        results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")
    
    # Find matching files
    pattern = f"{model_name}_*"
    if timestamp:
        pattern = f"{model_name}_*{timestamp}*"
    
    matching_files = list(results_path.glob(pattern))
    
    if not matching_files:
        raise FileNotFoundError(f"No results found for model {model_name} in {results_dir}")
    
    results = {}
    
    try:
        # Load transformed data
        train_files = [f for f in matching_files if 'train_transformed' in f.name]
        if train_files:
            latest_train = max(train_files, key=os.path.getctime)
            results['train_transformed'] = pd.read_csv(latest_train, index_col=0)
            logger.info(f"Loaded training data from: {latest_train}")
        
        test_files = [f for f in matching_files if 'test_transformed' in f.name]
        if test_files:
            latest_test = max(test_files, key=os.path.getctime)
            results['test_transformed'] = pd.read_csv(latest_test, index_col=0)
            logger.info(f"Loaded test data from: {latest_test}")
        
        # Load training history
        history_files = [f for f in matching_files if 'training_history' in f.name]
        if history_files:
            latest_history = max(history_files, key=os.path.getctime)
            with open(latest_history, 'r') as f:
                results['training_history'] = json.load(f)
            logger.info(f"Loaded training history from: {latest_history}")
        
        # Load parameters
        param_files = [f for f in matching_files if 'parameters' in f.name]
        if param_files:
            latest_params = max(param_files, key=os.path.getctime)
            with open(latest_params, 'r') as f:
                results['parameters'] = json.load(f)
            logger.info(f"Loaded parameters from: {latest_params}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading model results: {str(e)}")
        raise