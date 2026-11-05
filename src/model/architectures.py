import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from model.dataset import BasicDataset, SequentialDataset

class FFNNClassifier(nn.Module):
    def __init__(self, params: dict = None, seed: int = 42, device: str = None):
        """Initialize the FNN Classifier with the same interface as LSTMClassifier."""
        super().__init__()
        
        # Set seed for reproducibility
        self.seed = seed
        self._set_seed(self.seed)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Default parameters
        default_params = {
            'hidden_dims': [256, 128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 1e-3,
            'batch_size': 128,
            'n_epochs': 100
        }
        
        # Update with provided parameters
        if params is not None:
            default_params.update(params)
        self.params = default_params
        
        # Extract parameters
        self.hidden_dims = self.params['hidden_dims']
        self.dropout_rate = self.params['dropout_rate']
        self.learning_rate = self.params['learning_rate']
        self.batch_size = self.params['batch_size']
        self.n_epochs = self.params['n_epochs']
        
        # Model attributes that will be set during fit
        self.fnn_model = None
        self.scaler = None
        self.feature_cols = None
        self.n_features = None
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = {}
        
    def _set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _build_fnn_model(self, input_dim):
        """Build the actual FNN model"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            ])
            prev_dim = hidden_dim
        
        backbone = nn.Sequential(*layers)
        classifier = nn.Linear(prev_dim, 1)
        
        # Create the full model
        model = nn.Sequential(backbone, classifier)
        return model
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.fnn_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def fit(self, train_df: pd.DataFrame, y_train: np.array) -> 'FNNClassifier':
        """Train the FNN classifier."""
        self._set_seed(self.seed)
        
        # Get feature columns (exclude metadata)
        self.feature_cols = [col for col in train_df.columns if col not in ['Country', 'Date']]
        self.n_features = len(self.feature_cols)
        
        # Create and fit scaler
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(train_df[self.feature_cols])
        
        # Build the model
        self.fnn_model = self._build_fnn_model(self.n_features)
        self.fnn_model.to(self.device)
        self._initialize_weights()
        
        # Create dataset and data loader
        train_dataset = BasicDataset(X_scaled, y_train)
        
        # Split for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        
        # Create optimizer and criterion
        optimizer = torch.optim.AdamW(self.fnn_model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        if 'criterion' in self.params:
            criterion = self.params['criterion']
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }

        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.fnn_model.train()
            epoch_train_loss = 0
            num_train_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).float()
                
                optimizer.zero_grad()
                logits = self.fnn_model(batch_X)
                loss = criterion(logits.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.fnn_model.eval()
            epoch_val_loss = 0
            num_val_batches = 0
            all_val_predictions = []
            all_val_probabilities = []
            all_val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device).float()
                    
                    logits = self.fnn_model(batch_X)
                    val_loss = criterion(logits.squeeze(), batch_y)
                    
                    epoch_val_loss += val_loss.item()
                    num_val_batches += 1
                    
                    # Get predictions for metrics
                    probabilities = torch.sigmoid(logits.squeeze())
                    predictions = (probabilities > 0.5).float()
                    
                    all_val_predictions.extend(predictions.cpu().numpy())
                    all_val_probabilities.extend(probabilities.cpu().numpy())
                    all_val_targets.extend(batch_y.cpu().numpy())
            
            avg_val_loss = epoch_val_loss / num_val_batches
            self.val_losses.append(avg_val_loss)
            
            # Calculate validation metrics
            all_val_predictions = np.array(all_val_predictions)
            all_val_probabilities = np.array(all_val_probabilities)
            all_val_targets = np.array(all_val_targets)
            
            val_accuracy = accuracy_score(all_val_targets, all_val_predictions)
            val_precision = precision_score(all_val_targets, all_val_predictions, zero_division=0)
            val_recall = recall_score(all_val_targets, all_val_predictions, zero_division=0)
            val_f1 = f1_score(all_val_targets, all_val_predictions, zero_division=0)
            val_auc = roc_auc_score(all_val_targets, all_val_probabilities)
            
            # Store metrics
            self.val_metrics_history['accuracy'].append(val_accuracy)
            self.val_metrics_history['precision'].append(val_precision)
            self.val_metrics_history['recall'].append(val_recall)
            self.val_metrics_history['f1'].append(val_f1)
            self.val_metrics_history['auc'].append(val_auc)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                print(f"  Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        return self
    
    def forward(self, x):
        """Forward pass through the model"""
        if self.fnn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.fnn_model(x)
    
    def predict(self, data_df: pd.DataFrame) -> np.array:
        """Make predictions on new data."""
        if self.fnn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(data_df[self.feature_cols])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.fnn_model.eval()
        with torch.no_grad():
            logits = self.fnn_model(X_tensor)
            predictions = torch.sigmoid(logits.squeeze()) > 0.5
            
        return predictions.cpu().numpy().astype(int)
    
    def predict_proba(self, data_df: pd.DataFrame) -> np.array:
        """Predict class probabilities."""
        if self.fnn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Scale features
        X_scaled = self.scaler.transform(data_df[self.feature_cols])
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Make predictions
        self.fnn_model.eval()
        with torch.no_grad():
            logits = self.fnn_model(X_tensor)
            probabilities = torch.sigmoid(logits.squeeze())
        
        probs = probabilities.cpu().numpy()
        return np.column_stack([1 - probs, probs])

class LSTMClassifier(nn.Module):
    """LSTM-based classification model wrapper that can be trained with DataFrame input"""
    
    def __init__(self, params: dict = None, seed: int = 42, device: str = None):
        """
        Initialize the LSTM Classifier.
        
        Args:
            params: Dictionary containing model parameters
            seed: Random seed for reproducibility
            device: Device to run on ('cpu', 'cuda', or None for auto)
        """
        super(LSTMClassifier, self).__init__()
        
        # Set seed for reproducibility
        self.seed = seed
        self._set_seed(self.seed)
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Default parameters
        default_params = {
            'sequence_length': 12,
            'lstm_units': 256,
            'num_lstm_layers': 3,
            'dense_units': [128, 64],
            'dropout_rate': 0.4,
            'use_attention': True,
            'use_residual': True,
            'learning_rate': 1e-3,
            'batch_size': 32,
            'n_epochs': 100,
            'criterion': nn.BCEWithLogitsLoss()
        }
        
        # Update with provided parameters
        if params is not None:
            default_params.update(params)
        self.params = default_params
        
        # Extract parameters
        self.sequence_length = self.params['sequence_length']
        self.lstm_units = self.params['lstm_units']
        self.num_lstm_layers = self.params['num_lstm_layers']
        self.dense_units = self.params['dense_units']
        self.dropout_rate = self.params['dropout_rate']
        self.use_attention = self.params['use_attention']
        self.use_residual = self.params['use_residual']
        self.learning_rate = self.params['learning_rate']
        self.batch_size = self.params['batch_size']
        self.n_epochs = self.params['n_epochs']
        self.criterion = self.params['criterion']
        
        # Model attributes that will be set during fit
        self.lstm_model = None
        self.scaler = None
        self.feature_cols = None
        self.n_features = None
        self.dataset = None
        self.train_losses = []
        self.val_losses = []
        
    def _set_seed(self, seed):
        """Set random seed for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def _build_model(self, input_size):
        """Build the actual LSTM model"""
        model = _LSTMModel(
            input_size=input_size,
            lstm_units=self.lstm_units,
            num_lstm_layers=self.num_lstm_layers,
            dense_units=self.dense_units,
            dropout_rate=self.dropout_rate,
            use_attention=self.use_attention,
            use_residual=self.use_residual
        )
        return model
    
    def fit(self, train_df: pd.DataFrame, y_train: np.array) -> 'LSTMClassifier':
        """Train the LSTM classifier. """
        self._set_seed(self.seed)
        
        # Create training dataset
        train_dataset = SequentialDataset(
            train_df=train_df,
            y_train=y_train,
            sequence_length=self.sequence_length,
            fit_scaler=True
        )
        
        # Store the scaler and get input dimensions
        self.scaler = train_dataset.scaler
        input_size = train_dataset.n_features
        
        # Build the internal model
        self.lstm_model = self._build_model(input_size)
        self.lstm_model.to(self.device)
        
        # Initialize weights
        self._initialize_weights()
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Split training data for validation (80/20 split)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.lstm_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.params.get('weight_decay', 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            threshold_mode='abs'
        )
        
        # Initialize tracking lists
        self.train_losses = []
        self.val_losses = []
        self.val_metrics_history = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
                
        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.lstm_model.train()
            epoch_train_loss = 0
            num_train_batches = 0
            
            for batch_sequences, batch_labels in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device).float()
                
                optimizer.zero_grad()
                logits, _ = self.lstm_model(batch_sequences)
                loss = self.criterion(logits, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), max_norm=1.0)
                # optimizer.step()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            self.train_losses.append(avg_train_loss)
            
            # Validation phase
            self.lstm_model.eval()
            epoch_val_loss = 0
            num_val_batches = 0
            all_val_predictions = []
            all_val_probabilities = []
            all_val_targets = []
            
            with torch.no_grad():
                for batch_sequences, batch_labels in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device).float()
                    
                    logits, _ = self.lstm_model(batch_sequences)
                    val_loss = self.criterion(logits, batch_labels)
                    
                    epoch_val_loss += val_loss.item()
                    num_val_batches += 1
                    
                    # Get predictions for metrics
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > 0.5).float()
                    
                    all_val_predictions.extend(predictions.cpu().numpy())
                    all_val_probabilities.extend(probabilities.cpu().numpy())
                    all_val_targets.extend(batch_labels.cpu().numpy())
            
            avg_val_loss = epoch_val_loss / num_val_batches
            self.val_losses.append(avg_val_loss)
            
            # Calculate validation metrics
            all_val_predictions = np.array(all_val_predictions)
            all_val_probabilities = np.array(all_val_probabilities)
            all_val_targets = np.array(all_val_targets)
                        
            val_accuracy = accuracy_score(all_val_targets, all_val_predictions)
            val_precision = precision_score(all_val_targets, all_val_predictions, zero_division=0)
            val_recall = recall_score(all_val_targets, all_val_predictions, zero_division=0)
            val_f1 = f1_score(all_val_targets, all_val_predictions, zero_division=0)
            val_auc = roc_auc_score(all_val_targets, all_val_probabilities)
            
            # Store metrics
            self.val_metrics_history['accuracy'].append(val_accuracy)
            self.val_metrics_history['precision'].append(val_precision)
            self.val_metrics_history['recall'].append(val_recall)
            self.val_metrics_history['f1'].append(val_f1)
            self.val_metrics_history['auc'].append(val_auc)
            
            # Step scheduler
            scheduler.step(val_precision)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.n_epochs}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                print(f"  Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        return self
    
    def _initialize_weights(self):
        """Initialize model weights deterministically"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        torch.nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        torch.nn.init.constant_(param, 0)
        
        self.lstm_model.apply(init_weights)
    
    def forward(self, x):
        """Forward pass through the model"""
        if self.lstm_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.lstm_model(x)
    
    def predict(self, data_df: pd.DataFrame) -> np.array:
        """
        Make predictions on new data.
        
        Args:
            data_df: DataFrame with sequential data.
            
        Returns:
            Predicted class labels (0 or 1).
        """
        if self.lstm_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create dataset without fitting scaler
        test_dataset = SequentialDataset(
            train_df=data_df,
            y_train=np.zeros(len(data_df)),  # Dummy labels
            sequence_length=self.sequence_length,
            scaler=self.scaler,
            fit_scaler=False
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Make predictions
        self.lstm_model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch_sequences, _ in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                logits, _ = self.lstm_model(batch_sequences)
                predictions = torch.sigmoid(logits) > 0.5
                all_predictions.extend(predictions.cpu().numpy())
        
        return np.array(all_predictions, dtype=int)
    
    def predict_proba(self, data_df: pd.DataFrame) -> np.array:
        """
        Predict class probabilities.
        
        Args:
            data_df: DataFrame with sequential data.
            
        Returns:
            Predicted probabilities for each class.
        """
        if self.lstm_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create dataset without fitting scaler
        test_dataset = SequentialDataset(
            train_df=data_df,
            y_train=np.zeros(len(data_df)),  # Dummy labels
            sequence_length=self.sequence_length,
            scaler=self.scaler,
            fit_scaler=False
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Make predictions
        self.lstm_model.eval()
        all_probabilities = []
        
        with torch.no_grad():
            for batch_sequences, _ in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                logits, _ = self.lstm_model(batch_sequences)
                probabilities = torch.sigmoid(logits)
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Return probabilities for both classes
        probs = np.array(all_probabilities)
        return np.column_stack([1 - probs, probs])
    
    def predict_with_indices(self, data_df: pd.DataFrame) -> tuple:
        """
        Make predictions and return the valid indices.
        
        Args:
            data_df: DataFrame with sequential data.
            
        Returns:
            Tuple of (predictions, probabilities, valid_indices)
        """
        if self.lstm_model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create dataset without fitting scaler
        test_dataset = SequentialDataset(
            train_df=data_df,
            y_train=np.zeros(len(data_df)),  # Dummy labels
            sequence_length=self.sequence_length,
            scaler=self.scaler,
            fit_scaler=False
        )
        
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Make predictions
        self.lstm_model.eval()
        all_probabilities = []
        
        with torch.no_grad():
            for batch_sequences, _ in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                logits, _ = self.lstm_model(batch_sequences)
                probabilities = torch.sigmoid(logits)
                all_probabilities.extend(probabilities.detach().cpu().numpy())
        
        probs = np.array(all_probabilities)
        predictions = (probs > 0.5).astype(int)
        
        return predictions, np.column_stack([1 - probs, probs]), test_dataset.valid_indices

    def get_aligned_labels(self, data_df: pd.DataFrame, y_true: np.array) -> np.array:
        """
        Get labels aligned with the sequential predictions.
        
        Args:
            data_df: DataFrame with sequential data
            y_true: Original labels array
            
        Returns:
            Labels aligned with sequential predictions
        """
        test_dataset = SequentialDataset(
            train_df=data_df,
            y_train=np.zeros(len(data_df)),  # Dummy labels
            sequence_length=self.sequence_length,
            scaler=self.scaler,
            fit_scaler=False
        )
        
        return y_true[test_dataset.valid_indices]

    def fine_tune(self, train_df: pd.DataFrame, y_train: np.array, 
                  fine_tune_percent: float = 0.3, 
                  fine_tune_params: dict = None) -> 'LSTMClassifier':
        """
        Fine-tune the top percentage of layers on new data.
        
        Args:
            train_df: New training data
            y_train: New training labels
            fine_tune_percent: Percentage of layers to fine-tune (0.0 to 1.0)
            fine_tune_params: Optional parameters for fine-tuning (learning_rate, n_epochs, etc.)
            
        Returns:
            Self for method chaining
        """
        if self.lstm_model is None:
            raise ValueError("Model not fitted. Call fit() first before fine-tuning.")
        
        # Set default fine-tuning parameters
        default_fine_tune_params = {
            'learning_rate': self.learning_rate * 0.1,  # Lower learning rate for fine-tuning
            'n_epochs': max(20, self.n_epochs // 5),    # Fewer epochs
            'batch_size': self.batch_size,
            'weight_decay': 1e-5
        }
        
        if fine_tune_params is not None:
            default_fine_tune_params.update(fine_tune_params)
        
        # Freeze/unfreeze layers based on fine_tune_percent
        self._set_trainable_layers(fine_tune_percent)
        
        # Create fine-tuning dataset
        fine_tune_dataset = SequentialDataset(
            train_df=train_df,
            y_train=y_train,
            sequence_length=self.sequence_length,
            scaler=self.scaler,  # Use existing scaler
            fit_scaler=False     # Don't refit the scaler
        )
        
        # Split for validation (80/20 split)
        train_size = int(0.8 * len(fine_tune_dataset))
        val_size = len(fine_tune_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            fine_tune_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        train_loader = DataLoader(train_subset, batch_size=default_fine_tune_params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=default_fine_tune_params['batch_size'], shuffle=False)
        
        # Create optimizer only for trainable parameters
        trainable_params = [p for p in self.lstm_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=default_fine_tune_params['learning_rate'],
            weight_decay=default_fine_tune_params['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, threshold_mode='abs'
        )
        
        print(f"Fine-tuning {fine_tune_percent:.1%} of model layers...")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print(f"Total parameters: {sum(p.numel() for p in self.lstm_model.parameters()):,}")
        
        # Fine-tuning loop
        fine_tune_train_losses = []
        fine_tune_val_losses = []
        fine_tune_val_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(default_fine_tune_params['n_epochs']):
            # Training phase
            self.lstm_model.train()
            epoch_train_loss = 0
            num_train_batches = 0
            
            for batch_sequences, batch_labels in train_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device).float()
                
                optimizer.zero_grad()
                logits, _ = self.lstm_model(batch_sequences)
                loss = self.criterion(logits, batch_labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = epoch_train_loss / num_train_batches
            fine_tune_train_losses.append(avg_train_loss)
            
            # Validation phase
            self.lstm_model.eval()
            epoch_val_loss = 0
            num_val_batches = 0
            all_val_predictions = []
            all_val_probabilities = []
            all_val_targets = []
            
            with torch.no_grad():
                for batch_sequences, batch_labels in val_loader:
                    batch_sequences = batch_sequences.to(self.device)
                    batch_labels = batch_labels.to(self.device).float()
                    
                    logits, _ = self.lstm_model(batch_sequences)
                    val_loss = self.criterion(logits, batch_labels)
                    
                    epoch_val_loss += val_loss.item()
                    num_val_batches += 1
                    
                    # Get predictions for metrics
                    probabilities = torch.sigmoid(logits)
                    predictions = (probabilities > 0.5).float()
                    
                    all_val_predictions.extend(predictions.cpu().numpy())
                    all_val_probabilities.extend(probabilities.cpu().numpy())
                    all_val_targets.extend(batch_labels.cpu().numpy())
            
            avg_val_loss = epoch_val_loss / num_val_batches
            fine_tune_val_losses.append(avg_val_loss)
            
            # Calculate validation metrics
            all_val_predictions = np.array(all_val_predictions)
            all_val_probabilities = np.array(all_val_probabilities)
            all_val_targets = np.array(all_val_targets)
            
            val_accuracy = accuracy_score(all_val_targets, all_val_predictions)
            val_precision = precision_score(all_val_targets, all_val_predictions, zero_division=0)
            val_recall = recall_score(all_val_targets, all_val_predictions, zero_division=0)
            val_f1 = f1_score(all_val_targets, all_val_predictions, zero_division=0)
            val_auc = roc_auc_score(all_val_targets, all_val_probabilities)
            
            # Store metrics
            fine_tune_val_metrics['accuracy'].append(val_accuracy)
            fine_tune_val_metrics['precision'].append(val_precision)
            fine_tune_val_metrics['recall'].append(val_recall)
            fine_tune_val_metrics['f1'].append(val_f1)
            fine_tune_val_metrics['auc'].append(val_auc)
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = {k: v.cpu().clone() for k, v in self.lstm_model.state_dict().items()}
            
            # Step scheduler
            scheduler.step(val_f1)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Fine-tune Epoch [{epoch+1}/{default_fine_tune_params['n_epochs']}]")
                print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                print(f"  Val Acc: {val_accuracy:.4f}, Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
        
        # Load best model
        if best_model_state is not None:
            # Move state dict back to device
            for k, v in best_model_state.items():
                best_model_state[k] = v.to(self.device)
            self.lstm_model.load_state_dict(best_model_state)
            print(f"Loaded best model with validation F1: {best_val_f1:.4f}")
        
        # Store fine-tuning history
        if not hasattr(self, 'fine_tune_history'):
            self.fine_tune_history = []
        
        self.fine_tune_history.append({
            'fine_tune_percent': fine_tune_percent,
            'train_losses': fine_tune_train_losses,
            'val_losses': fine_tune_val_losses,
            'val_metrics': fine_tune_val_metrics,
            'best_val_f1': best_val_f1,
            'params': default_fine_tune_params
        })
        
        # Unfreeze all layers after fine-tuning
        self._unfreeze_all_layers()
        
        return self
    
    def _set_trainable_layers(self, fine_tune_percent: float):
        """
        Set which layers should be trainable based on the fine-tune percentage.
        
        Args:
            fine_tune_percent: Percentage of layers to fine-tune (0.0 to 1.0)
        """
        if not (0.0 <= fine_tune_percent <= 1.0):
            raise ValueError("fine_tune_percent must be between 0.0 and 1.0")
        
        # Get all named parameters
        all_params = list(self.lstm_model.named_parameters())
        total_layers = len(all_params)
        
        # Calculate how many layers to fine-tune (from the end/top)
        layers_to_fine_tune = max(1, int(total_layers * fine_tune_percent))
        
        # Freeze all parameters first
        for param in self.lstm_model.parameters():
            param.requires_grad = False
        
        # Unfreeze the top layers
        # Prioritize: classifier, dense blocks, then attention, then LSTM layers
        trainable_layers = []
        
        # Always include classifier
        for name, param in all_params:
            if 'classifier' in name:
                trainable_layers.append((name, param))
        
        # Add dense blocks (in reverse order - top layers first)
        for name, param in reversed(all_params):
            if 'dense_blocks' in name and len(trainable_layers) < layers_to_fine_tune:
                if (name, param) not in trainable_layers:
                    trainable_layers.append((name, param))
        
        # Add attention layers
        for name, param in reversed(all_params):
            if ('attention' in name or 'attention_norm' in name) and len(trainable_layers) < layers_to_fine_tune:
                if (name, param) not in trainable_layers:
                    trainable_layers.append((name, param))
        
        # Add LSTM layers (later layers first)
        for name, param in reversed(all_params):
            if 'lstm' in name and len(trainable_layers) < layers_to_fine_tune:
                if (name, param) not in trainable_layers:
                    trainable_layers.append((name, param))
        
        # Add remaining layers if needed
        for name, param in reversed(all_params):
            if len(trainable_layers) < layers_to_fine_tune:
                if (name, param) not in trainable_layers:
                    trainable_layers.append((name, param))
        
        # Set requires_grad=True for selected layers
        trainable_names = set()
        for name, param in trainable_layers:
            param.requires_grad = True
            trainable_names.add(name)
        
        print(f"Trainable layers ({len(trainable_names)}/{total_layers}):")
        for name in sorted(trainable_names):
            print(f"  - {name}")
    
    def _unfreeze_all_layers(self):
        """Unfreeze all layers after fine-tuning."""
        for param in self.lstm_model.parameters():
            param.requires_grad = True
    
    def get_fine_tune_history(self):
        """Get the fine-tuning history."""
        return getattr(self, 'fine_tune_history', [])


class _LSTMModel(nn.Module):
    """Internal LSTM model implementation"""
    
    def __init__(self, input_size, lstm_units=256, num_lstm_layers=3, 
                 dense_units=[128, 64], dropout_rate=0.4, use_attention=True,
                 use_residual=True):
        super(_LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.lstm_units = lstm_units
        self.num_lstm_layers = num_lstm_layers
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Input normalization
        # self.input_norm = nn.LayerNorm(input_size)
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # Input projection for LSTM residual connection (if needed)
        lstm_output_size = lstm_units * 2  # bidirectional
        if use_residual and input_size != lstm_output_size:
            self.input_projection = nn.Linear(input_size, lstm_output_size)
        else:
            self.input_projection = nn.Identity()
        
        # LSTM/GRU layers
        self.lstm = nn.GRU(
            input_size=input_size,
            hidden_size=lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Batch normalization after LSTM
        self.lstm_bn = nn.BatchNorm1d(lstm_units * 2)
        
        # Attention mechanism for sequence weighting
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_units * 2,  # *2 for bidirectional
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(lstm_units * 2)
        
        # Feature extraction from sequence
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate feature dimensions after pooling
        lstm_output_size = lstm_units * 2  # bidirectional
        pooled_features = lstm_output_size * 2  # max + avg pooling
        
        # Add last timestep features
        final_feature_size = pooled_features + lstm_output_size
        
        # Dense, batch norm layers, residual connections
        self.dense_blocks = nn.ModuleList()
        self.residual_projections = nn.ModuleList()

        prev_size = (lstm_units * 2) * 3  # pooled features

        if isinstance(dense_units, int):
            dense_units = [dense_units]  

        for dense_unit in dense_units:
            block = nn.Sequential(
                nn.Linear(prev_size, dense_unit),
                nn.BatchNorm1d(dense_unit),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.dense_blocks.append(block)
            # Residual projection
            if prev_size != dense_unit:
                self.residual_projections.append(nn.Linear(prev_size, dense_unit))
            else:
                self.residual_projections.append(nn.Identity())
            
            prev_size = dense_unit      
        
        # Output layer
        self.classifier = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Kaiming initialization"""
        for module in self.modules():
            if isinstance(module, nn.LSTM) or isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name or 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Apply input batch norm (reshape for BN)
        batch_size, seq_len, features = x.shape
        x = self.input_bn(x.view(-1, features)).view(batch_size, seq_len, features)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        projected_input = self.input_projection(x)
        # lstm_out: (batch_size, seq_len, lstm_units * 2)
        
        # Add LSTM-level residual connection
        if self.use_residual:
            lstm_out = lstm_out + projected_input
        
        # Apply batch norm to LSTM output
        lstm_out = self.lstm_bn(lstm_out.transpose(1,2)).transpose(1,2)
        
        # Attention mechanism
        if self.use_attention:
            # Self-attention to weight important time steps
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)  # Residual connection
        
        # Multiple feature extraction strategies
        # 1. Global max pooling across time dimension
        max_pooled = self.global_max_pool(lstm_out.transpose(1, 2)).squeeze(-1)
        
        # 2. Global average pooling across time dimension  
        avg_pooled = self.global_avg_pool(lstm_out.transpose(1, 2)).squeeze(-1)
        
        # 3. Last timestep output
        last_output = lstm_out[:, -1, :]
        
        # Combine all features
        combined_features = torch.cat([max_pooled, avg_pooled, last_output], dim=1)
        
        # Dense layers with enhanced residual connections
        dense_out = combined_features
        for i, block in enumerate(self.dense_blocks):
            residual = dense_out
            dense_out = block(dense_out)  # Apply: Linear → BN → GELU → Dropout
            
            # Residual connection
            if self.use_residual:
                projected_residual = self.residual_projections[i](residual)
                dense_out = dense_out + projected_residual
        
        # Final classification (return logits for focal loss)
        logits = self.classifier(dense_out)
    
        # Ensure logits is always 1D, even for single samples
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)  # Remove the last dimension if it's 1
        elif logits.dim() == 0:
            logits = logits.unsqueeze(0)  # Add a dimension if it's scalar
        
        return logits, dense_out