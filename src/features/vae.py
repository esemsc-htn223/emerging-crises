import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from typing import List
import matplotlib.pyplot as plt
import math

from features.base import DimensionalityReduction
from data.log_utilities import setup_logging
from model.utilities import set_seed
from model.dataset import SequentialDataset

class VAE(nn.Module):
    """
    Variational Autoencoder neural network implementation.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [256, 128, 64]):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout
        
        # Latent space layers
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        decoder_layers = []
        reversed_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(reversed_dims) - 2):
            decoder_layers.extend([
                nn.Linear(reversed_dims[i], reversed_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
        
        # Final layer without activation (will be handled by loss function)
        decoder_layers.append(nn.Linear(reversed_dims[-2], reversed_dims[-1]))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        """Encode input to latent parameters."""
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

class BaseVAE(DimensionalityReduction):
    """
    Variational Autoencoder (VAE) implementation for dimensionality reduction.
    
    VAE learns a probabilistic latent representation that can generate new data
    and provides meaningful interpolations between data points.
    """
    
    def __init__(self, n_components: int = 2, hidden_dims: List[int] = [256, 128, 64],
                 learning_rate: float = 1e-3, batch_size: int = 32, 
                 n_epochs: int = 100, beta: float = 1.0, seed: int = 42, device: str = None):
        """
        Initialize the VAE class.
        
        Args:
            n_components: Latent dimension size.
            hidden_dims: List of hidden layer dimensions.
            learning_rate: Learning rate for optimization.
            batch_size: Batch size for training.
            n_epochs: Number of training epochs.
            beta: Weight for KL divergence loss (beta-VAE).
            device: Device to run on ('cpu', 'cuda', or None for auto).
        """
        self.seed = seed
        set_seed(self.seed)

        self.n_components = n_components
        self.logger = setup_logging()
        self._validate_vae_components()
        
        # VAE specific parameters
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.beta = beta  # Beta-VAE parameter
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Model attributes
        self.vae_model = None
        self.scaler = None
        self.numeric_columns = None
        self.losses = {'total': [], 'kl': [], 'recon': []}

    def _validate_vae_components(self):
        """Validate n_components parameter for VAE."""
        if not isinstance(self.n_components, int):
            raise TypeError("n_components must be an integer for VAE")
        if self.n_components <= 0:
            raise ValueError("n_components must be positive for VAE")
    
    def _vae_loss_function(self, recon_x, x, mu, logvar):
        """
        VAE loss function combining reconstruction and KL divergence losses.
        
        Args:
            recon_x: Reconstructed input.
            x: Original input.
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            
        Returns:
            Total loss, reconstruction loss, KL loss.
        """
        # Reconstruction loss (MSE for continuous data)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def _initialize_weights(self):
        """Initialize model weights deterministically."""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.vae_model.apply(init_weights)
    
    def fit(self, data: pd.DataFrame, labels: np.array) -> 'BaseVAE':
        """
        Fit VAE to the provided data.
        
        Args:
            data: DataFrame with time series data.
            labels: Array of class labels corresponding to the data.
        
        Returns:
            Self for method chaining.
        """
        set_seed(self.seed)
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(self.numeric_columns) == 0:
            raise ValueError("No numeric columns found in the data")
        
        # Scale data to [0, 1]
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data[self.numeric_columns])
        
        # Convert to PyTorch tensors
        train_tensor = torch.FloatTensor(scaled_data).to(self.device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                  worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id))
        
        # Initialize VAE model
        input_dim = len(self.numeric_columns)
        self.vae_model = VAE(input_dim, self.n_components, self.hidden_dims).to(self.device)
        
        # Initialize optimizer
        optimizer = optim.Adam(self.vae_model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.losses = {'total': [], 'recon': [], 'kl': []}
        
        self.logger.info(f"Starting VAE training for {self.n_epochs} epochs...")
        
        for epoch in range(self.n_epochs):
            self.vae_model.train()
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            for _, (batch_data,) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar, _ = self.vae_model(batch_data)
                
                # Compute losses
                total_loss, recon_loss, kl_loss = self._vae_loss_function(
                    recon_batch, batch_data, mu, logvar)
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.vae_model.parameters(), max_norm = 1.0)
                optimizer.step()
                
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
            
            # Average losses
            avg_loss = epoch_loss / len(train_loader)
            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_kl_loss = epoch_kl_loss / len(train_loader)

            self.losses['total'].append(avg_loss)
            self.losses['recon'].append(avg_recon_loss)
            self.losses['kl'].append(avg_kl_loss)

            if (epoch + 1) % 10 == 0:
                self.logger.info(f'Epoch [{epoch+1}/{self.n_epochs}], '
                                 f'Loss: {avg_loss:.4f}, '
                                 f'Recon: {avg_recon_loss:.4f}, '
                                 f'KL: {avg_kl_loss:.4f}')
        
        self.logger.info(f"Fitted VAE with {self.n_components} latent dimensions")
        return self
    
    def transform(self, data: pd.DataFrame, labels: np.array = None) -> pd.DataFrame:
        """
        Transform data to latent space using the fitted VAE.
        
        Args:
            data: DataFrame with time series data to transform.
            labels: Array of class labels (optional).
            
        Returns:
            Transformed data as DataFrame with preserved index.
        """
        if self.vae_model is None:
            raise ValueError("VAE model not fitted. Call fit() first.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if labels is not None and not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if labels is not None and len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        if not all(col in data.columns for col in self.numeric_columns):
            raise ValueError("Data does not contain all numeric columns used in fitting")
        
        # Scale data
        scaled_data = self.scaler.transform(data[self.numeric_columns])
        
        # Convert to tensor
        data_tensor = torch.FloatTensor(scaled_data).to(self.device)
        
        # Encode to latent space
        self.vae_model.eval()
        with torch.no_grad():
            mu, logvar = self.vae_model.encode(data_tensor)
            # Use mean of distribution for deterministic encoding
            latent_data = mu.cpu().numpy()
        
        self.logger.info(f"Transformed data to VAE latent space, shape: {latent_data.shape}")
        return self.to_dataframe(latent_data, data.index, "VAE")
    
    def reconstruct(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct data through the VAE (encode then decode).
        
        Args:
            data: DataFrame to reconstruct.
            
        Returns:
            Reconstructed data as DataFrame.
        """
        if self.vae_model is None:
            raise ValueError("VAE model not fitted. Call fit() first.")
        
        # Scale data
        scaled_data = self.scaler.transform(data[self.numeric_columns])
        data_tensor = torch.FloatTensor(scaled_data).to(self.device)
        
        # Reconstruct
        self.vae_model.eval()
        with torch.no_grad():
            recon, _, _, _ = self.vae_model(data_tensor)
            recon_data = recon.cpu().numpy()
        
        # Inverse transform scaling
        recon_data = self.scaler.inverse_transform(recon_data)
        
        return pd.DataFrame(recon_data, index=data.index, columns=self.numeric_columns)
    
    def generate_samples(self, n_samples: int = 100) -> pd.DataFrame:
        """
        Generate new samples from the VAE's latent space.
        
        Args:
            n_samples: Number of samples to generate.
            
        Returns:
            Generated samples as DataFrame.
        """
        if self.vae_model is None:
            raise ValueError("VAE model not fitted. Call fit() first.")
        
        self.vae_model.eval()
        with torch.no_grad():
            # Sample from standard normal distribution
            z = torch.randn(n_samples, self.n_components).to(self.device)
            
            # Decode to data space
            generated = self.vae_model.decode(z).cpu().numpy()
        
        # Inverse transform scaling
        generated = self.scaler.inverse_transform(generated)
        
        return pd.DataFrame(generated, columns=self.numeric_columns)
    
    def plot_training_history(self, save_path: str = None, show: bool = True):
        """Plot training loss history."""
        if not self.losses['total']:
            raise ValueError("No training history available. Train the model first.")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total loss
        axes[0].plot(self.losses['total'], 'b-', linewidth=2)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(self.losses['recon'], 'r-', linewidth=2)
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        # KL loss
        axes[2].plot(self.losses['kl'], 'g-', linewidth=2)
        axes[2].set_title('KL Divergence Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('VAE Training History', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in VAE fitting."""
        if self.numeric_columns is None:
            raise ValueError("VAE model not fitted. Call fit() first.")
        return self.numeric_columns.tolist()
    
    def get_component_names(self) -> List[str]:
        """Get component names after transformation."""
        if self.vae_model is None:
            raise ValueError("VAE model not fitted. Call fit() first.")
        return [f"VAE{i+1}" for i in range(self.n_components)]


class RecurrentVAE(nn.Module):
    """
    Recurrent Variational Autoencoder implementation.
    """
    def __init__(self, sequence_length: int, input_dim: int, hidden_size: int,
                 latent_dim: int, num_layers: int = 1, block: str = 'LSTM'):
        
        super(RecurrentVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Batch normalization layers
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.encoder_hidden_bn = nn.BatchNorm1d(hidden_size)
        self.decoder_hidden_bn = nn.BatchNorm1d(hidden_size)
        self.output_bn = nn.BatchNorm1d(input_dim)
        
        # Latent processing
        self.latent_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        # Latent space layers
        self.hidden_to_mu = nn.Linear(hidden_size // 2, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_size // 2, latent_dim)

        # Build encoder
        if block == 'LSTM':
            self.encoder = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=0.2 if self.num_layers > 1 else 0,
                batch_first=True)
        elif block == 'GRU':
            self.encoder = nn.GRU(
                input_size=self.input_dim,
                hidden_size=self.hidden_size, 
                num_layers=self.num_layers,
                dropout=0.2 if self.num_layers > 1 else 0,
                batch_first=True)
        else:
            raise ValueError(f"Unsupported block type: {block}")

        # Build decoder
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_size)
        if block == 'LSTM':
            self.decoder = nn.LSTM(
                input_size=self.latent_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=0.2 if self.num_layers > 1 else 0,
                batch_first=True)
        elif block == 'GRU':
            self.decoder = nn.GRU(
                input_size=self.latent_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=0.2 if self.num_layers > 1 else 0,
                batch_first=True)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.input_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.hidden_to_mu.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight, gain=0.01)
        nn.init.constant_(self.hidden_to_logvar.bias, -2.0)
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def encode(self, x):
        batch_size, seq_len, features = x.shape
        x = self.input_bn(x.view(-1, features)).view(batch_size, seq_len, features)

        if isinstance(self.encoder, nn.LSTM):
            output, (h_n, _) = self.encoder(x)
        else:
            output, h_n = self.encoder(x)
        
        # Apply batch norm to LSTM outputs
        output = self.encoder_hidden_bn(
            output.reshape(-1, self.hidden_size)
            ).reshape(batch_size, seq_len, self.hidden_size)
        
        # Use the last hidden state
        h_last = output[:, -1, :]
        
        h_processed = self.latent_layers(h_last)
        mu = self.hidden_to_mu(h_processed)
        logvar = self.hidden_to_logvar(h_processed)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z):
        batch_size = z.size(0)
        
        h_0 = self.latent_to_hidden(z)
        h_0 = h_0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        decoder_input = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        if isinstance(self.decoder, nn.LSTM):
            c_0 = torch.zeros_like(h_0)
            decoder_output, _ = self.decoder(decoder_input, (h_0, c_0))
        else:
            decoder_output, _ = self.decoder(decoder_input, h_0)
        
        # Apply batch norm to decoder outputs
        decoder_output = self.decoder_hidden_bn(
            decoder_output.reshape(-1, self.hidden_size)
        ).reshape(batch_size, self.sequence_length, self.hidden_size)
        
        output = self.hidden_to_output(decoder_output)
        
        return output
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

class KLAnnealer:
    def __init__(self, n_epochs, mode='linear', warmup_epochs=10, min_beta=0.0, max_beta=1.0):
        self.n_epochs = n_epochs
        self.mode = mode
        self.warmup_epochs = warmup_epochs
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.annealing_epochs = max(1, n_epochs - warmup_epochs)
        
    def get_beta(self, epoch):
        # Warmup phase: keep beta at minimum
        if epoch < self.warmup_epochs:
            return self.min_beta
            
        # Calculate annealing progress
        annealing_progress = (epoch - self.warmup_epochs) / self.annealing_epochs
        annealing_progress = min(1.0, annealing_progress)
        
        if self.mode == 'linear':
            beta = self.min_beta + (self.max_beta - self.min_beta) * annealing_progress
        elif self.mode == 'cosine':
            beta = self.min_beta + (self.max_beta - self.min_beta) * \
                   (1 - math.cos(math.pi * annealing_progress)) / 2
        elif self.mode == 'cyclical':
            n_cycles = 3
            cycle_length = self.annealing_epochs / n_cycles
            cycle_position = (epoch - self.warmup_epochs) % cycle_length
            cycle_progress = cycle_position / cycle_length
            beta = self.min_beta + (self.max_beta - self.min_beta) * cycle_progress
        else:
            raise ValueError(f"Unknown annealing mode: {self.mode}")
            
        return min(self.max_beta, max(self.min_beta, beta))
        
class TimeSeriesVAE(DimensionalityReduction):
    """
    Time Series Recurrent VAE implementation for data.
    Uses RecurrentVAE.
    """
    def __init__(self, params: dict = None,
                 seed: int = 42, device: str = None):
        """
        Initialize the TimeSeriesVAE class.

        Args:
            params: Dictionary of parameters for RecurrentVAE.
            seed: Random seed for reproducibility.
            device: Device to run on ('cpu', 'cuda', or None for auto).
        """
        self.seed = seed
        set_seed(self.seed)

        self.logger = setup_logging()

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Default parameters for RecurrentVAE
        default_params = {
            'sequence_length': 12,
            'hidden_size': 64,
            'n_components': 4,
            'num_layers': 1,
            'block': 'LSTM',
            'learning_rate': 1e-3,
            'batch_size': 32,
            'n_epochs': 100,
            'beta': 1.0,
            'KL_annealing': False,
            'warmup_epochs': 10,
            'annealing_mode': 'linear'
        }

        self.params = default_params.copy()
        if params:
            self.params.update(params)
        
        self._validate_vae_components()

        # Model attributes
        self.rvae = None  # Fixed: consistent naming
        self.scaler = None
        self.numeric_columns = None

        # Training history
        self.losses = {'total': [], 'recon': [], 'kl': []}

    def _validate_vae_components(self):
        """Validate n_components parameter for VAE."""
        n_components = self.params.get('n_components')
        if not isinstance(n_components, int):
            raise TypeError("n_components must be an integer for RecurrentVAE")
        if n_components <= 0:
            raise ValueError("n_components must be positive for RecurrentVAE")

    def _vae_loss_function(self, recon_x, x, mu, logvar, beta):
        """
        VAE loss function combining reconstruction and KL divergence losses.
        
        Args:
            recon_x: Reconstructed input.
            x: Original input.
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            beta: Beta parameter for KL weighting.
            
        Returns:
            Total loss, reconstruction loss, KL loss.
        """
        # Reconstruction loss (MSE for continuous data)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss with clamping
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = beta * torch.clamp(kl_loss, max=50.0 * x.size(0))

        # Total loss with beta weighting
        total_loss = recon_loss + kl_loss

        return total_loss, recon_loss, kl_loss
    
    def _train_vae(self, data, params, losses_dict):
        """
        Train a single VAE model.
        """
        set_seed(self.seed)
        
        # Convert to dataset and dataloader
        training_data = data.copy()
        labels_for_dataset = training_data['label'].values
        training_data = training_data.drop(columns=['label'])
        
        # Convert to dataset and dataloader
        train_dataset = SequentialDataset(
            train_df=training_data,
            y_train=labels_for_dataset,
            sequence_length=params['sequence_length'],
            fit_scaler=True
        )
        self.scaler = train_dataset.scaler
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            drop_last=False
        )
        
        # Initialize VAE model
        rvae_model = RecurrentVAE(
            sequence_length=params['sequence_length'],
            input_dim=train_dataset.n_features,
            hidden_size=params['hidden_size'],
            latent_dim=params['n_components'],
            num_layers=params['num_layers'],
            block=params['block']
        ).to(self.device)
        
        # Initialize optimizer
        optimizer = optim.Adam(rvae_model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        self.logger.info(f"Starting RecurrentVAE training for {params['n_epochs']} epochs...")
        
        for epoch in range(params['n_epochs']):
            rvae_model.train()
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            for _, (batch_data, _) in enumerate(train_loader):
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar, _ = rvae_model(batch_data)
                
                # Add KL annealing
                if params['KL_annealing']:
                    annealer = KLAnnealer(params['n_epochs'], warmup_epochs=params['warmup_epochs'], mode=params['annealing_mode'], max_beta=params['beta'])
                    beta_current = annealer.get_beta(epoch)
                else:
                    beta_current = params['beta']

                # Compute losses
                total_loss, recon_loss, kl_loss = self._vae_loss_function(
                    recon_batch, batch_data, mu, logvar, beta_current)
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(rvae_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
            
            # Average losses
            avg_loss = epoch_loss / len(train_loader)
            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_kl_loss = epoch_kl_loss / len(train_loader)
            
            losses_dict['total'].append(avg_loss)
            losses_dict['recon'].append(avg_recon_loss)
            losses_dict['kl'].append(avg_kl_loss)
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f'TimeSeriesVAE Epoch [{epoch+1}/{params["n_epochs"]}], '
                               f'Loss: {avg_loss:.4f}, '
                               f'Recon: {avg_recon_loss:.4f}, '
                               f'KL: {avg_kl_loss:.4f}')
        
        return rvae_model
    
    def fit(self, data: pd.DataFrame, labels: np.array) -> 'TimeSeriesVAE':
        """Fit VAE model to the provided data."""
        set_seed(self.seed)
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        
        # Validate sequence length
        min_samples_per_country = data.groupby('Country').size().min() if 'Country' in data.columns else len(data)
        if min_samples_per_country < self.params['sequence_length']:
            raise ValueError(f"Minimum samples per country ({min_samples_per_country}) "
                             f"is less than sequence_length ({self.params['sequence_length']})")
    
        self.numeric_columns = [col for col in data.columns if col not in ['Country', 'Date', 'label'] and col in data.select_dtypes(include=[np.number]).columns]
        
        if len(self.numeric_columns) == 0:
            raise ValueError("No numeric columns found in the data")
        
        # Scale data to [0, 1]
        self.scaler = MinMaxScaler()
        # scaled_data = self.scaler.fit_transform(data[self.numeric_columns])
        
        # Create DataFrame with labels for SequentialDataset
        data_with_labels = data.copy()
        data_with_labels['label'] = labels
        
        self.logger.info(f"Training TimeSeriesVAE with {len(self.numeric_columns)} features")
        
        # Train RecurrentVAE model - SequentialDataset will handle scaling
        self.rvae = self._train_vae(data_with_labels, self.params, self.losses)
        
        self.logger.info(f"Fitted TimeSeriesVAE.")
        return self
    
    def transform(self, data: pd.DataFrame, labels: np.array = None) -> pd.DataFrame:
        """Transform data using fitted VAE model."""
        if self.rvae is None:
            raise ValueError("RecurrentVAE model not fitted. Call fit() first.")
        
        # Convert to dataset and dataloader
        test_dataset = SequentialDataset(
            train_df=data.drop(columns=['label']) if 'label' in data.columns else data,
            y_train=labels if labels is not None else np.zeros(len(data)),
            sequence_length=self.params['sequence_length'],
            scaler=self.scaler,
            fit_scaler=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=False,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id),
            drop_last=False
        )

        self.rvae.eval()
        
        with torch.no_grad():
            z_run = []

            for batch_data, _ in test_loader:
                batch_data = batch_data.to(self.device)

                mu, logvar = self.rvae.encode(batch_data)
                z_run_each = self.rvae.reparameterize(mu, logvar).cpu().numpy()
                z_run.append(z_run_each)

            z_run = np.concatenate(z_run, axis=0)

        self.logger.info(f"Original data: {len(data)} rows")
        self.logger.info(f"Sequential dataset: {len(test_dataset)} sequences")
        self.logger.info(f"Final embeddings: {len(z_run)} samples")
    
        output_indices = test_dataset.get_valid_indices()[:len(z_run)]
        transformed_data = self.to_dataframe(z_run, pd.Index(output_indices), "RecurrentVAE")

        self.logger.info(f"Transformed data with TimeSeriesVAE, shape: {transformed_data.shape}")
        return transformed_data
    
    def reconstruct(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reconstruct data through the TimeSeriesVAE (encode then decode).
        
        Args:
            data: DataFrame to reconstruct (should be original data, not embeddings).
            
        Returns:
            Reconstructed data as DataFrame.
        """
        if self.rvae is None:
            raise ValueError("RecurrentVAE model not fitted. Call fit() first.")
        
        # Convert to dataset and dataloader
        test_dataset = SequentialDataset(
            train_df=data,
            y_train=np.zeros(len(data)),
            sequence_length=self.params['sequence_length'],
            scaler=self.scaler,
            fit_scaler=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.params['batch_size'], 
            shuffle=False,
            drop_last=False
        )

        self.rvae.eval()
        
        with torch.no_grad():
            reconstructions = []

            for batch_data, _ in test_loader:
                batch_data = batch_data.to(self.device)
                
                # Full reconstruction through VAE
                recon_batch, _, _, _ = self.rvae(batch_data)
                reconstructions.append(recon_batch.cpu().numpy())

            reconstructions = np.concatenate(reconstructions, axis=0)

        # Get original indices for sequences
        output_indices = test_dataset.get_valid_indices()[:len(reconstructions)]
        
        # Inverse transform scaling for the last timestep only
        # reconstructions shape: (n_sequences, sequence_length, n_features)
        last_timestep_recon = reconstructions[:, -1, :]  # Take last timestep
        last_timestep_recon = self.scaler.inverse_transform(last_timestep_recon)
        
        return pd.DataFrame(last_timestep_recon, index=pd.Index(output_indices), columns=self.numeric_columns)
    
    def plot_training_history(self, save_path: str = None, show: bool = True):
        """Plot training loss history."""
        if not self.losses['total']:
            raise ValueError("No training history available. Train the model first.")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total loss
        axes[0].plot(self.losses['total'], 'b-', linewidth=2)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[1].plot(self.losses['recon'], 'r-', linewidth=2)
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)
        
        # KL loss
        axes[2].plot(self.losses['kl'], 'g-', linewidth=2)
        axes[2].set_title('KL Divergence Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle('VAE Training History', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()

    def get_feature_names(self) -> List[str]:
        """Get feature names used in VAE fitting."""
        if self.numeric_columns is None:
            raise ValueError("VAE model not fitted. Call fit() first.")
        return self.numeric_columns.tolist()
    
    def get_component_names(self) -> List[str]:
        """Get component names after transformation."""
        if self.rvae is None:
            raise ValueError("TimeSeriesVAE model not fitted. Call fit() first.")
        return [f"RecurrentVAE{i+1}" for i in range(self.params['n_components'])]

class ClassSpecificVAE(DimensionalityReduction):
    """
    Class-specific VAE implementation for data. Two transformations are applied:
    - One for the majority class
    - One for the minority class
    
    This allows each class to have its own latent representation, which can be
    beneficial for imbalanced datasets or when classes have different underlying
    distributions.
    """
    
    def __init__(self, 
                 majority_params: dict = None, minority_params: dict = None,
                 seed: int = 42, device: str = None):
        """
        Initialize the ClassSpecificVAE class.
        
        Args:
            majority_params: Dictionary of parameters for majority class VAE.
                            Keys: hidden_dims, learning_rate, batch_size, n_epochs, beta
            minority_params: Dictionary of parameters for minority class VAE.
                            Keys: hidden_dims, learning_rate, batch_size, n_epochs, beta
            seed: Random seed for reproducibility.
            device: Device to run on ('cpu', 'cuda', or None for auto).
        """
        self.seed = seed
        set_seed(self.seed)
        
        self.logger = setup_logging()
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Default parameters for majority class VAE
        default_majority = {
            'hidden_dims': [256, 128, 64],
            'learning_rate': 1e-3,
            'batch_size': 32,
            'n_epochs': 100,
            'beta': 1.0,
            'n_components': 2
        }
        
        # Default parameters for minority class VAE
        default_minority = {
            'hidden_dims': [128, 64, 32],
            'learning_rate': 1e-3,
            'batch_size': 16,
            'n_epochs': 150,
            'beta': 1.0,
            'n_components': 2
        }
        
        # Update with provided parameters
        self.majority_params = default_majority.copy()
        if majority_params:
            self.majority_params.update(majority_params)
            
        self.minority_params = default_minority.copy()
        if minority_params:
            self.minority_params.update(minority_params)
        
        self._validate_vae_components()

        # Model attributes
        self.vae_majority = None
        self.vae_minority = None
        self.scaler = None
        self.numeric_columns = None
        
        # Training history
        self.majority_losses = {'total': [], 'recon': [], 'kl': []}
        self.minority_losses = {'total': [], 'recon': [], 'kl': []}
    
    def _validate_vae_components(self):
        """Validate n_components parameter for VAE."""
        for n_components in [self.majority_params['n_components'], self.minority_params['n_components']]:
            if not isinstance(n_components, int):
                raise TypeError("n_components must be an integer for VAE")
            if n_components <= 0:
                raise ValueError("n_components must be positive for VAE")

    def _vae_loss_function(self, recon_x, x, mu, logvar, beta):
        """
        VAE loss function combining reconstruction and KL divergence losses.
        
        Args:
            recon_x: Reconstructed input.
            x: Original input.
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            beta: Beta parameter for KL weighting.
            
        Returns:
            Total loss, reconstruction loss, KL loss.
        """
        # Reconstruction loss (MSE for continuous data)
        recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss with beta weighting
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def _initialize_weights(self, model):
        """Initialize model weights deterministically."""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
    
    def _train_vae(self, data, params, class_name, losses_dict):
        """
        Train a single VAE model.
        
        Args:
            data: Training data for this class.
            params: Parameters dictionary for this VAE.
            losses_dict: Dictionary to store training losses.
            
        Returns:
            Trained VAE model.
        """
        set_seed(self.seed)
        
        # Convert to PyTorch tensors
        train_tensor = torch.FloatTensor(data).to(self.device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True,
            worker_init_fn=lambda worker_id: np.random.seed(self.seed + worker_id)
        )
        
        # Initialize VAE model
        input_dim = data.shape[1]
        vae_model = VAE(input_dim, params['n_components'], params['hidden_dims']).to(self.device)
        self._initialize_weights(vae_model)
        
        # Initialize optimizer
        optimizer = optim.Adam(vae_model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        self.logger.info(f"Starting {class_name} VAE training for {params['n_epochs']} epochs...")
        
        for epoch in range(params['n_epochs']):
            vae_model.train()
            epoch_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0
            
            for _, (batch_data,) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar, _ = vae_model(batch_data)
                
                # Compute losses
                total_loss, recon_loss, kl_loss = self._vae_loss_function(
                    recon_batch, batch_data, mu, logvar, params['beta'])
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(vae_model.parameters(), max_norm = 1.0)
                optimizer.step()
                
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
            
            # Average losses
            avg_loss = epoch_loss / len(train_loader)
            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_kl_loss = epoch_kl_loss / len(train_loader)

            losses_dict['total'].append(avg_loss)
            losses_dict['recon'].append(avg_recon_loss)
            losses_dict['kl'].append(avg_kl_loss)
            
            if (epoch + 1) % 20 == 0:
                self.logger.info(f'{class_name} VAE Epoch [{epoch+1}/{params["n_epochs"]}], '
                               f'Loss: {avg_loss:.4f}, '
                               f'Recon: {avg_recon_loss:.4f}, '
                               f'KL: {avg_kl_loss:.4f}')
        
        return vae_model
    
    def fit(self, data: pd.DataFrame, labels: np.array) -> 'ClassSpecificVAE':
        """
        Fit VAE models to the provided data for both majority and minority classes.
        
        Args:
            data: DataFrame with time series data.
            labels: Array of class labels corresponding to the data.
        
        Returns:
            Self for method chaining.
        """
        set_seed(self.seed)
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(self.numeric_columns) == 0:
            raise ValueError("No numeric columns found in the data")
        
        # Scale data to [0, 1] for better VAE training
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data[self.numeric_columns])
        
        # Separate majority and minority classes
        majority_data = scaled_data[labels == 0]
        minority_data = scaled_data[labels == 1]
        
        if len(majority_data) == 0:
            raise ValueError("No majority class samples found (labels == 0)")
        if len(minority_data) == 0:
            raise ValueError("No minority class samples found (labels == 1)")
        
        self.logger.info(f"Training ClassSpecificVAE: {len(majority_data)} majority, {len(minority_data)} minority samples")
        
        # Train VAE models for both classes
        self.vae_majority = self._train_vae(
            majority_data, self.majority_params, "Majority", self.majority_losses)
        
        self.vae_minority = self._train_vae(
            minority_data, self.minority_params, "Minority", self.minority_losses)
        
        self.logger.info(f"Fitted ClassSpecificVAE.")
        return self
    
    def transform(self, data: pd.DataFrame, labels: np.array = None) -> pd.DataFrame:
        """
        Transform data using both fitted VAE models and concatenate results.
        
        Args:
            data: DataFrame with time series data to transform.
            labels: Array of class labels (optional).
            
        Returns:
            Transformed data as DataFrame with preserved index.
        """
        if self.vae_majority is None or self.vae_minority is None:
            raise ValueError("VAE models not fitted. Call fit() first.")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        if labels is not None and not isinstance(labels, np.ndarray):
            raise TypeError("Labels must be a NumPy array")
        if labels is not None and len(data) != len(labels):
            raise ValueError("Data and labels must have the same length")
        if not all(col in data.columns for col in self.numeric_columns):
            raise ValueError("Data does not contain all numeric columns used in fitting")
        
        # Scale data
        scaled_data = self.scaler.transform(data[self.numeric_columns])
        data_tensor = torch.FloatTensor(scaled_data).to(self.device)
        
        # Transform using both VAE models
        self.vae_majority.eval()
        self.vae_minority.eval()
        
        with torch.no_grad():
            # Majority class VAE transformation
            mu_maj, logvar_maj = self.vae_majority.encode(data_tensor)
            latent_majority = mu_maj.cpu().numpy()
            
            # Minority class VAE transformation
            mu_min, logvar_min = self.vae_minority.encode(data_tensor)
            latent_minority = mu_min.cpu().numpy()
        
        # Convert to DataFrames with appropriate column names
        transformed_majority = self.to_dataframe(latent_majority, data.index, "Majority_VAE")
        transformed_minority = self.to_dataframe(latent_minority, data.index, "Minority_VAE")
        
        # Combine transformed data
        transformed_data = pd.concat([transformed_majority, transformed_minority], axis=1)
        
        self.logger.info(f"Transformed data with ClassSpecificVAE, shape: {transformed_data.shape}")
        return transformed_data
    
    def reconstruct(self, data: pd.DataFrame, use_model: str = 'both') -> pd.DataFrame:
        """
        Reconstruct data through the VAE models.
        
        Args:
            data: DataFrame to reconstruct.
            use_model: Which model to use ('majority', 'minority', or 'both').
            
        Returns:
            Reconstructed data as DataFrame.
        """
        if self.vae_majority is None or self.vae_minority is None:
            raise ValueError("VAE models not fitted. Call fit() first.")
        
        if use_model not in ['majority', 'minority', 'both']:
            raise ValueError("use_model must be 'majority', 'minority', or 'both'")
        
        # Scale data
        scaled_data = self.scaler.transform(data[self.numeric_columns])
        data_tensor = torch.FloatTensor(scaled_data).to(self.device)
        
        self.vae_majority.eval()
        self.vae_minority.eval()
        
        with torch.no_grad():
            if use_model == 'majority':
                recon, _, _, _ = self.vae_majority(data_tensor)
                recon_data = recon.cpu().numpy()
                recon_data = self.scaler.inverse_transform(recon_data)
                return pd.DataFrame(recon_data, index=data.index, columns=self.numeric_columns)
            
            elif use_model == 'minority':
                recon, _, _, _ = self.vae_minority(data_tensor)
                recon_data = recon.cpu().numpy()
                recon_data = self.scaler.inverse_transform(recon_data)
                return pd.DataFrame(recon_data, index=data.index, columns=self.numeric_columns)
            
            else:  # both
                recon_maj, _, _, _ = self.vae_majority(data_tensor)
                recon_min, _, _, _ = self.vae_minority(data_tensor)
                
                recon_maj_data = self.scaler.inverse_transform(recon_maj.cpu().numpy())
                recon_min_data = self.scaler.inverse_transform(recon_min.cpu().numpy())
                
                # Average the reconstructions
                avg_recon = (recon_maj_data + recon_min_data) / 2
                
                return pd.DataFrame(avg_recon, index=data.index, columns=self.numeric_columns)
    
    def plot_training_history(self, save_path: str = None, show: bool = True):
        """Plot training loss history for both VAE models."""
        if not self.majority_losses['total'] or not self.minority_losses['total']:
            raise ValueError("No training history available. Train the models first.")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Majority class plots
        axes[0, 0].plot(self.majority_losses['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Majority VAE - Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.majority_losses['recon'], 'r-', linewidth=2)
        axes[0, 1].set_title('Majority VAE - Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(self.majority_losses['kl'], 'g-', linewidth=2)
        axes[0, 2].set_title('Majority VAE - KL Divergence Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Minority class plots
        axes[1, 0].plot(self.minority_losses['total'], 'b-', linewidth=2)
        axes[1, 0].set_title('Minority VAE - Total Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.minority_losses['recon'], 'r-', linewidth=2)
        axes[1, 1].set_title('Minority VAE - Reconstruction Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(self.minority_losses['kl'], 'g-', linewidth=2)
        axes[1, 2].set_title('Minority VAE - KL Divergence Loss')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('ClassSpecificVAE Training History', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used in VAE fitting."""
        if self.numeric_columns is None:
            raise ValueError("VAE models not fitted. Call fit() first.")
        return self.numeric_columns.tolist()
    
    def get_component_names(self) -> List[str]:
        """Get component names after transformation."""
        if self.vae_majority is None or self.vae_minority is None:
            raise ValueError("VAE models not fitted. Call fit() first.")
        return [f"Majority_VAE{i+1}" for i in range(self.majority_params['n_components'])] + \
               [f"Minority_VAE{i+1}" for i in range(self.minority_params['n_components'])]
    
    def get_model_parameters(self) -> dict:
        """Get the parameters used for both VAE models."""
        return {
            'majority_params': self.majority_params,
            'minority_params': self.minority_params,
            'device': str(self.device)
        }