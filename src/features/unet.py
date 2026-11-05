import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from features.vae import TimeSeriesVAE, KLAnnealer
from model.utilities import set_seed
from model.dataset import SequentialDataset

class RecurrentUNET(nn.Module):
    def __init__(self, sequence_length: int, input_dim: int, hidden_size: int,
                 latent_dim: int, num_layers: int = 1, block: str = 'LSTM'):
        super(RecurrentUNET, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Batch normalization layers
        self.input_bn = nn.BatchNorm1d(self.input_dim)
        self.encoder_hidden_bn = nn.BatchNorm1d(self.hidden_size)
        self.decoder_hidden_bn = nn.BatchNorm1d(self.hidden_size)
        # self.output_bn = nn.BatchNorm1d(self.input_dim)
        
        # Build encoder with multiple layers for skip connections
        if block == 'LSTM':
            self.encoder_layers = nn.ModuleList([
                nn.LSTM(input_size=self.input_dim if i == 0 else self.hidden_size,
                       hidden_size=self.hidden_size,
                       num_layers=1,
                       batch_first=True)
                for i in range(self.num_layers)
            ])
        elif block == 'GRU':
            self.encoder_layers = nn.ModuleList([
                nn.GRU(input_size=self.input_dim if i == 0 else self.hidden_size,
                      hidden_size=self.hidden_size,
                      num_layers=1,
                      batch_first=True)
                for i in range(self.num_layers)
            ])
        
        # Skip connection projection layers (for encoder)
        self.skip_projections_enc = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(self.num_layers - 1)
        ])
        
        # Latent space layers - combine all skip connections
        total_hidden = self.hidden_size * self.num_layers
        self.hidden_to_mu = nn.Linear(total_hidden, self.latent_dim)
        self.hidden_to_logvar = nn.Linear(total_hidden, self.latent_dim)

        # Build decoder (reverse order for U-Net structure)
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_size)
        
        if block == 'LSTM':
            self.decoder_layers = nn.ModuleList([
                nn.LSTM(
                    input_size=self.latent_dim if i == 0 else self.hidden_size + self.hidden_size,  # +hidden_size for skip
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True
                ) for i in range(self.num_layers)
            ])
        elif block == 'GRU':
            self.decoder_layers = nn.ModuleList([
                nn.GRU(
                    input_size=self.latent_dim if i == 0 else self.hidden_size + self.hidden_size,  # +hidden_size for skip
                    hidden_size=self.hidden_size,
                    num_layers=1,
                    batch_first=True
                ) for i in range(self.num_layers)
            ])
        
        # Skip connection projection layers (for decoder)
        self.skip_projections_dec = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(self.num_layers - 1)
        ])
        
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
        """Encode with skip connections."""
        batch_size, seq_len, features = x.shape
        
        # Apply input batch normalization
        x = self.input_bn(x.reshape(-1, features)).reshape(batch_size, seq_len, features)
        
        # Store outputs from all layers for skip connections
        encoder_features = []  # Store for decoder
        layer_hidden_states = []
        current_input = x
        
        for i, layer in enumerate(self.encoder_layers):
            if isinstance(layer, nn.LSTM):
                output, (h_n, _) = layer(current_input)
            else:  # GRU
                output, h_n = layer(current_input)
            
            # Apply batch norm to LSTM/GRU outputs
            output = self.encoder_hidden_bn(
                output.reshape(-1, self.hidden_size)
            ).reshape(batch_size, seq_len, self.hidden_size)
            
            # Store both output and hidden state
            encoder_features.append(output)  # For skip connections
            layer_hidden_states.append(h_n[-1])  # For latent representation
            
            # Add skip connection if not the first layer
            if i > 0:
                skip_proj = self.skip_projections_enc[i-1](layer_hidden_states[i-1])
                current_input = output + skip_proj.unsqueeze(1).repeat(1, output.size(1), 1)
            else:
                current_input = output
        
        self.encoder_features = encoder_features
        
        # Concatenate all layer hidden states for latent representation
        combined_hidden = torch.cat(layer_hidden_states, dim=1)
        
        mu = self.hidden_to_mu(combined_hidden)
        logvar = self.hidden_to_logvar(combined_hidden)
        
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
        """Decode with skip connections."""
        batch_size = z.size(0)
        
        h_0 = self.latent_to_hidden(z).unsqueeze(0)
        decoder_input = z.unsqueeze(1).repeat(1, self.sequence_length, 1)
        current_input = decoder_input
        
        for i, layer in enumerate(self.decoder_layers):
            if i > 0 and hasattr(self, 'encoder_features'):
                # connect decoder layer i to encoder layer (num_layers - 1 - i)
                skip_idx = self.num_layers - 1 - i
                if skip_idx >= 0 and skip_idx < len(self.encoder_features):
                    encoder_feature = self.encoder_features[skip_idx]
                    current_input = torch.cat([current_input, encoder_feature], dim=-1)
            
            if isinstance(layer, nn.LSTM):
                c_0 = torch.zeros_like(h_0)
                output, _ = layer(current_input, (h_0, c_0))
            else:  # GRU
                output, _ = layer(current_input, h_0)
            
            # Apply batch norm to decoder outputs
            output = self.decoder_hidden_bn(
                output.reshape(-1, self.hidden_size)
            ).reshape(batch_size, self.sequence_length, self.hidden_size)
            
            current_input = output
        
        output = self.hidden_to_output(current_input)
        # output = self.output_bn(
        #     output.reshape(-1, self.input_dim)
        # ).reshape(batch_size, self.sequence_length, self.input_dim)
        
        return output
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

class TimeSeriesUNET(TimeSeriesVAE):
    def __init__(self, params: dict, seed: int = 42, device: str = None):
        super().__init__(params, seed, device)

        # Skip-specific parameters
        skip_defaults = {
            'use_layer_norm': True,
            'dropout_rate': 0.1
        }
        self.params.update(skip_defaults)

    def _train_vae(self, data, params, losses_dict):
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

        rvae_model = RecurrentUNET(
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