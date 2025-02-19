"""
Neural network models for NILM (Non-Intrusive Load Monitoring) and state prediction.
Reference Paper: 'New hybrid Deep Learning Models for multi-target NILM disaggregation'
"""
from typing import Tuple, Any, Union, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.visualization import visualize_model_structure


class BaseModel(nn.Module):
    """Base class for all models"""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class ConvBlock(nn.Module):
    """Convolution block with batch normalization and dropout"""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN_RNN(BaseModel):
    """CNN-RNN hybrid for sequence processing"""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.cnn = nn.Sequential(
            ConvBlock(input_size, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            nn.MaxPool1d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            nn.MaxPool1d(2)
        )

        self.rnn1 = nn.RNN(64, 32, batch_first=True)
        self.rnn2 = nn.RNN(32, 64, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        # CNN feature extraction
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)

        # RNN processing
        _, h_n1 = self.rnn1(x)
        x = h_n1[-1]
        _, h_n2 = self.rnn2(x.unsqueeze(1))
        x = h_n2[-1]

        # Classification
        return self.classifier(x)


class CNN_LSTM(BaseModel):
    """CNN-LSTM hybrid for sequence processing"""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.cnn = nn.Sequential(
            ConvBlock(input_size, 32),
            ConvBlock(32, 32),
            ConvBlock(32, 32),
            nn.MaxPool1d(2),
            ConvBlock(32, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            nn.MaxPool1d(2)
        )

        self.lstm1 = nn.LSTM(64, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CNN feature extraction
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        # LSTM processing
        _, (h_n1, _) = self.lstm1(x)
        x = h_n1[-1]
        _, (h_n2, _) = self.lstm2(x.unsqueeze(1))
        x = h_n2[-1]

        # Classification
        return self.classifier(x)


class DAE_Base(BaseModel):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        # Encoder
        self.encoder = nn.Sequential(
            ConvBlock(input_size, 32),
            nn.MaxPool1d(2),  # seq_len -> seq_len/2
            ConvBlock(32, 64),
            nn.MaxPool1d(2),  # seq_len/2 -> seq_len/4
            ConvBlock(64, 128)
        )

        # Decoder
        self.decoder = nn.Sequential(
            ConvBlock(128, 128),
            nn.Upsample(scale_factor=2, mode='linear'),  # seq_len/4 -> seq_len/2
            ConvBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='linear'),  # seq_len/2 -> seq_len
            ConvBlock(64, input_size),
            nn.Tanh()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.permute(0, 2, 1))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder(x)
        return decoded.permute(0, 2, 1)


class DAE_RNN(DAE_Base):
    """DAE-RNN model for sequence processing"""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.rnn1 = nn.RNN(128, 64, batch_first=True)
        self.rnn2 = nn.RNN(64, 64, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        encoded = encoded.permute(0, 2, 1)  # (batch, seq_len, features)
        _, h_n1 = self.rnn1(encoded)
        x = h_n1[-1]
        _, h_n2 = self.rnn2(x.unsqueeze(1))
        x = h_n2[-1]

        return self.classifier(x), decoded


class DAE_LSTM(DAE_Base):
    """DAE-LSTM model for sequence processing"""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.lstm1 = nn.LSTM(128, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoding and reconstruction
        encoded = self.encode(x)
        decoded = self.decode(encoded)

        # LSTM processing using encoded features
        encoded = encoded.permute(0, 2, 1)  # (batch, seq_len, features)
        _, (h_n1, _) = self.lstm1(encoded)
        x = h_n1[-1]
        _, (h_n2, _) = self.lstm2(x.unsqueeze(1))
        x = h_n2[-1]

        # Return both classification and reconstruction
        return self.classifier(x), decoded


class AttentionModule(nn.Module):
    """Multi-head self attention module with layer normalization"""
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)  # Residual connection
        return self.norm(x)  # Layer normalization


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""

    def __init__(self, in_channels: int, out_channels: int,
                 stride: int = 1, dropout: float = 0.2):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv_block(x)
        out += identity  # Residual connection
        out = F.relu(out)
        return self.dropout(out)


class FeatureExtractor(nn.Module):
    """Enhanced CNN feature extractor with residual connections"""

    def __init__(self, input_size: int, base_channels: int = 64):
        super().__init__()

        # Initial convolution
        self.input_conv = nn.Sequential(
            nn.Conv1d(input_size, base_channels, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Residual blocks with increasing channels
        self.layer1 = self._make_residual_layer(base_channels, base_channels, 2)
        self.layer2 = self._make_residual_layer(base_channels, base_channels * 2, 2, 2)
        self.layer3 = self._make_residual_layer(base_channels * 2, base_channels * 4, 2, 2)
        self.layer4 = self._make_residual_layer(base_channels * 4, base_channels * 8, 2, 2)

    def _make_residual_layer(self, in_channels: int, out_channels: int,
                             blocks: int, stride: int = 1) -> nn.Sequential:
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)  # Output: base_channels * 8 = 512


class BaseSemanticModel(BaseModel):
    """Base Semantic model with common components"""
    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)
        self.feature_extractor = FeatureExtractor(input_size)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),  # Changed from hidden_size * 2
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )


class SemanticCNNLSTM(BaseSemanticModel):
    """Improved CNN-LSTM with attention mechanism"""

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.2):
        super().__init__(input_size, num_classes)

        self.attention = AttentionModule(embed_dim=512, dropout=dropout)

        self.lstm = nn.LSTM(
            input_size=512,  # Feature extractor output size
            hidden_size=256,  # Reduced for bidirectional concatenation
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction (output: batch_size, seq_len, 512)
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = x.permute(0, 2, 1)

        # Apply attention and LSTM
        x = self.attention(x)  # Shape remains (batch_size, seq_len, 512)
        _, (hidden, _) = self.lstm(x)

        # Concatenate bidirectional states (2 * 256 = 512)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        return self.classifier(hidden)


class SemanticDAELSTM(BaseSemanticModel):
    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.2):
        super().__init__(input_size, num_classes)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.ConvTranspose1d(32, input_size, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

        self.attention = AttentionModule(embed_dim=512, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoding
        x = x.permute(0, 2, 1)
        encoded = self.feature_extractor(x)

        # Reconstruction (matching original sequence length)
        decoded = self.decoder(encoded)
        decoded = decoded.permute(0, 2, 1)

        # Classification path
        encoded = encoded.permute(0, 2, 1)
        encoded = self.attention(encoded)
        _, (hidden, _) = self.lstm(encoded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.classifier(hidden)

        return out, decoded


def create_model(model_name: str,
                 input_size: int,
                 num_classes: int,
                 **kwargs) -> nn.Module:
    """Factory function to create the specified model"""
    models = {
        'CNN-RNN': CNN_RNN,
        'CNN-LSTM': CNN_LSTM,
        'DAE-RNN': DAE_RNN,
        'DAE-LSTM': DAE_LSTM,
        'Semantic-CNN-LSTM': SemanticCNNLSTM,
        'Semantic-DAE-LSTM': SemanticDAELSTM
    }

    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Available models: {list(models.keys())}")

    return models[model_name](input_size, num_classes, **kwargs)


# ==========================================================
# Usage example
if __name__ == '__main__':
    model = create_model('DAE-RNN', input_size=2, num_classes=81)
    visualize_model_structure(model, input_size=2, sequence_length=50)
    print(model)
