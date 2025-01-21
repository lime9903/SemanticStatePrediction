"""
Neural network models for NILM (Non-Intrusive Load Monitoring) and state prediction.
Reference Paper: 'New hybrid Deep Learning Models for multi-target NILM disaggregation'
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all neural network models."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class CNNBlock(nn.Module):
    """Common CNN block used across different models."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 padding: int = 1, use_maxpool: bool = True):
        super().__init__()
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        ]
        if use_maxpool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN_RNN(BaseModel):
    """CNN-RNN hybrid model for sequence processing."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.cnn = nn.Sequential(
            CNNBlock(input_size, 32),
            CNNBlock(32, 64)
        )

        self.rnn1 = nn.RNN(64, 32, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(32, 64, num_layers=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch_size, input_size, seq_len)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)

        _, h_n1 = self.rnn1(x)
        x = h_n1[-1]
        _, h_n2 = self.rnn2(x.unsqueeze(1))
        x = h_n2[-1]

        return self.fc(x)


class CNN_LSTM(BaseModel):
    """CNN-LSTM hybrid model for sequence processing."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.cnn = nn.Sequential(
            CNNBlock(input_size, 32),
            CNNBlock(32, 64)
        )

        self.lstm1 = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, num_layers=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        _, (h_n1, _) = self.lstm1(x)
        x = h_n1[-1]
        _, (h_n2, _) = self.lstm2(x.unsqueeze(1))
        x = h_n2[-1]

        return self.fc(x)


class DAE_Base(BaseModel):
    """Base class for DAE (Denoising Autoencoder) models."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.encoder = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self._init_weights()


class DAE_RNN(DAE_Base):
    """Denoising Autoencoder with RNN for sequence processing."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.rnn1 = nn.RNN(32, 32, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(32, 64, num_layers=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)

        _, h_n1 = self.rnn1(x)
        x = h_n1[-1]
        _, h_n2 = self.rnn2(x.unsqueeze(1))
        x = h_n2[-1]

        return self.fc(x)


class DAE_LSTM(DAE_Base):
    """Denoising Autoencoder with LSTM for sequence processing."""

    def __init__(self, input_size: int, num_classes: int):
        super().__init__(input_size, num_classes)

        self.lstm1 = nn.LSTM(32, 64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, num_layers=1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)

        _, (h_n1, _) = self.lstm1(x)
        x = h_n1[-1]
        _, (h_n2, _) = self.lstm2(x.unsqueeze(1))
        x = h_n2[-1]

        return self.fc(x)


class AttentionModel(BaseModel):
    """Attention-based model with CNN and BiLSTM for sequence processing."""

    def __init__(self, input_size: int, num_classes: int, dropout: float = 0.2):
        super().__init__(input_size, num_classes)

        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,
            bidirectional=True
        )

        self.attention = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.size()

        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        return self.classifier(context_vector)


def get_model(model_name: str, input_size: int, num_classes: int) -> BaseModel:
    """Factory function to get the specified model."""
    models = {
        'CNN-RNN': CNN_RNN,
        'CNN-LSTM': CNN_LSTM,
        'DAE-RNN': DAE_RNN,
        'DAE-LSTM': DAE_LSTM,
        'Attention': AttentionModel
    }

    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")

    return models[model_name](input_size, num_classes)


# Usage example
# model = get_model('DAE-RNN', input_size=2, num_classes=343)
# print(model)
