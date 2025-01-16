import torch
import torch.nn as nn


class AttentionModel(nn.Module):
    def __init__(self, input_size, num_classes, dropout=0.2):
        super(AttentionModel, self).__init__()

        # CNN layers for feature extraction
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

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        batch_size, seq_len, features = x.size()

        x = x.permute(0, 2, 1)  # (batch_size, features, sequence_length)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch_size, sequence_length, hidden_size)

        lstm_out, (hidden, cell) = self.lstm(x)

        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        output = self.classifier(context_vector)
        return output
