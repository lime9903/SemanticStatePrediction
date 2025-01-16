"""
Paper: 'New hybrid Deep Learning Models for multi-target NILM disaggregation'
"""
import torch.nn as nn


# CNN-RNN Model
class CNN_RNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN_RNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.rnn1 = nn.RNN(64, 32, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(32, 64, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # if x.dim() == 2:
        #     x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, h_n1 = self.rnn1(x)
        x = h_n1[-1]
        _, h_n2 = self.rnn2(x.unsqueeze(1))
        x = h_n2[-1]
        x = self.fc(x)
        return x


# CNN-LSTM Model
class CNN_LSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.lstm1 = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n1, _) = self.lstm1(x)
        x = h_n1[-1]
        _, (h_n2, _) = self.lstm2(x.unsqueeze(1))
        x = h_n2[-1]
        x = self.fc(x)
        return x


# DAE-RNN Model
class DAE_RNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DAE_RNN, self).__init__()
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
        self.rnn1 = nn.RNN(32, 32, num_layers=1, batch_first=True)
        self.rnn2 = nn.RNN(32, 64, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
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
        x = self.fc(x)
        return x


# DAE-LSTM Model
class DAE_LSTM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DAE_LSTM, self).__init__()
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
        self.lstm1 = nn.LSTM(32, 64, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(64, 64, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
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
        x = self.fc(x)
        return x
