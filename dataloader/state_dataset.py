import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# TODO: preprocess_data - purpose for is_train?
# TODO: sequence - equal distribution for each states?
class StateDataProcessor:
    """
    Data processor for state prediction that handles encoding and scaling.
    """
    def __init__(self, args):
        assert args.scaler_type in ['standard', 'minmax'], "Scaler type must be 'standard' or 'minmax'"
        self.args = args
        self.label_encoders = {}
        if args.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()

        self.features = ['Total Power (W)', 'Unix Time']  # Used features
        self.label_column = 'state'  # Target
        self.categorical_cols = ['Label', 'Location Name', 'Appliance', 'state']
        self.numerical_cols = ['Power (W)', 'Energy (Wh)', 'Total Power (W)', 'Unix Time']
        self.Label_encoder = LabelEncoder()

        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def preprocess_data(self, df, is_train=True):
        """
        Preprocess the data by encoding categorical variables and scaling numerical variables.

        """
        processed_df = df.copy()

        processed_df.drop(columns='Timestamp', inplace=True)
        activity_cols = [col for col in processed_df.columns if 'user' in col and 'activity' in col]
        if activity_cols:
            processed_df.drop(columns=activity_cols, inplace=True)

        # Encode categorical variables
        for col in self.categorical_cols:
            if col in processed_df.columns:
                if is_train:
                    self.label_encoders[col] = LabelEncoder()
                    processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col])
                else:
                    processed_df[col] = self.label_encoders[col].transform(processed_df[col])

        # Scale numerical variables
        numerical_cols_present = [col for col in self.numerical_cols if col in processed_df.columns]
        if numerical_cols_present:
            if is_train:
                processed_df[numerical_cols_present] = self.scaler.fit_transform(processed_df[numerical_cols_present])
            else:
                processed_df[numerical_cols_present] = self.scaler.transform(processed_df[numerical_cols_present])

        return processed_df

    def create_sequences(self, features, labels=None, stride=1):
        """
        Create sequences with optional stride and balanced sampling
        """
        features = features if isinstance(features, np.ndarray) else features.to_numpy()
        if labels is not None:
            labels = labels if isinstance(labels, np.ndarray) else labels.to_numpy()

        sequences = []
        sequence_labels = []

        valid_length = len(features) - self.args.sequence_length + 1

        for i in range(0, valid_length, stride):
            seq = features[i:i + self.args.sequence_length]
            sequences.append(seq)
            if labels is not None:
                # Use the label from the last timestep of the sequence
                sequence_labels.append(labels[i + self.args.sequence_length - 1])

        sequences = np.array(sequences)
        if labels is not None:
            sequence_labels = np.array(sequence_labels)
            return sequences, sequence_labels
        return sequences

    def prepare_train_test_data(self, processed_df):
        """
        Prepare training and test datasets.
        """
        practical_drop_cols = ['Timestamp', 'Label', 'Location Name', 'Power (W)', 'Energy (Wh)', 'Appliance']

        X = processed_df.drop(columns=['state'])
        y = processed_df['state']

        train_size = round(len(X) * (1 - self.args.test_size - self.args.val_size))
        val_size = round(len(X) * self.args.val_size)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        if self.args.aggregate:
            # for aggregated dataset
            X_practical = X.drop(columns=[col for col in practical_drop_cols if col in X.columns])
            X_train = X_practical[:train_size]
            X_val = X_practical[train_size:train_size + val_size]
            X_test = X_practical[train_size + val_size:]

        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, stride=self.args.stride)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val, stride=self.args.stride)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, stride=self.args.stride)

        self.X_train, self.y_train = X_train_seq, y_train_seq
        self.X_val, self.y_val = X_val_seq, y_val_seq
        self.X_test, self.y_test = X_test_seq, y_test_seq

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def create_data_loaders(self, df):
        """
        Create PyTorch DataLoaders for training and testing.
        """
        processed_df = self.preprocess_data(df, is_train=True)
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_train_test_data(processed_df)

        train_dataset = StateDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = StateDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        test_dataset = StateDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader


class StateDataset(Dataset):
    """
    PyTorch Dataset for state prediction.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
