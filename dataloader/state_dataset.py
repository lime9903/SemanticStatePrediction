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
        self.categorical_cols = ['Label', 'Location Name', 'Appliance', 'state']
        self.numerical_cols = ['Power (W)', 'Energy (Wh)', 'Total Power (W)', 'Unix Time']

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

        # Remove user activity columns if they exist
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
        X_temp = X[train_size:]
        y_train = y[:train_size]
        y_temp = y[train_size:]

        X_val = X_temp[:val_size]
        X_test = X_temp[val_size:]
        y_val = y_temp[:val_size]
        y_test = y_temp[val_size:]

        if self.args.aggregate:
            # for aggregated dataset
            X_practical = X.drop(columns=[col for col in practical_drop_cols if col in X.columns])
            X_train = X_practical[:train_size]
            X_val = X_practical[train_size:train_size + val_size]
            X_test = X_practical[train_size + val_size:]

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test

    def create_sequences(self, data):
        sequences = []
        for i in range(len(data) - self.args.sequence_length + 1):
            seq = data[i:i + self.args.sequence_length]
            sequences.append(seq)
        return np.array(sequences)

    def create_data_loaders(self, df):
        """
        Create PyTorch DataLoaders for training and testing.
        """
        processed_df = self.preprocess_data(df, is_train=True)
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_train_test_data(processed_df)
        X_train_seq = self.create_sequences(X_train)
        X_val_seq = self.create_sequences(X_val)
        X_test_seq = self.create_sequences(X_test)

        X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_seq, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

        train_dataset = StateDataset(X_train_tensor, y_train_tensor)
        val_dataset = StateDataset(X_val_tensor, y_val_tensor)
        test_dataset = StateDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

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
