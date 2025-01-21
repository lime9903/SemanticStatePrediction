"""
State data processor module for processing and preparing data for model trainer.
"""
from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


# TODO: preprocess_data - purpose for is_train?
# TODO: sequence - equal distribution for each states?
class StateDataset(Dataset):
    """PyTorch Dataset for state prediction."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        """
        Initialize the dataset.

        Args:
            features: Input features tensor
            labels: Target labels tensor
        """
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class StateDataProcessor:
    """Data processor for state prediction that handles encoding and scaling."""

    def __init__(self, args: Any):
        """
        Initialize the data processor with given arguments.

        Args:
            args: Configuration arguments containing processing parameters
        """
        self._validate_args(args)
        self.args = args

        # Initialize encoders and scaler
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler() if args.scaler_type == "standard" else MinMaxScaler()

        # Define column categories
        self.features = ['Total Power (W)', 'Unix Time']
        self.label_column = 'state'
        self.categorical_cols = ['Label', 'Location Name', 'Appliance', 'state']
        self.numerical_cols = ['Power (W)', 'Energy (Wh)', 'Total Power (W)', 'Unix Time']

        # Initialize data splits
        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    @staticmethod
    def _validate_args(args: Any) -> None:
        """
        Validate initialization arguments.

        Args:
            args: Configuration arguments to validate

        Raises:
            ValueError: If arguments are invalid
        """
        if not hasattr(args, 'scaler_type') or args.scaler_type not in ['standard', 'minmax']:
            raise ValueError("Scaler type must be 'standard' or 'minmax'")

    def preprocess_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Preprocess the data by encoding categorical variables and scaling numerical ones.

        Args:
            df: Input DataFrame to process
            is_train: Whether this is trainer data (affects fitting behavior)

        Returns:
            Processed DataFrame
        """
        processed_df = df.copy()

        # Remove unnecessary columns
        processed_df.drop(columns='Timestamp', inplace=True)
        activity_cols = [col for col in processed_df.columns if 'user' in col and 'activity' in col]
        if activity_cols:
            processed_df.drop(columns=activity_cols, inplace=True)

        # Process categorical columns
        for col in self.categorical_cols:
            if col in processed_df.columns:
                if is_train:
                    self.label_encoders[col] = LabelEncoder()
                    processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col])
                else:
                    processed_df[col] = self.label_encoders[col].transform(processed_df[col])

        # Scale numerical columns
        numerical_cols_present = [col for col in self.numerical_cols if col in processed_df.columns]
        if numerical_cols_present:
            if is_train:
                processed_df[numerical_cols_present] = self.scaler.fit_transform(
                    processed_df[numerical_cols_present]
                )
            else:
                processed_df[numerical_cols_present] = self.scaler.transform(
                    processed_df[numerical_cols_present]
                )

        return processed_df

    def create_sequences(
            self,
            features: Union[np.ndarray, pd.DataFrame],
            labels: Optional[Union[np.ndarray, pd.Series]] = None,
            stride: int = 1
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for time series data with optional stride.

        Args:
            features: Input features to create sequences from
            labels: Optional target labels
            stride: Step size between sequences

        Returns:
            Tuple of (feature sequences, label sequences)
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
                sequence_labels.append(labels[i + self.args.sequence_length - 1])

        sequences = np.array(sequences)
        if labels is not None:
            sequence_labels = np.array(sequence_labels)
            return sequences, sequence_labels
        return sequences, None

    def _split_data(self, processed_df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame,
        pd.Series, pd.Series, pd.Series
    ]:
        """
        Split the processed data into train, validation, and test sets.

        Args:
            processed_df: Processed DataFrame to split

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Split features and labels
        X = processed_df.drop(columns=['state'])
        y = processed_df['state']

        # Calculate split sizes
        total_size = len(processed_df)
        train_size = int(total_size * (1 - self.args.test_size - self.args.val_size))
        val_size = int(total_size * self.args.val_size)

        # Create splits
        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]

        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]

        if self.args.debug:
            print("\nState distribution:")
            print("Unique state in train set:")
            print(sorted(y_train.unique()))
            print("\nUnique state in test set:")
            print(sorted(y_test.unique()))
            print("\nUnique state in validation set:")
            print(sorted(y_val.unique()))

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_train_test_data(self, processed_df: pd.DataFrame) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """
        Prepare trainer, validation and test datasets.

        Args:
            processed_df: Processed DataFrame to prepare

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test) as numpy arrays
        """
        # Define columns to drop
        practical_drop_cols = [
            'Timestamp', 'Label', 'Location Name',
            'Power (W)', 'Energy (Wh)', 'Appliance'
        ]

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(processed_df)

        # Handle aggregated dataset case
        if self.args.aggregate:
            drop_cols = [col for col in practical_drop_cols if col in X_train.columns]
            X_train = X_train.drop(columns=drop_cols)
            X_val = X_val.drop(columns=drop_cols)
            X_test = X_test.drop(columns=drop_cols)

        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(
            X_train, y_train, stride=self.args.stride
        )
        X_val_seq, y_val_seq = self.create_sequences(
            X_val, y_val, stride=self.args.stride
        )
        X_test_seq, y_test_seq = self.create_sequences(
            X_test, y_test, stride=self.args.stride
        )

        # Store data splits
        self.X_train, self.y_train = X_train_seq, y_train_seq
        self.X_val, self.y_val = X_val_seq, y_val_seq
        self.X_test, self.y_test = X_test_seq, y_test_seq

        return (
            self.X_train, self.X_val, self.X_test,
            self.y_train, self.y_val, self.y_test
        )

    def create_data_loaders(
            self,
            df: pd.DataFrame
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for trainer, validation and testing.

        Args:
            df: Input DataFrame to process

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Preprocess data
        processed_df = self.preprocess_data(df, is_train=True)

        # Prepare train/val/test splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_train_test_data(processed_df)

        # Create datasets
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

        # Create data loaders
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

