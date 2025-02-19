"""
State data processor module for processing and preparing data for model trainer.
"""
import os
from typing import Tuple, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from config import args
from dataloader.data_loader import DataCollectionLoader
from utils.visualization import visualize_dataloaders, plot_state_distribution, visualize_aggregated_power, \
    plot_state_distribution_pie


# TODO: preprocess_data - purpose for is_train?
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

        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler() if args.scaler_type == "standard" else MinMaxScaler()
        self.processed_df = None

        self.features = ['Total Power (W)', 'Unix Time']
        self.label_column = 'state'
        self.categorical_cols = ['Label', 'Location Name', 'Appliance', 'state']
        self.numerical_cols = ['Power (W)', 'Energy (Wh)', 'Total Power (W)', 'Unix Time']

        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
        # Step 1: aggregate power by timestamp
        aggregated_df = df.groupby(['Timestamp', 'Unix Time'])['Power (W)'].sum().reset_index()
        aggregated_df = aggregated_df.rename(columns={'Power (W)': 'Aggregated Power'})

        # Step 2: merge back the state information
        aggregated_df = pd.merge(
            aggregated_df,
            df[['Timestamp', 'state']].drop_duplicates(),
            on='Timestamp',
            how='left'
        )

        processed_df = aggregated_df.copy()

        # Step 3: process categorical columns
        if 'state' in processed_df.columns:
            if is_train:
                self.label_encoders['state'] = LabelEncoder()
                processed_df['state'] = self.label_encoders['state'].fit_transform(processed_df['state'])
            else:
                processed_df['state'] = self.label_encoders['state'].transform(processed_df['state'])

        # Step 4: scale numerical features
        numerical_cols = ['Aggregated Power', 'Unix Time']
        if is_train:
            processed_df[numerical_cols] = self.scaler.fit_transform(processed_df[numerical_cols])
        else:
            processed_df[numerical_cols] = self.scaler.transform(processed_df[numerical_cols])

        self.processed_df = processed_df

        return processed_df

    def create_sequences(self, features: Union[np.ndarray, pd.DataFrame],
                         labels: Optional[Union[np.ndarray, pd.Series]] = None,
                         stride: int = 1) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences with zero padding to include labels from the first data point.

        Args:
            features: Input features
            labels: Input labels
            stride: Stride for sequence creation

        Returns:
            Tuple of (sequences, sequence_labels)
        """
        if isinstance(features, pd.DataFrame):
            features = features.values
        if labels is not None and isinstance(labels, pd.Series):
            labels = labels.values

        # zero padding
        pad_width = self.args.seq_len - 1
        feature_shape = features.shape[1] if len(features.shape) > 1 else 1
        front_padding = np.zeros((pad_width, feature_shape))
        padded_features = np.vstack((front_padding, features))

        sequences = []
        sequence_labels = []

        for i in range(len(features)):
            seq = padded_features[i:i + self.args.seq_len]
            sequences.append(seq)
            if labels is not None:
                sequence_labels.append(labels[i])
        # print(sequence_labels)
        return np.array(sequences), np.array(sequence_labels) if labels is not None else None

    def _split_data(self, processed_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series]:
        """
        Split the processed data while maintaining state transition order.
        """
        df_sorted = processed_df.sort_values('Unix Time')

        # Find consecutive segments with the same state
        state_changes = df_sorted['state'].ne(df_sorted['state'].shift()).cumsum()
        segments = df_sorted.groupby(state_changes)

        # Split segments into three parts
        train_indices = []
        val_indices = []
        test_indices = []

        for _, segment in segments:
            segment_indices = segment.index
            n = len(segment_indices)

            # Calculate split points for this segment
            first_split = n // 3
            second_split = (n * 2) // 3

            # Add indices to respective datasets
            train_indices.extend(segment_indices[:first_split])
            val_indices.extend(segment_indices[first_split:second_split])
            test_indices.extend(segment_indices[second_split:])

        # Sort indices to maintain temporal order
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()

        features = df_sorted[['Unix Time', 'Aggregated Power']]
        labels = df_sorted['state']

        return (
            features.loc[train_indices],
            features.loc[val_indices],
            features.loc[test_indices],
            labels.loc[train_indices],
            labels.loc[val_indices],
            labels.loc[test_indices]
        )

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
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(processed_df)

        X_train_seq, y_train_seq = self.create_sequences(
            X_train, y_train, stride=self.args.stride
        )
        X_val_seq, y_val_seq = self.create_sequences(
            X_val, y_val, stride=self.args.stride
        )
        X_test_seq, y_test_seq = self.create_sequences(
            X_test, y_test, stride=self.args.stride
        )

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

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

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


if __name__ == "__main__":
    args.num_dc = 5
    dc_loader = DataCollectionLoader(args)
    processor = StateDataProcessor(args)
    df = dc_loader.load_preprocess()
    train_loader, val_loader, test_loader = processor.create_data_loaders(df)
    train_dataset, val_dataset, test_dataset = processor.train_dataset, processor.val_dataset, processor.test_dataset

    visualize_dataloaders(train_dataset, val_dataset, test_dataset)
    plot_state_distribution(train_dataset, val_dataset, test_dataset)
    visualize_aggregated_power(processor.processed_df)
    plot_state_distribution_pie(processor.y_train, processor.y_val, processor.y_test, args)
