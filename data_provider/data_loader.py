"""
Data loader classes for all supported datasets.

This module provides dataset classes for loading and preprocessing
time series data from various domains. Each class handles the specific
data format, train/val/test splitting, and feature engineering for
its corresponding dataset.

Supported datasets:
    - Dataset_ETT_hour: ETTh1/ETTh2 hourly energy transformer data
    - Dataset_ETT_minute: ETTm1/ETTm2 minute-level energy transformer data
    - Dataset_Custom: Generic custom datasets (ECL, Traffic, etc.)
    - Dataset_M4: M4 competition dataset
    - Dataset_city_network: City network traffic data (China Mobile)
    - Dataset_Residential: Residential power load data
    - Dataset_IOT: IoT sensor traffic flow data
    - Dataset_M5_daily: M5 retail sales (daily aggregation)
    - Dataset_Interstate: Interstate traffic data

Note: The actual data loader implementations should be adapted from
the Time-Series-Library (https://github.com/thuml/Time-Series-Library).
Please refer to their data_loader.py for the base implementations
and customize as needed for your specific datasets.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
from utils.timefeatures import time_features

warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    """ETT-hour dataset (ETTh1, ETTh2).
    
    Electricity Transformer Temperature dataset with hourly granularity.
    Contains 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT.
    """

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        percent=100,
        seasonal_patterns=None,
    ):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border1 = 0
            border2 = int(border2s[0] * self.percent / 100)

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp["year"] = df_stamp.date.apply(lambda row: row.year)
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = np.zeros((len(df_stamp), 4))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset_ETT_hour):
    """ETT-minute dataset (ETTm1, ETTm2).
    
    Minute-level variant with different train/val/test boundaries.
    """

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp["year"] = df_stamp.date.apply(lambda row: row.year)
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = np.zeros((len(df_stamp), 4))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


class Dataset_Custom(Dataset_ETT_hour):
    """Generic custom dataset loader.
    
    Supports arbitrary CSV datasets with a 'date' column and
    configurable train/val/test split ratios (7:1:2 by default).
    """

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp["year"] = df_stamp.date.apply(lambda row: row.year)
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(["date"], axis=1).values
        elif self.timeenc == 1:
            data_stamp = np.zeros((len(df_stamp), 4))

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp


class Dataset_M4(Dataset):
    """M4 competition dataset."""

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="M4.csv",
        target="OT",
        scale=False,
        timeenc=0,
        freq="h",
        seasonal_patterns="Monthly",
    ):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.seasonal_patterns = seasonal_patterns
        self.flag = flag
        self.__read_data__()

    def __read_data__(self):
        # M4 dataset loading logic
        # Adapt from Time-Series-Library implementation
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return 0

    def last_insample_window(self):
        pass


class Dataset_city_network(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="city_network.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        percent=100,
        seasonal_patterns=None,
    ):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        # self.percent = percent
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            16 * 30 * 24 - self.seq_len,
            20 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            16 * 30 * 24,
            20 * 30 * 24,
            26 * 30 * 24,
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            # Add year information to time features
            df_stamp["year"] = df_stamp.date.apply(lambda row: row.year, 1)
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=["date"]).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, ...]
        seq_y = self.data_y[r_begin:r_end, ...]
        # seq_x = self.data_x[s_begin:s_end, feat_id : feat_id + 1]
        # seq_y = self.data_y[r_begin:r_end, feat_id : feat_id + 1]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    # (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_IOT(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="ML_IOT.csv",
        target="Junction3",
        scale=True,
        timeenc=0,
        freq="h",
        percent=100,
        seasonal_patterns=None,
    ):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        # Validate dataset length
        if self.tot_len <= 0:
            raise ValueError(
                f"Dataset too small for {flag} split. "
                f"Available: {len(self.data_x)}, Required: {self.seq_len + self.pred_len - 1}, "
                f"Resulting length: {self.tot_len}"
            )

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        total_length = len(df_raw)

        # Validate that we have the expected 16320 data points
        if total_length != 14592:
            print(f"Warning: Expected 14592 data points, but found {total_length}")

        # Time-continuous boundaries based on actual data length (16320 points)
        # Assuming hourly data: ~22.7 months total
        # Split: 12 months training, 4 months validation, 4 months testing

        # Calculate boundaries based on time periods
        # 14 months training: 12 * 30 * 24 = 8640 points
        # 4 months validation: 4 * 30 * 24 = 2880 points
        # Remaining for test: 14592 - 8640 - 2880 = 3072 points

        train_end = min(8640, total_length)
        val_end = min(train_end + 2880, total_length)

        # Define time-continuous boundaries
        border1s = [
            0,  # Train start
            train_end - self.seq_len,  # Val start (with sequence overlap)
            val_end - self.seq_len,  # Test start (with sequence overlap)
        ]

        border2s = [
            train_end,  # Train end
            val_end,  # Val end
            total_length,  # Test end
        ]

        # Validate boundaries don't go negative due to sequence length
        border1s = [max(0, b1) for b1 in border1s]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Apply percentage sampling for training set only
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            border2 = min(border2, border2s[0])  # Don't exceed original train boundary

        # Final validation of boundaries
        if border1 >= border2:
            raise ValueError(
                f"Invalid boundaries for {['train', 'val', 'test'][self.set_type]} set: "
                f"start={border1}, end={border2}"
            )

        if border2 > total_length:
            raise ValueError(
                f"Boundary exceeds dataset: end={border2}, total={total_length}"
            )

        # Calculate and display split information
        split_name = ["train", "val", "test"][self.set_type]
        data_length = border2 - border1
        sequence_count = max(0, data_length - self.seq_len - self.pred_len + 1)

        # Calculate time period covered (assuming hourly data)
        days_covered = data_length / 24
        months_covered = days_covered / 30

        print(
            f"{split_name.capitalize()} boundaries: [{border1}:{border2}], "
            f"data_length: {data_length}, sequences: {sequence_count}"
        )
        print(
            f"{split_name.capitalize()} time coverage: {days_covered:.1f} days "
            f"({months_covered:.1f} months)"
        )

        # Display overall split summary for training set
        if self.set_type == 0:
            print(f"Data split summary:")
            print(
                f"  Training: [0:{train_end}] = {train_end} points ({train_end/24:.1f} days)"
            )
            print(
                f"  Validation: [{train_end-self.seq_len}:{val_end}] = {val_end-(train_end-self.seq_len)} points ({(val_end-(train_end-self.seq_len))/24:.1f} days)"
            )
            print(
                f"  Testing: [{val_end-self.seq_len}:{total_length}] = {total_length-(val_end-self.seq_len)} points ({(total_length-(val_end-self.seq_len))/24:.1f} days)"
            )
            print(
                f"  Total coverage: {total_length} points ({total_length/24:.1f} days)"
            )

        # Process features
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        # Apply scaling using training data for consistency
        if self.scale:
            # Always use training data for fitting scaler to maintain consistency
            train_data = df_data[0:train_end]
            if len(train_data) == 0:
                raise ValueError("No training data available for scaling")
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Process timestamps
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp["year"] = df_stamp.date.apply(lambda row: row.year, 1)
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=["date"]).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Residential(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="transformer_data_selected.csv",
        target="C",
        scale=True,
        timeenc=0,
        freq="h",
        percent=100,
        seasonal_patterns=None,
    ):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

        # Validate dataset length
        if self.tot_len <= 0:
            raise ValueError(
                f"Dataset too small for {flag} split. "
                f"Available: {len(self.data_x)}, Required: {self.seq_len + self.pred_len - 1}, "
                f"Resulting length: {self.tot_len}"
            )

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        total_length = len(df_raw)

        # Validate that we have the expected 16320 data points
        if total_length != 16320:
            print(f"Warning: Expected 16320 data points, but found {total_length}")

        # Time-continuous boundaries based on actual data length (16320 points)
        # Assuming hourly data: ~22.7 months total
        # Split: 14 months training, 4 months validation, 4.7 months testing

        train_end = min(10080, total_length)
        val_end = min(train_end + 2880, total_length)

        # Define time-continuous boundaries
        border1s = [
            0,  # Train start
            train_end - self.seq_len,  # Val start (with sequence overlap)
            val_end - self.seq_len,  # Test start (with sequence overlap)
        ]

        border2s = [
            train_end,  # Train end
            val_end,  # Val end
            total_length,  # Test end
        ]

        # Validate boundaries don't go negative due to sequence length
        border1s = [max(0, b1) for b1 in border1s]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Apply percentage sampling for training set only
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
            border2 = min(border2, border2s[0])  # Don't exceed original train boundary

        # Final validation of boundaries
        if border1 >= border2:
            raise ValueError(
                f"Invalid boundaries for {['train', 'val', 'test'][self.set_type]} set: "
                f"start={border1}, end={border2}"
            )

        if border2 > total_length:
            raise ValueError(
                f"Boundary exceeds dataset: end={border2}, total={total_length}"
            )

        # Calculate and display split information
        split_name = ["train", "val", "test"][self.set_type]
        data_length = border2 - border1
        sequence_count = max(0, data_length - self.seq_len - self.pred_len + 1)

        # Calculate time period covered (assuming hourly data)
        days_covered = data_length / 24
        months_covered = days_covered / 30

        print(
            f"{split_name.capitalize()} boundaries: [{border1}:{border2}], "
            f"data_length: {data_length}, sequences: {sequence_count}"
        )
        print(
            f"{split_name.capitalize()} time coverage: {days_covered:.1f} days "
            f"({months_covered:.1f} months)"
        )

        # Display overall split summary for training set
        if self.set_type == 0:
            print(f"Data split summary:")
            print(
                f"  Training: [0:{train_end}] = {train_end} points ({train_end/24:.1f} days)"
            )
            print(
                f"  Validation: [{train_end-self.seq_len}:{val_end}] = {val_end-(train_end-self.seq_len)} points ({(val_end-(train_end-self.seq_len))/24:.1f} days)"
            )
            print(
                f"  Testing: [{val_end-self.seq_len}:{total_length}] = {total_length-(val_end-self.seq_len)} points ({(total_length-(val_end-self.seq_len))/24:.1f} days)"
            )
            print(
                f"  Total coverage: {total_length} points ({total_length/24:.1f} days)"
            )

        # Process features
        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        # Apply scaling using training data for consistency
        if self.scale:
            # Always use training data for fitting scaler to maintain consistency
            train_data = df_data[0:train_end]
            if len(train_data) == 0:
                raise ValueError("No training data available for scaling")
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Process timestamps
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp["year"] = df_stamp.date.apply(lambda row: row.year, 1)
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(columns=["date"]).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, ...]
        seq_y = self.data_y[r_begin:r_end, ...]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)




class Dataset_M5_daily(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="m5.csv",  
        target="HOUSEHOLD",  
        scale=True,
        timeenc=0,
        freq="D",  
        percent=100,
        seasonal_patterns=None,
    ):
        
        if size == None:
            self.seq_len = 90  
            self.label_len = 30  
            self.pred_len = 30  
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        total_days = 1969


        border1s = [
            0,  
            1378 - self.seq_len,  
            1673 - self.seq_len,  
        ]
        border2s = [
            1378,  
            1673,  
            1969,  
        ]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len


        if self.features == "M" or self.features == "MS":

            cols_data = df_raw.columns[1:]  
            df_data = df_raw[cols_data]
        elif self.features == "S":

            df_data = df_raw[[self.target]]


        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 
        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp["year"] = df_stamp.date.apply(lambda row: row.year, 1)
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)

            df_stamp["day_of_year"] = df_stamp.date.apply(
                lambda row: row.timetuple().tm_yday, 1
            )
            data_stamp = df_stamp.drop(columns=["date"]).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end, ...]
        seq_y = self.data_y[r_begin:r_end, ...]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


