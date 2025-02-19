"""
Data loader module for loading and processing NILM(Non-Intrusive Load Monitoring) data collections.
"""
from typing import List, Dict, Tuple, Optional
import os
from datetime import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from config import args


class DataCollectionLoader:
    """Load and process data collections from power monitoring devices."""

    def __init__(self, args):
        """Initialize data collection loader with configuration."""
        self._validate_args(args)
        self.args = args
        self.num_dc = args.num_dc

        # Initialize mappings and state information
        self.dc_appliances_mapping: List[Dict[str, str]] = []
        self.dc_activities_mapping: List[Dict[str, str]] = []
        self.appliances_mapping: Optional[Dict[str, str]] = None
        self.activities_mapping: Optional[Dict[str, str]] = None
        self.dc_actions: List[List[str]] = []
        self.actions: Optional[List[str]] = None
        self.dc_users: Optional[List[List[str]]] = None
        self.dc_num_users: List[int] = []
        self.users: Optional[List[str]] = None
        self.num_users: Optional[int] = None
        self.dc_states: List[List[Tuple]] = []
        self.dc_state_ids: List[Dict[Tuple, str]] = []
        self.states: Optional[List[Tuple]] = None
        self.state_ids: Optional[Dict[Tuple, str]] = None
        self.dc_time_interval: List[List[Tuple]] = []
        self.time_interval: Optional[List[Tuple]] = None

        self._initialize_parameters()

    @staticmethod
    def _validate_args(args) -> None:
        """Validate initialization arguments."""

        if not hasattr(args, 'num_dc') or args.num_dc not in [1, 2, 3, 4, 5]:
            raise ValueError("num_dc must be 1, 2, 3, 4 or 5.")

        if not Path(args.data_dir_path).exists():
            print(args.data_dir_path)
            raise ValueError("data_dir_path does not exist.")

    def _initialize_parameters(self) -> None:
        """Initialize all data collection parameters and mappings."""
        self._init_appliances_mapping()
        self._init_activities_mapping()
        self._init_users()
        self._init_time_intervals()
        self._init_states()
        self._init_time_intervals()

    def _init_appliances_mapping(self) -> None:
        """Initialize appliance mappings for each data collection."""
        # Data collection #1 - appliances
        dc1_appliances_mapping = {
            "SMP01": "Refrigerator(207)",
            "SMP02": "Projector",
            "SMP03": "Haewon Monitor",
            "SMP04": "Jeein Monitor",
            "SMP05": "Jeein Computer",
            "Door01": "Door",
            "Motion01": "Motion",
            "Camera01": "Camera",
            "Vacuum01": "Vacuum",
            "SMP06": "Refrigerator(205)",
            "SMP07": "Haewon Computer",
            "SMP08": "Air Purifier",
            "SMP09": "Yurim Computer(205)",
            "SMP10": "Jaewon Computer",
            "SMP11": "NA",
            "SMP12": "Laptop Charger(Yurim)",
            "SMP13": "Printer(207)",
        }

        # Data collection #2 - appliances
        dc2_appliances_mapping = {
            "SMP01": "Refrigerator(207)",
            "SMP02": "Projector",
            "SMP03": "Haewon Computer",
            "SMP04": "Jeein Monitor",
            "SMP05": "Jeein Computer",
            "Door01": "Door",
            "Motion01": "Motion",
            "Camera01": "Camera",
            "Vacuum01": "Vacuum",
            "SMP06": "Refrigerator(205)",
            "SMP07": "NA",
            "SMP08": "Air Purifier",
            "SMP09": "Yurim Computer(205)",
            "SMP10": "Jaewon Computer",
            "SMP11": "NA",
            "SMP12": "Yurim Monitor(207)",
            "SMP13": "Printer(207)",
            "SMP14": "Haewon Monitor",
            "SMP15": "Giup Computer",
            "SMP16": "Jeein Lamp",
            "SMP17": "Haewon Lamp",
            "SMP18": "Yurim Lamp",
            "SMP19": "Yurim Computer(207)",
            "SMP20": "Laptop Charger(Projector)",
        }

        # Data collection #3 - appliances
        dc3_appliances_mapping = {
            "SMP01": "Refrigerator(207)",
            "SMP02": "Projector",
            "SMP03": "Haewon Computer",
            "SMP04": "Jeein Monitor",
            "SMP05": "Jeein Computer",
            "Door01": "Door",
            "Motion01": "Motion",
            "Camera01": "Camera",
            "Vacuum01": "Vacuum",
            "SMP06": "Refrigerator(205)",
            "SMP07": "NA",
            "SMP08": "Air Purifier",
            "SMP09": "Yurim Computer(205)",
            "SMP10": "Jaewon Computer",
            "SMP11": "NA",
            "SMP12": "Yurim Monitor(207)",
            "SMP13": "Printer(207)",
            "SMP14": "Haewon Monitor",
            "SMP15": "Giup Computer",
            "SMP16": "Jeein Lamp",
            "SMP17": "Haewon Lamp",
            "SMP18": "Yurim Lamp",
            "SMP19": "Yurim Computer(207)",
            "SMP20": "Laptop Charger(Projector)",
        }

        # Data collection #4 - appliances
        dc4_appliances_mapping = {
            "SMP01": "Refrigerator(207)",
            "SMP02": "Projector",
            "SMP03": "Haewon Computer",
            "SMP04": "Jeein Monitor",
            "SMP05": "Jeein Computer",
            "Door01": "Door",
            "Motion01": "Motion",
            "Camera01": "Camera",
            "Vacuum01": "Vacuum",
            "SMP06": "Refrigerator(205)",
            "SMP07": "NA",
            "SMP08": "Air Purifier",
            "SMP09": "Yurim Computer(205)",
            "SMP10": "Jaewon Computer",
            "SMP11": "NA",
            "SMP12": "Heehun Computer",
            "SMP13": "Printer(207)",
            "SMP14": "Haewon Lamp",
            "SMP15": "Giup Computer",
            "SMP16": "Jeein Lamp",
            "SMP17": "Haewon Monitor",
            "SMP18": "Heehun Lamp",
            "SMP19": "NA",
            "SMP20": "Laptop Charger(Projector)",
        }

        # Data collection #5 - appliances
        dc5_appliances_mapping = {
            "SMP01": "Refrigerator(207)",
            "SMP02": "Projector",
            "SMP03": "Haewon Computer",
            "SMP04": "Jeein Monitor",
            "SMP05": "Jeein Computer",
            "Door01": "Door",
            "Motion01": "Motion",
            "Camera01": "Camera",
            "Vacuum01": "Vacuum",
            "SMP06": "Refrigerator(205)",
            "SMP07": "Heehun Lamp",
            "SMP08": "Air Purifier",
            "SMP09": "Yurim Computer(205)",
            "SMP10": "Jaewon Computer",
            "SMP11": "NA",
            "SMP12": "Janghyun Computer",
            "SMP13": "NA",
            "SMP14": "NA",
            "SMP15": "Giup Computer",
            "SMP16": "Jeein Lamp",
            "SMP17": "Haewon Monitor",
            "SMP18": "Heehun Computer",
            "SMP19": "Humidifier(207)",
            "SMP20": "Haewon Lamp",
            "SMP21": "Speaker(207)",
            "SMP22": "Printer(207)",
            "SMP23": "Heehun Monitor",
            "SMP24": "Seungnam Computer",
            "SMP25": "Seungnam Monitor",
        }

        # Total appliances mapping list
        self.dc_appliances_mapping.extend([
            dc1_appliances_mapping,
            dc2_appliances_mapping,
            dc3_appliances_mapping,
            dc4_appliances_mapping,
            dc5_appliances_mapping
        ])
        self.appliances_mapping = self.dc_appliances_mapping[self.num_dc - 1]

    def _init_activities_mapping(self) -> None:
        """Initialize activity mappings for each data collection."""
        # Data collection #1 - activity
        dc1_activities_mapping = {
            "abs": "Absence",
            "wpc": "Working with PC",
            "wopc": "Working without PC",
            "l": "Lunch"
        }

        # Data collection #2 - activity
        dc2_activities_mapping = {
            "abs": "Absence",
            "wpc": "Working with PC",
            "wopc": "Working without PC",
            "m": "Meeting",
            "l": "Lunch"
        }

        # Data collection #3 - activity
        dc3_activities_mapping = {
            "abs": "Absence",
            "wpc": "Working with PC",
            "wopc": "Working without PC",
            "l": "Lunch"
        }

        # Data collection #4 - activity
        dc4_activities_mapping = {
            "abs": "Absence",
            "wpc": "Working with PC",
            "wopc": "Working without PC",
            "l": "Lunch"
        }

        # Data collection #5 - activity
        dc5_activities_mapping = {
            "abs": "Absence",
            "wpc": "Working with PC",
            "wopc": "Working without PC"
        }

        # Total activities mapping list
        self.dc_activities_mapping.extend([
            dc1_activities_mapping,
            dc2_activities_mapping,
            dc3_activities_mapping,
            dc4_activities_mapping,
            dc5_activities_mapping
        ])
        self.activities_mapping = self.dc_activities_mapping[self.num_dc - 1]

        for activity in self.dc_activities_mapping:
            self.dc_actions.append(list(activity.keys()))
        self.actions = self.dc_actions[self.num_dc - 1]

    def _init_users(self) -> None:
        """Initialize user information for each data collection."""
        self.dc_users = [
            ['JI', 'HW', 'YR'],   # Data collection #1 - users
            ['JI', 'HW', 'YR'],   # Data collection #2 - users
            ['JI', 'YR'],         # Data collection #3 - users
            ['JI', 'HW'],         # Data collection #4 - users
            ['SN', 'HH', 'JI', 'HW']  # Data collection #5 - users
        ]
        self.dc_num_users = [len(users) for users in self.dc_users]
        self.users = self.dc_users[self.num_dc - 1]
        self.num_users = self.dc_num_users[self.num_dc - 1]

    def _init_states(self) -> None:
        """Initialize state information for each data collection."""
        for idx, actions in enumerate(self.dc_actions):
            if idx == 4:  # DC5
                # Extract all unique states from time intervals
                states_set = set()
                if isinstance(self.time_interval, list) and len(self.time_interval) == 2:
                    dc5_1_time_interval, dc5_2_time_interval = self.time_interval

                    for _, _, state in dc5_1_time_interval + dc5_2_time_interval:
                        states_set.add(state)

                    states = sorted(list(states_set))
                else:
                    states = list(product(actions, repeat=len(self.dc_users[idx])))
            else:
                states = list(product(actions, repeat=len(self.dc_users[idx])))

            state_ids = {state: f"S{index}" for index, state in enumerate(states)}
            self.dc_states.append(states)
            self.dc_state_ids.append(state_ids)

        self.states = self.dc_states[self.num_dc - 1]
        self.state_ids = self.dc_state_ids[self.num_dc - 1]

    def _init_time_intervals(self) -> None:
        """Initialize time intervals for each data collection."""
        # Data collection #1 - activity time interval
        dc1_time_interval = []  # TODO

        # Data collection #2 - activity time interval
        dc2_time_interval = [
            (time(15, 15), time(15, 18), ('wpc', 'abs', 'abs')),
            (time(15, 18), time(15, 21), ('wpc', 'wpc', 'abs')),
            (time(15, 21), time(15, 30), ('wpc', 'wpc', 'wpc')),
            (time(15, 30), time(15, 35), ('wpc', 'wopc', 'wpc')),
            (time(15, 35), time(15, 37), ('wpc', 'wopc', 'l')),
            (time(15, 37), time(15, 50), ('wpc', 'wopc', 'wpc')),
            (time(15, 50), time(15, 52), ('l', 'wopc', 'wpc')),
            (time(15, 52), time(15, 55), ('wpc', 'wopc', 'wpc')),
            (time(15, 55), time(16, 00), ('wopc', 'wopc', 'wpc')),
            (time(16, 00), time(16, 10), ('m', 'm', 'm')),
            (time(16, 10), time(16, 12), ('wpc', 'l', 'wopc')),
            (time(16, 12), time(16, 25), ('wpc', 'wpc', 'wopc')),
            (time(16, 25), time(16, 27), ('abs', 'wpc', 'abs')),
            (time(16, 27), time(16, 30), ('abs', 'abs', 'abs'))
        ]

        # Data collection #3 - activity time interval
        dc3_time_interval = []  # TODO

        # Data collection #4 - activity time interval
        dc4_time_interval = [
            (time(14, 0), time(14, 5), ('abs', 'abs')),
            (time(14, 5), time(14, 10), ('wpc', 'abs')),
            (time(14, 10), time(14, 15), ('wpc', 'wpc')),
            (time(14, 15), time(14, 25), ('wpc', 'wopc')),
            (time(14, 25), time(14, 35), ('wpc', 'l')),
            (time(14, 35), time(14, 45), ('abs', 'l')),
            (time(14, 45), time(14, 55), ('abs', 'wopc')),
            (time(14, 55), time(15, 5), ('abs', 'wpc')),
            (time(15, 5), time(15, 10), ('wpc', 'wpc')),
            (time(15, 10), time(15, 20), ('wopc', 'wpc')),
            (time(15, 20), time(15, 30), ('l', 'wpc')),
            (time(15, 30), time(15, 40), ('l', 'abs')),
            (time(15, 40), time(15, 50), ('wopc', 'abs')),
            (time(15, 50), time(15, 55), ('wpc', 'abs')),
            (time(15, 55), time(16, 0), ('abs', 'abs'))
        ]

        # Data collection #5 - activity time interval
        dc5_1_time_interval = [
            (time(14, 0), time(14, 3), ('abs', 'abs', 'abs', 'abs')),
            (time(14, 3), time(14, 6), ('wpc', 'abs', 'abs', 'abs')),
            (time(14, 6), time(14, 9), ('wpc', 'wpc', 'abs', 'abs')),
            (time(14, 9), time(14, 12), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(14, 12), time(14, 15), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(14, 15), time(14, 18), ('wpc', 'wpc', 'wpc', 'wopc')),
            (time(14, 18), time(14, 21), ('wpc', 'wpc', 'wopc', 'wopc')),
            (time(14, 21), time(14, 24), ('wpc', 'wpc', 'wopc', 'wpc')),
            (time(14, 24), time(14, 27), ('wpc', 'wopc', 'wpc', 'wpc')),
            (time(14, 27), time(14, 30), ('wpc', 'wopc', 'wpc', 'wopc')),
            (time(14, 30), time(14, 33), ('wpc', 'wopc', 'wopc', 'wopc')),
            (time(14, 33), time(14, 36), ('wpc', 'wopc', 'wopc', 'wpc')),
            (time(14, 36), time(14, 39), ('wopc', 'wpc', 'wpc', 'wpc')),
            (time(14, 39), time(14, 42), ('wopc', 'wpc', 'wpc', 'wopc')),
            (time(14, 42), time(14, 45), ('wopc', 'wpc', 'wopc', 'wopc')),
            (time(14, 45), time(14, 48), ('wopc', 'wpc', 'wopc', 'wpc')),
            (time(14, 48), time(14, 51), ('wopc', 'wopc', 'wpc', 'wpc')),
            (time(14, 51), time(14, 54), ('wopc', 'wopc', 'wpc', 'wopc')),
            (time(14, 54), time(14, 57), ('wopc', 'wopc', 'wopc', 'wopc')),
            (time(14, 57), time(15, 0), ('wopc', 'wopc', 'wopc', 'wpc')),
            (time(15, 0), time(15, 3), ('wpc', 'wopc', 'wpc', 'wpc')),
            (time(15, 3), time(15, 6), ('abs', 'wopc', 'wpc', 'wpc')),
            (time(15, 6), time(15, 9), ('abs', 'wopc', 'wpc', 'wopc')),
            (time(15, 9), time(15, 12), ('abs', 'wopc', 'wopc', 'wopc')),
            (time(15, 12), time(15, 15), ('abs', 'wopc', 'wopc', 'wpc')),
            (time(15, 15), time(15, 18), ('abs', 'wpc', 'wpc', 'wpc')),
            (time(15, 18), time(15, 21), ('abs', 'wpc', 'wpc', 'wopc')),
            (time(15, 21), time(15, 24), ('abs', 'wpc', 'wopc', 'wopc')),
            (time(15, 24), time(15, 27), ('abs', 'wpc', 'wopc', 'wpc')),
            (time(15, 27), time(15, 30), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(15, 30), time(15, 33), ('wpc', 'abs', 'wpc', 'wopc')),
            (time(15, 33), time(15, 36), ('wpc', 'abs', 'wopc', 'wopc')),
            (time(15, 36), time(15, 39), ('wpc', 'abs', 'wopc', 'wpc')),
            (time(15, 39), time(15, 42), ('wopc', 'abs', 'wpc', 'wpc')),
            (time(15, 42), time(15, 45), ('wopc', 'abs', 'wpc', 'wopc')),
            (time(15, 45), time(15, 48), ('wopc', 'abs', 'wopc', 'wopc')),
            (time(15, 48), time(15, 51), ('wopc', 'abs', 'wopc', 'wpc')),
            (time(15, 51), time(15, 54), ('wpc', 'abs', 'wopc', 'wpc')),
            (time(15, 54), time(15, 57), ('abs', 'abs', 'wpc', 'wpc')),
            (time(15, 57), time(16, 0), ('abs', 'abs', 'wpc', 'wopc')),
            (time(16, 0), time(16, 3), ('abs', 'abs', 'wopc', 'wopc')),
            (time(16, 3), time(16, 6), ('abs', 'abs', 'wopc', 'wpc')),
            (time(16, 6), time(16, 9), ('abs', 'abs', 'wopc', 'abs')),
            (time(16, 9), time(16, 12), ('abs', 'abs', 'wpc', 'abs')),
            (time(16, 12), time(16, 15), ('abs', 'abs', 'abs', 'abs'))
        ]

        dc5_2_time_interval = [
            (time(15, 0), time(15, 3), ('abs', 'abs', 'abs', 'abs')),
            (time(15, 3), time(15, 6), ('abs', 'wpc', 'wpc', 'abs')),
            (time(15, 6), time(15, 9), ('abs', 'wpc', 'wopc', 'abs')),
            (time(15, 9), time(15, 12), ('abs', 'wopc', 'wopc', 'abs')),
            (time(15, 12), time(15, 15), ('abs', 'wopc', 'wpc', 'abs')),
            (time(15, 15), time(15, 18), ('abs', 'wpc', 'abs', 'abs')),
            (time(15, 18), time(15, 21), ('abs', 'wpc', 'abs', 'wpc')),
            (time(15, 21), time(15, 24), ('abs', 'wopc', 'abs', 'wpc')),
            (time(15, 24), time(15, 27), ('abs', 'wopc', 'abs', 'wopc')),
            (time(15, 27), time(15, 30), ('abs', 'wpc', 'abs', 'wpc')),
            (time(15, 30), time(15, 33), ('abs', 'abs', 'abs', 'wpc')),
            (time(15, 33), time(15, 36), ('abs', 'abs', 'abs', 'wopc')),
            (time(15, 36), time(15, 39), ('abs', 'wpc', 'abs', 'wopc')),
            (time(15, 39), time(15, 42), ('abs', 'wpc', 'abs', 'wpc')),
            (time(15, 42), time(15, 45), ('abs', 'wopc', 'abs', 'abs')),
            (time(15, 45), time(15, 48), ('wpc', 'wopc', 'abs', 'abs')),
            (time(15, 48), time(15, 51), ('wpc', 'wopc', 'wpc', 'abs')),
            (time(15, 51), time(15, 54), ('wpc', 'wopc', 'wopc', 'abs')),
            (time(15, 54), time(15, 57), ('wpc', 'wopc', 'wpc', 'abs')),
            (time(15, 57), time(16, 0), ('wpc', 'wopc', 'abs', 'wpc')),
            (time(16, 0), time(16, 3), ('wpc', 'wopc', 'abs', 'wopc')),
            (time(16, 3), time(16, 6), ('wopc', 'wopc', 'abs', 'wopc')),
            (time(16, 6), time(16, 9), ('wopc', 'wopc', 'abs', 'wpc')),
            (time(16, 9), time(16, 12), ('wopc', 'wopc', 'abs', 'abs')),
            (time(16, 12), time(16, 15), ('wopc', 'wopc', 'wpc', 'abs')),
            (time(16, 15), time(16, 18), ('wopc', 'wopc', 'wopc', 'abs')),
            (time(16, 18), time(16, 21), ('wopc', 'wpc', 'wopc', 'abs')),
            (time(16, 21), time(16, 24), ('wopc', 'wpc', 'wpc', 'abs')),
            (time(16, 24), time(16, 27), ('wopc', 'wpc', 'abs', 'abs')),
            (time(16, 27), time(16, 30), ('wopc', 'wpc', 'abs', 'wpc')),
            (time(16, 30), time(16, 33), ('wopc', 'wpc', 'abs', 'wopc')),
            (time(16, 33), time(16, 36), ('wpc', 'wpc', 'abs', 'wopc')),
            (time(16, 36), time(16, 39), ('wpc', 'wpc', 'abs', 'wpc')),
            (time(16, 39), time(16, 42), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(16, 42), time(16, 45), ('wpc', 'wpc', 'wopc', 'abs')),
            (time(16, 45), time(16, 48), ('wpc', 'abs', 'wopc', 'abs')),
            (time(16, 48), time(16, 51), ('wpc', 'abs', 'wpc', 'abs')),
            (time(16, 51), time(16, 54), ('wpc', 'abs', 'abs', 'wpc')),
            (time(16, 54), time(16, 57), ('wpc', 'abs', 'abs', 'wopc')),
            (time(16, 57), time(17, 0), ('wopc', 'abs', 'abs', 'wopc')),
            (time(17, 0), time(17, 3), ('wopc', 'abs', 'abs', 'wpc')),
            (time(17, 3), time(17, 6), ('wopc', 'abs', 'wpc', 'abs')),
            (time(17, 6), time(17, 9), ('wopc', 'abs', 'wopc', 'abs')),
            (time(17, 9), time(17, 12), ('wopc', 'abs', 'wpc', 'abs')),
            (time(17, 12), time(17, 15), ('wopc', 'abs', 'abs', 'abs')),
            (time(17, 15), time(17, 18), ('wpc', 'abs', 'abs', 'abs')),
            (time(17, 18), time(17, 21), ('abs', 'abs', 'abs', 'abs'))
        ]

        self.dc_time_interval.extend([
            dc1_time_interval,
            dc2_time_interval,
            dc3_time_interval,
            dc4_time_interval,
            [dc5_1_time_interval, dc5_2_time_interval]
        ])
        self.time_interval = self.dc_time_interval[self.num_dc - 1]

    def dc1_load(self, data_dir: str, power_files: List[str],
                 energy_files: List[str]) -> List[pd.DataFrame]:
        """Load and process data for data collection #1."""
        assert len(power_files) == len(energy_files), \
            "There should be the same number of power and energy files."

        num_appliances = len(power_files)
        dfs = []

        for idx in range(num_appliances):
            power_file = power_files[idx]
            energy_file = energy_files[idx]
            power_file_path = os.path.join(data_dir, power_file)
            energy_file_path = os.path.join(data_dir, energy_file)
            device_id = power_file.split('_')[0]

            # Read power and energy data
            power_df = pd.read_csv(str(power_file_path))
            energy_df = pd.read_csv(str(energy_file_path))

            # Process timestamps and rename columns
            power_df['Timestamp'] = pd.to_datetime(power_df['Time'])
            energy_df['Timestamp'] = pd.to_datetime(energy_df['Time'])
            power_df.rename(columns={'Value': 'Power (W)'}, inplace=True)
            energy_df.rename(columns={'Value': 'Energy (Wh)'}, inplace=True)

            # Merge power and energy data
            df = pd.merge_asof(
                power_df.sort_values('Timestamp'),
                energy_df.sort_values('Timestamp'),
                on='Timestamp',
                direction='nearest'
            )

            # Create final DataFrame with selected columns
            final_df = pd.DataFrame()
            final_df['Timestamp'] = df['Timestamp']
            final_df['Label'] = df['Device Label_x']
            final_df['Location Name'] = df['Location_x']
            final_df['Power (W)'] = df['Power (W)']
            final_df['Energy (Wh)'] = df['Energy (Wh)']
            final_df['Appliance'] = self.appliances_mapping.get(device_id, "Unknown")

            dfs.append(final_df)

        return dfs

    def load_data(self) -> pd.DataFrame:
        """Load data based on the data collection number."""
        dc_data_dirs = [
            '20241122',  # dc 1
            '20241128',  # dc 2
            '20241205',  # dc 3
            '20250108',  # dc 4
            ['20250122', '20250123']  # dc 5
        ]

        data_dirs = dc_data_dirs[self.num_dc - 1]
        if not isinstance(data_dirs, list):
            data_dirs = [data_dirs]

        all_device_dfs = {}

        for data_dir in data_dirs:
            data_path = os.path.join(self.args.data_dir_path, data_dir)
            files = [file for file in os.listdir(data_path) if file.endswith('.csv')]

            if self.num_dc == 1:
                power_files = [file for file in files if "power" in file]
                energy_files = [file for file in files if "energy" in file]
                dfs = self.dc1_load(data_path, power_files, energy_files)

                for df in dfs:
                    device = df['Label'].iloc[0]
                    if device not in all_device_dfs:
                        all_device_dfs[device] = []
                    all_device_dfs[device].append(df)
            else:
                for file in files:
                    data_file_path = os.path.join(data_path, file)
                    df = pd.read_csv(data_file_path)
                    device_id = file.split('_')[0]
                    df['Appliance'] = self.appliances_mapping.get(device_id, "Unknown")

                    if device_id not in all_device_dfs:
                        all_device_dfs[device_id] = []
                    all_device_dfs[device_id].append(df)

        device_dfs = []
        for device in sorted(all_device_dfs.keys()):
            device_df = pd.concat(all_device_dfs[device], ignore_index=True)
            device_df['Timestamp'] = pd.to_datetime(device_df['Timestamp'])
            device_df = device_df.sort_values('Timestamp')
            device_dfs.append(device_df)

        final_df = pd.concat(device_dfs, ignore_index=True)

        return final_df

    def load_preprocess(self) -> pd.DataFrame:
        """Load and preprocess data including state assignments."""
        df = self.load_data()

        # Initialize user activity columns
        for i in range(self.num_users):
            df[f'user{i + 1}_activity'] = None

        if not self.time_interval:
            print("No time interval definitions available.")
            return df

        # Assign states based on time intervals
        df['Time'] = df['Timestamp'].dt.time
        df['Date'] = df['Timestamp'].dt.date

        if self.num_dc == 5:
            # Handle DCs has multiple time intervals
            dc5_1_time_interval, dc5_2_time_interval = self.time_interval

            dates = sorted(df['Date'].unique())
            if len(dates) != 2:
                raise ValueError(f"Expected 2 dates for DC5, but got {len(dates)}")

            df_day1 = df[df['Date'] == dates[0]].copy()
            df_day2 = df[df['Date'] == dates[1]].copy()

            for start_time, end_time, states in dc5_1_time_interval:
                mask = self._create_time_mask(df_day1, start_time, end_time)
                self._assign_states(df_day1, mask, states)

            for start_time, end_time, states in dc5_2_time_interval:
                mask = self._create_time_mask(df_day2, start_time, end_time)
                self._assign_states(df_day2, mask, states)

            devices = sorted(df['Label'].unique())
            final_dfs = []

            for device in devices:
                device_day1 = df_day1[df_day1['Label'] == device]
                device_day2 = df_day2[df_day2['Label'] == device]
                device_df = pd.concat([device_day1, device_day2]).sort_values('Timestamp')
                final_dfs.append(device_df)

            df = pd.concat(final_dfs, ignore_index=True)
        else:
            # Original behavior for other DCs (1, 2, 3, 4)
            for start_time, end_time, states in self.time_interval:
                mask = self._create_time_mask(df, start_time, end_time)
                self._assign_states(df, mask, states)

        # Calculate total power and add Unix timestamp
        power_cols = [col for col in df.columns if 'Power (W)' in col]
        if power_cols:
            df['Total Power (W)'] = df[power_cols].sum(axis=1)
        df['Unix Time'] = (pd.to_datetime(df['Timestamp']).astype('int64') // 10 ** 9)

        # Clean up and finalize
        df.drop(columns=['Time', 'Date'], inplace=True)
        df['state'] = df.apply(self.get_current_state, axis=1)

        if self.args.make_equal_dist:
            df = df[df['state'] != 'Unknown']

        self._save_processed_data(df)
        return df

    def get_state_id(self, activities: Tuple[str, ...]) -> str:
        """Get the state ID for a given combination of activities."""
        return self.state_ids.get(activities, "Unknown")

    def get_current_state(self, row: pd.Series) -> str:
        """Get the current state ID based on all users' activities."""
        activities = []
        for i in range(self.num_users):
            activity = row[f'user{i + 1}_activity']
            activities.append(activity)
        return self.get_state_id(tuple(activities))

    def _create_time_mask(self, df: pd.DataFrame, start_time: time, end_time: time) -> pd.Series:
        """Create a boolean mask for time interval selection."""
        if start_time is None:
            return df['Time'] < end_time
        elif end_time is None:
            return df['Time'] >= start_time
        return df['Time'].between(start_time, end_time)

    def _assign_states(self, df: pd.DataFrame, mask: pd.Series, states: Tuple[str, ...]) -> None:
        """Assign states to the dataframe based on the mask."""
        if mask.any():
            for user_idx, state in enumerate(states):
                if user_idx < self.num_users:
                    df.loc[mask, f'user{user_idx + 1}_activity'] = state

    def _save_processed_data(self, df: pd.DataFrame) -> None:
        """Save processed data to CSV file."""
        if self.args.save_data_path:
            save_dir = Path(self.args.save_data_path)
            save_dir.mkdir(parents=True, exist_ok=True)

            start_date = df['Timestamp'].min().strftime('%Y%m%d')
            end_date = df['Timestamp'].max().strftime('%Y%m%d')
            filename = f"dc{self.num_dc}_processed_data_{start_date}_{end_date}.csv"
            file_path = save_dir / filename

            df.to_csv(file_path, index=False)
            print(f"Data saved to: {file_path}")

    def plot_power_consumption(self, save_vis_path: Optional[str] = None) -> None:
        """Plot power consumption visualizations."""
        save_path = save_vis_path or self.args.save_vis_path
        df = self.load_data()
        self._plot_combined_power_usage(df, save_path)
        self._plot_individual_power_usage(df, save_path)

    def plot_state_distribution(self, save_vis_path: Optional[str] = None) -> None:
        """Plot state distribution visualizations."""
        save_path = save_vis_path or self.args.save_vis_path
        df = self.load_preprocess()
        self._plot_state_distribution(df, save_path)

    def _plot_combined_power_usage(self, df: pd.DataFrame, save_path: str) -> None:
        """Plot combined power usage for all devices."""
        plt.figure(figsize=(15, 8))
        unique_labels = df['Label'].unique()
        palette = sns.color_palette("husl", len(unique_labels))

        for i, label in enumerate(unique_labels):
            label_df = df[df['Label'] == label]
            plt.plot(label_df['Timestamp'], label_df['Power (W)'],
                     label=f'{label} Power', color=palette[i], alpha=0.7)
            plt.fill_between(label_df['Timestamp'], label_df['Power (W)'],
                             color=palette[i], alpha=0.2)

        plt.xlabel('Timestamp')
        plt.ylabel('Power (W)')
        plt.title('Power Usage Over Time by Label')
        plt.legend()
        plt.grid(alpha=0.3)

        output_file = os.path.join(save_path, f"dc{self.num_dc}_combined_power_usage.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved combined graph: {output_file}")

    def _plot_individual_power_usage(self, df: pd.DataFrame, save_path: str) -> None:
        """Plot individual power usage for each device."""
        for label in df['Label'].unique():
            plt.figure(figsize=(12, 6))
            label_df = df[df['Label'] == label]
            plt.plot(label_df['Timestamp'], label_df['Power (W)'],
                     label='Power (W)', alpha=0.7)
            plt.fill_between(label_df['Timestamp'], label_df['Power (W)'], alpha=0.2)

            plt.xlabel('Timestamp')
            plt.ylabel('Power (W)')
            plt.title(f'Power Usage Over Time - {label}')
            plt.legend()
            plt.grid(alpha=0.3)

            output_file = os.path.join(save_path, f"dc{self.num_dc}_{label}_power_usage.png")
            plt.savefig(output_file, dpi=300)
            plt.close()
            print(f"Saved individual graph: {output_file}")

    def _plot_state_distribution(self, df: pd.DataFrame, save_path: str) -> None:
        """Plot the distribution of states in a pie chart."""
        sns.set_theme(style="whitegrid")
        state_counts = df['state'].value_counts()

        # Create pie chart
        fig, ax = plt.subplots(figsize=(15, 10))
        colors = sns.color_palette("husl", n_colors=len(state_counts))

        patches, texts, autotexts = ax.pie(
            state_counts,
            labels=state_counts.index,
            autopct='%1.1f%%',
            pctdistance=0.75,
            labeldistance=1.2,
            colors=colors,
            startangle=90
        )

        # Format autopct labels
        for i, autotext in enumerate(autotexts):
            count = state_counts.iloc[i]
            autotext.set_text(f'{count:,d}\n({autotext.get_text()})')

        plt.title(f'DC{self.num_dc} State Distribution', pad=20, size=16)

        # Create legend with state descriptions
        id_to_state = {v: k for k, v in self.state_ids.items()}
        legend_labels = []
        for state_id in state_counts.index:
            if state_id in id_to_state:
                state_tuple = id_to_state[state_id]
                state_str = ' | '.join(str(s) for s in state_tuple)
                legend_labels.append(f'{state_id}: {state_str}')
            else:
                legend_labels.append(state_id)

        plt.legend(
            patches,
            legend_labels,
            title="States",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize='small',
            title_fontsize='medium'
        )

        plt.tight_layout()
        output_file = os.path.join(save_path, f"dc{self.num_dc}_state_distribution.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved state distribution pie chart: {output_file}")


class TestDataCollectionLoader:
    """Load and process test data collections."""
    def __init__(self, args):
        """Initialize test data loader with configuration and test dates."""
        self._validate_args(args)
        self.args = args
        self.num_test = args.num_test
        self.test_data_path = args.data_dir_path

        # All test information
        self.test_dates = []
        self.test_appliances_mapping = []
        self.test_activities_mapping = []
        self.test_actions = []
        self.test_time_interval = []
        self.test_users = None
        self.test_num_users = []
        self.test_states = []
        self.test_state_ids = []

        # Specific test information
        self.dates = None
        self.appliances_mapping = None
        self.activities_mapping = None
        self.actions = None
        self.time_interval = None
        self.users = None
        self.num_users = None
        self.states = None
        self.state_ids = None

        self._initialize_parameters()

    @staticmethod
    def _validate_args(args) -> None:
        """Validate initialization arguments."""

        if not hasattr(args, 'num_test') or args.num_test not in [1]:
            raise ValueError("num_test must be 1.")

        if not Path(args.data_dir_path).exists():
            print(args.data_dir_path)
            raise ValueError("data_dir_path does not exist.")

    def _initialize_parameters(self) -> None:
        """Initialize all data collection parameters and mappings."""
        self._init_test_dates()
        self._init_appliances_mapping()
        self._init_activities_mapping()
        self._init_users()
        self._init_time_intervals()
        self._init_states()
        self._init_time_intervals()

    def _init_test_dates(self) -> None:
        """Initialize test date dates."""
        self.dates = [
            ['20250213', '20250214', '20250215', '20250216', '20250217'],
            ]

        self.test_dates = self.dates[self.num_test - 1]

    def _init_appliances_mapping(self) -> None:
        """Initialize appliance mappings for each data collection."""
        # Test data collection #1 - appliances
        test1_appliances_mapping = {
            "SMP01": "Refrigerator(207)",
            "SMP02": "Projector",
            "SMP03": "Haewon Computer",
            "SMP04": "Jeein Monitor",
            "SMP05": "Jeein Computer",
            "Door01": "Door",
            "Motion01": "Motion",
            "Camera01": "Camera",
            "Vacuum01": "Vacuum",
            "SMP06": "Refrigerator(205)",
            "SMP07": "Heehun Lamp",
            "SMP08": "Air Purifier",
            "SMP09": "Yurim Computer(205)",
            "SMP10": "Jaewon Computer",
            "SMP11": "NA",
            "SMP12": "Janghyun Computer",
            "SMP13": "NA",
            "SMP14": "NA",
            "SMP15": "Giup Computer",
            "SMP16": "Jeein Lamp",
            "SMP17": "Haewon Monitor",
            "SMP18": "Heehun Computer",
            "SMP19": "Humidifier(207)",
            "SMP20": "Haewon Lamp",
            "SMP21": "Speaker(207)",
            "SMP22": "Printer(207)",
            "SMP23": "Heehun Monitor",
            "SMP24": "Seungnam Computer",
            "SMP25": "Seungnam Monitor",
        }

        # Total appliances mapping list
        self.test_appliances_mapping.extend([
            test1_appliances_mapping
        ])
        self.appliances_mapping = self.test_appliances_mapping[self.num_test - 1]

    def _init_activities_mapping(self) -> None:
        """Initialize activity mappings for each data collection."""
        # Test data collection #1 - activity
        test1_activities_mapping = {
            "abs": "Absence",
            "wpc": "Working with PC",
            "wopc": "Working without PC"
        }

        # Total activities mapping list
        self.test_activities_mapping.extend([
            test1_activities_mapping
        ])
        self.activities_mapping = self.test_activities_mapping[self.num_test - 1]

        for activity in self.test_activities_mapping:
            self.test_actions.append(list(activity.keys()))
        self.actions = self.test_actions[self.num_test - 1]

    def _init_users(self) -> None:
        """Initialize user information for each data collection."""
        self.test_users = [
            ['SN', 'HH', 'JI', 'HW']  # Test data collection #1 - users
        ]
        self.test_num_users = [len(users) for users in self.test_users]
        self.users = self.test_users[self.num_test - 1]
        self.num_users = self.test_num_users[self.num_test - 1]

    def _init_states(self) -> None:  # TODO: change...
        """Initialize state information for each data collection."""
        if not self.test_actions:
            self._init_activities_mapping()

        for idx, actions in enumerate(self.test_actions):
            states_set = set()
            if isinstance(self.time_interval, list):
                for day_intervals in self.time_interval:
                    for _, _, state in day_intervals:
                        states_set.add(state)

                states = sorted(list(states_set))
            else:
                states = list(product(actions, repeat=len(self.test_users[idx])))

            state_ids = {state: f"S{index}" for index, state in enumerate(states)}
            self.test_states.append(states)
            self.test_state_ids.append(state_ids)

        self.states = self.test_states[self.num_test - 1]
        self.state_ids = self.test_state_ids[self.num_test - 1]

    def _init_time_intervals(self) -> None:
        """Initialize time intervals for each data collection."""

        # 2025-02-13 Time intervals (Thursday)
        test_20250213_time_interval = [
            (time(10, 25), time(10, 29), ('wpc', 'abs', 'wpc', 'abs')),  # 승남pc, 지인pc
            (time(10, 29), time(10, 48), ('wpc', 'wpc', 'wpc', 'abs')),  # +희훈pc
            (time(10, 48), time(10, 49), ('wpc', 'wpc', 'wpc', 'abs')),  # +재원
            (time(10, 49), time(11, 0), ('abs', 'wpc', 'wpc', 'abs')),  # 승남 lunch
            (time(11, 0), time(11, 39), ('abs', 'wpc', 'abs', 'abs')),  # 지인 meeting
            (time(11, 39), time(11, 40), ('abs', 'wpc', 'wpc', 'abs')),  # 지인 back
            (time(11, 40), time(11, 41), ('abs', 'wpc', 'abs', 'abs')),
            (time(11, 41), time(11, 53), ('abs', 'wpc', 'wpc', 'abs')),
            (time(11, 53), time(11, 54), ('wpc', 'wpc', 'wpc', 'abs')),  # 승남 back
            (time(11, 54), time(12, 7), ('abs', 'wpc', 'wpc', 'abs')),
            (time(12, 7), time(12, 26), ('wpc', 'abs', 'wpc', 'abs')),  # 희훈 lunch
            (time(12, 26), time(12, 39), ('wpc', 'wpc', 'wpc', 'abs')),  # 희훈 back
            (time(12, 39), time(12, 59), ('wpc', 'wpc', 'wpc', 'wpc')),  # 해원 get to work
            (time(12, 59), time(13, 1), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(13, 1), time(13, 27), ('abs', 'wpc', 'wpc', 'wpc')),  # 승남 out
            (time(13, 27), time(13, 50), ('wpc', 'wpc', 'wpc', 'wpc')),  # 승남 back
            (time(13, 50), time(14, 14), ('wpc', 'wpc', 'abs', 'wpc')),  # 지인 lunch
            (time(14, 14), time(14, 27), ('wpc', 'wpc', 'wpc', 'wpc')),  # 지인 back
            (time(14, 27), time(14, 31), ('wpc', 'wpc', 'abs', 'wpc')),  # 지인 out
            (time(14, 31), time(14, 52), ('wpc', 'wpc', 'wpc', 'wpc')),  # 지인 back
            (time(14, 52), time(14, 55), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(14, 55), time(15, 25), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(15, 25), time(15, 33), ('wpc', 'abs', 'wpc', 'wpc')),  # 희훈 meeting
            (time(15, 33), time(15, 36), ('wpc', 'abs', 'wpc', 'abs')),  # 해원 out
            (time(15, 36), time(16, 6), ('wpc', 'abs', 'wpc', 'wpc')),  # 해원 back
            (time(16, 6), time(16, 16), ('abs', 'abs', 'wpc', 'wpc')),  # 승남 out
            (time(16, 16), time(16, 16, 30), ('abs', 'wpc', 'wpc', 'wpc')),  # 희훈 in/out
            (time(16, 16, 30), time(16, 25), ('abs', 'abs', 'wpc', 'wpc')),
            (time(16, 25), time(16, 27), ('wpc', 'abs', 'wpc', 'wpc')),  # 승남 in
            (time(16, 27), time(16, 28), ('wpc', 'abs', 'abs', 'wpc')),  # 지인 out
            (time(16, 28), time(16, 46), ('wpc', 'abs', 'wpc', 'wpc')),  # 지인 in
            (time(16, 46), time(17, 00), ('wpc', 'wpc', 'wpc', 'wpc')),  # 희훈 in
            (time(17, 00), time(17, 12), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(17, 12), time(17, 12, 30), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(17, 12, 30), time(17, 15), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(17, 15), time(18, 4), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(18, 4), time(18, 5), ('wpc', 'wpc', 'abs', 'wpc')),
            (time(18, 5), time(18, 7), ('wpc', 'abs', 'abs', 'wpc')),
            (time(18, 7), time(18, 8), ('wpc', 'wpc', 'abs', 'wpc')),
            (time(18, 8), time(18, 14), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(18, 14), time(18, 22), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(18, 22), time(18, 25), ('abs', 'abs', 'abs', 'abs')),  # dinner delivery
            (time(18, 25), time(18, 47), ('abs', 'abs', 'wpc', 'wpc')),
            (time(18, 47), time(18, 50), ('abs', 'abs', 'abs', 'wpc')),
            (time(18, 50), time(18, 57), ('abs', 'abs', 'wpc', 'wpc')),
            (time(18, 57), time(18, 58), ('abs', 'abs', 'wpc', 'abs')),  # 해원 get off work
            (time(18, 58), time(18, 58, 30), ('abs', 'wpc', 'wpc', 'abs')),
            (time(18, 58, 30), time(19, 8), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(19, 8), time(20, 0), ('wpc', 'wpc', 'abs', 'abs')),  # 지인 get off work
            (time(20, 0), time(20, 30), ('abs', 'wpc', 'abs', 'abs')),  # 승남 get off work
            (time(20, 30), time(23, 59), ('abs', 'abs', 'abs', 'abs')),  # 희훈 get off work
        ]

        # 2025-02-14 Time intervals (Friday)
        test_20250214_time_interval = [
            (time(9, 27), time(9, 47), ('wpc', 'abs', 'abs', 'abs')),  # 승남 출근
            (time(9, 47), time(10, 45), ('wpc', 'wpc', 'abs', 'abs')),  # 희훈 출근
            (time(10, 45), time(10, 47), ('wpc', 'wpc', 'wpc', 'abs')),  # 지인 출근
            (time(10, 47), time(11, 0), ('wpc', 'wpc', 'abs', 'abs')),  # 지인 미팅
            (time(11, 0), time(11, 5), ('wpc', 'wpc', 'wpc', 'abs')),  # 지인 미팅 끝
            (time(11, 5), time(11, 5, 30), ('abs', 'wpc', 'wpc', 'abs')),
            (time(11, 5, 30), time(11, 10), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(11, 10), time(11, 16), ('abs', 'wpc', 'abs', 'abs')),  # 지인, 승남 나감
            (time(11, 16), time(11, 31), ('abs', 'wpc', 'wpc', 'abs')),  # 지인 들어옴
            (time(11, 31), time(11, 32), ('abs', 'wpc', 'abs', 'abs')),  # 희훈 나감
            (time(11, 32), time(11, 34), ('abs', 'wpc', 'wpc', 'abs')),  # 승남, 희훈 들어옴
            (time(11, 34), time(11, 39), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(11, 39), time(11, 47), ('wpc', 'wpc', 'wpc', 'wpc')),  # 해원 출근
            (time(11, 47), time(12, 27), ('abs', 'abs', 'wpc', 'wpc')),  # 승남, 희훈 lunch
            (time(12, 27), time(12, 32), ('abs', 'abs', 'abs', 'wpc')),
            (time(12, 32), time(13, 33), ('abs', 'abs', 'wpc', 'wpc')),
            (time(13, 33), time(13, 37), ('wpc', 'wpc', 'wpc', 'wpc')),  # 승남, 희훈 in
            (time(13, 37), time(13, 39), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(13, 39), time(13, 40), ('wpc', 'wpc', 'wpc', 'wpc')),  # +희훈
            (time(13, 40), time(13, 58), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(13, 58), time(14, 48), ('abs', 'abs', 'abs', 'abs')),  # study
            (time(14, 48), time(14, 53), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(14, 53), time(14, 54), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(14, 54), time(14, 55), ('wpc', 'abs', 'wpc', 'abs')),
            (time(14, 55), time(14, 56), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(14, 56), time(14, 58), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(14, 58), time(15, 37), ('abs', 'wpc', 'wpc', 'abs')),  # 승남, 해원 meeting
            (time(15, 37), time(15, 38), ('abs', 'abs', 'wpc', 'abs')),
            (time(15, 38), time(15, 38, 30), ('abs', 'wpc', 'wpc', 'abs')),
            (time(15, 38, 30), time(15, 52), ('abs', 'abs', 'wpc', 'abs')),
            (time(15, 52), time(16, 0), ('abs', 'abs', 'abs', 'abs')),
            (time(16, 0), time(16, 32), ('abs', 'abs', 'wpc', 'abs')),
            (time(16, 32), time(16, 34), ('wpc', 'abs', 'wpc', 'wpc')),  # 승남, 해원 in
            (time(16, 34), time(16, 35), ('wpc', 'abs', 'abs', 'wpc')),
            (time(16, 35), time(16, 41), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(16, 41), time(16, 43), ('wpc', 'abs', 'abs', 'wpc')),  # 지인 meeting
            (time(16, 43), time(16, 49), ('wpc', 'abs', 'wpc', 'wpc')),  # 지인 in
            (time(16, 49), time(16, 51), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(16, 51), time(16, 53), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(16, 53), time(17, 2), ('wpc', 'abs', 'wpc', 'wpc')),  # 희훈 get off work
            (time(17, 2), time(17, 5), ('wpc', 'abs', 'wpc', 'abs')),
            (time(17, 5), time(17, 8), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(17, 8), time(17, 11), ('wpc', 'abs', 'wpc', 'abs')),  # 해원 get off work
            (time(17, 11), time(17, 14), ('abs', 'abs', 'wpc', 'abs')),
            (time(17, 14), time(18, 0), ('wpc', 'abs', 'wpc', 'abs')),
            (time(18, 0), time(18, 11), ('wpc', 'abs', 'abs', 'abs')),
            (time(18, 11), time(18, 15), ('wpc', 'abs', 'wpc', 'abs')),
            (time(18, 15), time(18, 17), ('wpc', 'abs', 'abs', 'abs')),  # 지인 get off work
            (time(18, 17), time(23, 59), ('abs', 'abs', 'abs', 'abs')),  # 승남 get off work
        ]

        # 2025-02-15, 2025-02-16 has no record due to weekend
        test_20250215_time_interval = []
        test_20250216_time_interval = []

        # 2025-02-17 Time intervals (Monday)
        test_20250217_time_interval = [
            (time(10, 26), time(10, 28), ('wpc', 'wpc', 'wpc', 'abs')),  # 지인 go to work
            (time(10, 28), time(10, 34), ('wpc', 'wpc', 'abs', 'abs')),
            (time(10, 34), time(10, 56), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(10, 56), time(11, 12), ('abs', 'wpc', 'wpc', 'abs')),
            (time(11, 12), time(11, 31), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(11, 31), time(11, 32), ('wpc', 'abs', 'wpc', 'abs')),
            (time(11, 32), time(11, 43), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(11, 43), time(11, 46), ('wpc', 'wpc', 'abs', 'abs')),
            (time(11, 46), time(11, 47), ('abs', 'abs', 'abs', 'abs')),  # 승남, 희훈 lunch(out)
            (time(11, 47), time(12, 7), ('abs', 'abs', 'wpc', 'abs')),
            (time(12, 7), time(12, 16), ('abs', 'abs', 'abs', 'abs')),
            (time(12, 16), time(12, 39), ('abs', 'abs', 'wpc', 'abs')),
            (time(12, 39), time(12, 39, 30), ('abs', 'wpc', 'wpc', 'abs')),
            (time(12, 39, 30), time(12, 44), ('abs', 'abs', 'wpc', 'abs')),
            (time(12, 44), time(12, 51), ('abs', 'abs', 'wpc', 'wpc')),  # 해원 go to work
            (time(12, 51), time(13, 30), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(13, 30), time(13, 34), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(13, 34), time(13, 35), ('wpc', 'abs', 'wpc', 'wpc')),  # 희훈 나감
            (time(13, 35), time(13, 42), ('wpc', 'wpc', 'wpc', 'wpc')),  # 희훈 들어옴
            (time(13, 42), time(13, 46), ('wpc', 'wpc', 'wpc', 'abs')),  # 해원 나감
            (time(13, 46), time(14, 22), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(14, 22), time(14, 24), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(14, 24), time(14, 50), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(14, 50), time(14, 54), ('wpc', 'wpc', 'wpc', 'abs')),
            (time(14, 54), time(15, 5), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(15, 5), time(15, 9), ('wpc', 'wpc', 'abs', 'wpc')),
            (time(15, 9), time(15, 16), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(15, 16), time(15, 17), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(15, 17), time(15, 20), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(15, 20), time(16, 3), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(16, 3), time(16, 4), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(16, 4), time(16, 6), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(16, 6), time(16, 24), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(16, 24), time(16, 35), ('wpc', 'wpc', 'abs', 'wpc')),  # 지인 lunch(out)
            (time(16, 35), time(16, 40), ('abs', 'wpc', 'abs', 'wpc')),
            (time(16, 40), time(16, 43), ('abs', 'wpc', 'wpc', 'wpc')),
            (time(16, 43), time(16, 44), ('abs', 'wpc', 'abs', 'wpc')),
            (time(16, 44), time(16, 47), ('wpc', 'wpc', 'abs', 'wpc')),
            (time(16, 47), time(16, 51), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(16, 51), time(16, 53), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(16, 53), time(16, 55), ('wpc', 'wpc', 'wpc', 'wpc')),
            (time(16, 55), time(17, 0), ('wpc', 'abs', 'wpc', 'wpc')),  # 희훈 get off work
            (time(17, 0), time(17, 1), ('abs', 'abs', 'wpc', 'wpc')),
            (time(17, 1), time(17, 15), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(17, 15), time(18, 2), ('abs', 'abs', 'wpc', 'wpc')),  # 승남 meeting
            (time(18, 2), time(18, 6), ('abs', 'abs', 'abs', 'wpc')),
            (time(18, 6), time(18, 13), ('abs', 'abs', 'wpc', 'wpc')),
            (time(18, 13), time(18, 14), ('wpc', 'abs', 'wpc', 'wpc')),
            (time(18, 14), time(18, 15), ('abs', 'abs', 'wpc', 'wpc')),
            (time(18, 15), time(18, 17), ('wpc', 'abs', 'wpc', 'wpc')),  # 측정 종료
        ]

        self.test_time_interval.extend([
            [test_20250213_time_interval, test_20250214_time_interval, test_20250215_time_interval,
             test_20250216_time_interval, test_20250217_time_interval]
        ])
        self.time_interval = self.test_time_interval[self.num_test - 1]

    def _create_time_mask(self, df: pd.DataFrame, start_time: time, end_time: time) -> pd.Series:
        """Create a boolean mask for time interval."""
        if isinstance(start_time, tuple):
            start_time = time(*start_time)
        if isinstance(end_time, tuple):
            end_time = time(*end_time)

        if hasattr(start_time, 'microsecond'):
            df_time = pd.to_datetime(df['Time'].astype(str)).dt.time
        else:
            df_time = df['Time']

        return (df_time >= start_time) & (df_time < end_time)

    def _assign_states(self, df: pd.DataFrame, mask: pd.Series, states: tuple) -> None:
        """Assign states to user activity columns."""
        for i, state in enumerate(states):
            df.loc[mask, f'user{i + 1}_activity'] = state

    def get_current_state(self, row: pd.Series) -> str:
        """Get the current state from user activities."""
        activities = tuple(row[f'user{i + 1}_activity'] for i in range(self.num_users))
        return self.state_ids.get(activities, 'Unknown')

    def load_test_data(self) -> pd.DataFrame:
        """Load data from all test directories."""
        all_device_dfs = {}

        for test_date in self.test_dates:
            test_data_path = os.path.join(self.test_data_path, test_date)

            if not os.path.exists(test_data_path):
                print(f"Warning: Directory {test_data_path} does not exist. Skipping...")
                continue

            files = [file for file in os.listdir(test_data_path) if file.endswith('.csv')]
            for file in files:
                data_file_path = os.path.join(test_data_path, file)
                df = pd.read_csv(data_file_path)
                device_id = file.split('_')[0]

                df['Appliance'] = self.appliances_mapping.get(device_id, "Unknown")

                if device_id not in all_device_dfs:
                    all_device_dfs[device_id] = []
                all_device_dfs[device_id].append(df)

        device_dfs = []
        for device in sorted(all_device_dfs.keys()):
            device_df = pd.concat(all_device_dfs[device], ignore_index=True)
            device_df['Timestamp'] = pd.to_datetime(device_df['Timestamp'])
            device_df = device_df.sort_values('Timestamp')
            device_dfs.append(device_df)

        final_df = pd.concat(device_dfs, ignore_index=True)
        return final_df

    def preprocess_test_data(self) -> pd.DataFrame:
        """Load and preprocess test data including state assignments."""
        df = self.load_test_data()

        for i in range(self.num_users):
            df[f'user{i + 1}_activity'] = None

        if not self.time_interval:
            print("No time interval definitions available.")
            return df

        df['Time'] = df['Timestamp'].dt.time
        df['Date'] = df['Timestamp'].dt.date

        dates = sorted(df['Date'].unique())
        dfs_by_date = {}

        for date, day_interval in zip(dates, self.time_interval):
            df_day = df[df['Date'] == date].copy()
            for start_time, end_time, states in day_interval:
                mask = self._create_time_mask(df_day, start_time, end_time)
                self._assign_states(df_day, mask, states)
            dfs_by_date[date] = df_day

        devices = sorted(df['Appliance'].unique())
        final_dfs = []

        for device in devices:
            device_dfs = []
            for date in dates:
                if date in dfs_by_date:
                    device_df = dfs_by_date[date][dfs_by_date[date]['Appliance'] == device]
                    device_dfs.append(device_df)
            if device_dfs:
                combined_device_df = pd.concat(device_dfs).sort_values('Timestamp')
                final_dfs.append(combined_device_df)

        df = pd.concat(final_dfs, ignore_index=True)

        power_cols = [col for col in df.columns if 'Power (W)' in col]
        if power_cols:
            df['Total Power (W)'] = df[power_cols].sum(axis=1)
        df['Unix Time'] = (pd.to_datetime(df['Timestamp']).astype('int64') // 10 ** 9)

        df.drop(columns=['Time', 'Date'], inplace=True)
        df['state'] = df.apply(self.get_current_state, axis=1)

        self._save_test_data(df)
        return df

    def _save_test_data(self, df: pd.DataFrame) -> None:
        """Save processed test data to CSV file."""
        if self.args.save_data_path:
            save_dir = Path(self.args.save_data_path) / 'test'
            save_dir.mkdir(parents=True, exist_ok=True)

            start_date = df['Timestamp'].min().strftime('%Y%m%d')
            end_date = df['Timestamp'].max().strftime('%Y%m%d')
            filename = f"test{self.num_test}_processed_data_{start_date}_{end_date}.csv"
            file_path = save_dir / filename

            df.to_csv(file_path, index=False)
            print(f"Test data saved to: {file_path}")

    def plot_state_distribution(self, save_vis_path: Optional[str] = None) -> None:
        """Plot state distribution visualizations."""
        save_path = save_vis_path or self.args.save_vis_path
        df = self.preprocess_test_data()
        self._plot_state_distribution(df, save_path)

    def _plot_state_distribution(self, df: pd.DataFrame, save_path: str) -> None:
        """Plot the distribution of states in a pie chart."""
        sns.set_theme(style="whitegrid")
        state_counts = df['state'].value_counts()

        fig, ax = plt.subplots(figsize=(15, 10))
        colors = sns.color_palette("husl", n_colors=len(state_counts))

        patches, texts, autotexts = ax.pie(
            state_counts,
            labels=state_counts.index,
            autopct='%1.1f%%',
            pctdistance=0.75,
            labeldistance=1.2,
            colors=colors,
            startangle=90
        )

        # Format autopct labels
        for i, autotext in enumerate(autotexts):
            count = state_counts.iloc[i]
            autotext.set_text(f'{count:,d}\n({autotext.get_text()})')

        plt.title(f'Test #{self.num_test} State Distribution', pad=20, size=16)

        # Create legend with state descriptions
        id_to_state = {v: k for k, v in self.state_ids.items()}
        legend_labels = []
        for state_id in state_counts.index:
            if state_id in id_to_state:
                state_tuple = id_to_state[state_id]
                state_str = ' | '.join(str(s) for s in state_tuple)
                legend_labels.append(f'{state_id}: {state_str}')
            else:
                legend_labels.append(state_id)

        plt.legend(
            patches,
            legend_labels,
            title="States",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize='small',
            title_fontsize='medium'
        )

        plt.tight_layout()
        output_file = os.path.join(save_path, f"test{self.num_test}_state_distribution.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved state distribution pie chart: {output_file}")


if __name__ == "__main__":
    # args.num_dc = 5
    # dc_loader = DataCollectionLoader(args)
    # df = dc_loader.load_preprocess()
    # print(df.head())
    # print(len(dc_loader.states))
    # print(dc_loader.state_ids)
    # dc_loader.plot_power_consumption()
    # dc_loader.plot_state_distribution()

    args.num_test = 1
    test_loader = TestDataCollectionLoader(args)
    test_df = test_loader.preprocess_test_data()
    print(test_df.head())
    print(len(test_loader.states))
    test_loader.plot_state_distribution()
