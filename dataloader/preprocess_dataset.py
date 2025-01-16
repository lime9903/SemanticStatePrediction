import os
import pandas as pd
from datetime import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import product


# TODO: split the data for time series
class DataCollectionLoader:
    def __init__(self, args):
        assert args.num_dc in [1, 2, 3, 4], "num_dc must be 1, 2, 3 or 4."
        assert Path(args.data_dir_path).exists(), "data_dir_path does not exist."
        self.args = args
        self.num_dc = args.num_dc
        self.data_dir_path = args.data_dir_path
        self.dc_appliances_mapping = []  # total dc appliances mapping
        self.dc_activities_mapping = []  # total dc activities mapping
        self.appliances_mapping = None   # specified appliances mapping
        self.activities_mapping = None   # specified activities mapping
        self.dc_actions = []             # total dc actions
        self.actions = None              # specified dc actions
        self.dc_users = None             # total dc users
        self.dc_num_users = []           # total dc number of users
        self.users = None                # specified users
        self.num_users = None            # specified number of users
        self.dc_states = []              # total dc states
        self.dc_state_ids = []           # total dc state indices
        self.states = None               # specified states
        self.state_ids = None            # specified state indices
        self.dc_time_interval = []       # total dc time intervals
        self.time_interval = None        # specified time interval
        self._initialize_parameters()

    def _initialize_parameters(self):
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

        # Total appliances mapping list
        self.dc_appliances_mapping.extend([
            dc1_appliances_mapping,
            dc2_appliances_mapping,
            dc3_appliances_mapping,
            dc4_appliances_mapping,
        ])
        self.appliances_mapping = self.dc_appliances_mapping[self.num_dc - 1]

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

        # Total activities mapping list
        self.dc_activities_mapping.extend([
            dc1_activities_mapping,
            dc2_activities_mapping,
            dc3_activities_mapping,
            dc4_activities_mapping,
        ])
        self.activities_mapping = self.dc_activities_mapping[self.num_dc - 1]

        for activity in self.dc_activities_mapping:
            self.dc_actions.append(list(activity.keys()))
        self.actions = self.dc_actions[self.num_dc - 1]

        # Participate users for each dc
        self.dc_users = [
            ['JI', 'HW', 'YR'],
            ['JI', 'HW', 'YR'],
            ['JI', 'YR'],
            ['JI', 'HW']
        ]
        for user in self.dc_users:
            self.dc_num_users.append(len(user))
        self.users = self.dc_users[self.num_dc - 1]
        self.num_users = self.dc_num_users[self.num_dc - 1]

        # TODO: need to adjust for dc4 states (assume: limited states)
        for idx, actions in enumerate(self.dc_actions):
            states = list(product(actions, repeat=len(self.dc_users[idx])))
            state_ids = {state: f"S{index}" for index, state in enumerate(states)}
            self.dc_states.append(states)
            self.dc_state_ids.append(state_ids)
        self.states = self.dc_states[self.num_dc - 1]
        self.state_ids = self.dc_state_ids[self.num_dc - 1]

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

        self.dc_time_interval.extend([
            dc1_time_interval,
            dc2_time_interval,
            dc3_time_interval,
            dc4_time_interval,
        ])
        self.time_interval = self.dc_time_interval[self.num_dc - 1]

    def dc1_load(self, data_dir, power_files, energy_files):
        assert len(power_files) == len(energy_files), "There should be the same number of power and energy files."
        num_appliances = len(power_files)
        dfs = []

        for idx in range(num_appliances):
            power_file = power_files[idx]
            energy_file = energy_files[idx]
            power_file_path = os.path.join(data_dir, power_file)
            energy_file_path = os.path.join(data_dir, energy_file)
            device_id = power_file.split('_')[0]

            power_df = pd.read_csv(str(power_file_path))
            energy_df = pd.read_csv(str(energy_file_path))

            power_df['Timestamp'] = pd.to_datetime(power_df['Time'])
            energy_df['Timestamp'] = pd.to_datetime(energy_df['Time'])
            power_df.rename(columns={'Value': 'Power (W)'}, inplace=True)
            energy_df.rename(columns={'Value': 'Energy (Wh)'}, inplace=True)

            df = pd.merge_asof(
                power_df.sort_values('Timestamp'),
                energy_df.sort_values('Timestamp'),
                on='Timestamp',
                direction='nearest'
            )

            final_df = pd.DataFrame()
            final_df['Timestamp'] = df['Timestamp']
            final_df['Label'] = df['Device Label_x']
            final_df['Location Name'] = df['Location_x']
            final_df['Power (W)'] = df['Power (W)']
            final_df['Energy (Wh)'] = df['Energy (Wh)']
            final_df['Appliance'] = self.appliances_mapping.get(device_id, "Unknown")

            dfs.append(final_df)

        return dfs

    def load_data(self):
        # TODO: change the path for your relative data directory path
        dc_data_dirs = [
            '20241122',  # dc 1
            '20241128',  # dc 2
            '20241205',  # dc 3
            '20250108'   # dc 4
        ]

        data_dir = dc_data_dirs[self.num_dc - 1]
        data_path = os.path.join(self.data_dir_path, data_dir)
        files = [file for file in os.listdir(data_path) if file.endswith('.csv')]
        dfs = []

        if self.num_dc == 1:
            power_files = [file for file in files if "power" in file]
            energy_files = [file for file in files if "energy" in file]
            dfs = self.dc1_load(data_path, power_files, energy_files)
        else:
            for file in files:
                data_file_path = os.path.join(data_path, file)
                df = pd.read_csv(data_file_path)
                device_id = file.split('_')[0]
                df['Appliance'] = self.appliances_mapping.get(device_id, "Unknown")
                dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])

        return combined_df

    def plot_power_consumption(self, save_vis_path):
        df = self.load_data()
        sns.set(style="whitegrid")
        unique_labels = df['Label'].unique()
        palette = sns.color_palette("husl", len(unique_labels))
        plt.figure(figsize=(15, 8))

        for i, label in enumerate(unique_labels):
            label_df = df[df['Label'] == label]
            plt.plot(label_df['Timestamp'], label_df['Power (W)'], label=f'{label} Power', color=palette[i],
                     alpha=0.7)
            plt.fill_between(label_df['Timestamp'], label_df['Power (W)'], color=palette[i], alpha=0.2)

        plt.xlabel('Timestamp')
        plt.ylabel('Power (W)')
        plt.title('Power Usage Over Time by Label')
        plt.legend()
        plt.grid(alpha=0.3)

        combined_output_file = os.path.join(save_vis_path, f"dc{self.num_dc}_combined_power_usage.png")
        plt.savefig(combined_output_file, dpi=300)
        plt.show()
        print(f"Saved combined graph: {combined_output_file}")

        n_cols = 5
        n_rows = -(-len(unique_labels) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()

        for i, label in enumerate(unique_labels):
            label_df = df[df['Label'] == label]
            ax = axes[i]
            ax.plot(label_df['Timestamp'], label_df['Power (W)'], label='Power (W)', color=palette[i], alpha=0.7)
            ax.fill_between(label_df['Timestamp'], label_df['Power (W)'], color=palette[i], alpha=0.2)
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Power (W)')
            ax.set_title(f'Power Usage - {label}')
            ax.legend()
            ax.grid(alpha=0.3)

        for j in range(len(unique_labels), len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

        for i, label in enumerate(unique_labels):
            plt.figure(figsize=(12, 6))
            label_df = df[df['Label'] == label]
            plt.plot(label_df['Timestamp'], label_df['Power (W)'], label='Power (W)', color=palette[i], alpha=0.7)
            plt.fill_between(label_df['Timestamp'], label_df['Power (W)'], color=palette[i], alpha=0.2)
            plt.xlabel('Timestamp')
            plt.ylabel('Power (W)')
            plt.title(f'Power Usage Over Time - {label}')
            plt.legend()
            plt.grid(alpha=0.3)

            individual_output_file = os.path.join(save_vis_path, f"dc{self.num_dc}_{label}_xtime_ypower.png")
            plt.savefig(individual_output_file, dpi=300)
            plt.close()
            print(f"Saved graph: {individual_output_file}")

    def load_preprocess(self):
        """
        Assign states to the DataFrame based on the given time intervals and corresponding states.
        Also calculates total power consumption and adds Unix timestamp.
        """
        df = self.load_data()

        for i in range(self.num_users):
            df[f'user{i + 1}_activity'] = None

        # If time interval is empty, end assigning states
        if not self.time_interval:
            print("There is no definition of the time interval.")
            return df

        df['Time'] = df['Timestamp'].dt.time
        for start_time, end_time, states in self.time_interval:
            if start_time is None:
                mask = df['Time'] < end_time
            elif end_time is None:
                mask = df['Time'] >= start_time
            else:
                mask = df['Time'].between(start_time, end_time)

            for user_idx, state in enumerate(states):
                if user_idx < self.num_users:
                    df.loc[mask, f'user{user_idx + 1}_activity'] = state

        power_cols = [col for col in df.columns if 'Power (W)' in col]
        if power_cols:
            df['Total Power (W)'] = df[power_cols].sum(axis=1)

        df['Unix Time'] = (pd.to_datetime(df['Timestamp']).astype('int64') // 10 ** 9)

        df.drop(columns=['Time'], inplace=True)
        df['state'] = df.apply(self.get_current_state, axis=1)

        if self.args.make_equal_dist:
            df = df[df['state'] != 'Unknown']

        self.save_to_csv(df, save_data_path=self.args.save_data_path)

        return df

    def get_state_id(self, activities):
        """
        Get the state ID for a given combination of activities.
        """
        activity_tuple = tuple(activities)
        return self.state_ids.get(activity_tuple, "Unknown")

    def get_current_state(self, row):
        """
        Get the current state ID based on all users' activities.
        """
        activities = []
        for i in range(self.num_users):
            activity = row[f'user{i + 1}_activity']
            activities.append(activity)

        return self.get_state_id(activities)

    def save_to_csv(self, df, save_data_path=None):
        """
        Save the processed DataFrame to a CSV file.
        """
        if save_data_path:
            save_dir = Path(save_data_path)
        else:
            save_dir = Path.cwd()

        save_dir.mkdir(parents=True, exist_ok=True)
        date_str = df['Timestamp'].min().strftime('%Y%m%d')
        filename = f"dc{self.num_dc}_processed_data_{date_str}.csv"
        file_path = save_dir / filename

        df.to_csv(file_path, index=False)
        print(f"Data saved to: {file_path}")

        return str(file_path)

    def plot_state_distribution(self, state_df):
        """
        Plot the distribution of states in a pie chart.
        """
        sns.set_theme(style="whitegrid")

        state_counts = state_df['state'].value_counts()

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

        for i, autotext in enumerate(autotexts):
            count = state_counts.iloc[i]
            autotext.set_text(f'{count:,d}\n({autotext.get_text()})')

        plt.title(f'DC{self.num_dc} State Distribution', pad=20, size=16)

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
        plt.show()

        return
