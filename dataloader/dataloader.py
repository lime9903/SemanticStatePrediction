import os
import pandas as pd


class ApplianceDataLoader:
    def __init__(self):
        self.appliances_mapping = []
        self._initialize_mappings()

    def _initialize_mappings(self):
        # Data collection #1
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

        # Data collection #2
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

        # Data collection #3
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

        # Data collection #4
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

        self.appliances_mapping.extend([
            dc1_appliances_mapping,
            dc2_appliances_mapping,
            dc3_appliances_mapping,
            dc4_appliances_mapping,
        ])

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
            final_df['Appliance'] = self.appliances_mapping[0].get(device_id, "Unknown")

            dfs.append(final_df)

        return dfs

    def load_data(self, num_dc):
        assert num_dc in [1, 2, 3, 4], "num_dc must be 1, 2, 3 or 4."

        # TODO: change the path for your relative data directory path
        dc_data_dirs = [
            r'C:\Users\lime9\PycharmProjects\semanticProject\data\20241122',
            r'C:\Users\lime9\PycharmProjects\semanticProject\data\20241128',
            r'C:\Users\lime9\PycharmProjects\semanticProject\data\20241205',
            r'C:\Users\lime9\PycharmProjects\semanticProject\data\20250108'
        ]

        data_dir = dc_data_dirs[num_dc - 1]
        files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
        dfs = []

        if num_dc == 1:
            power_files = [file for file in files if "power" in file]
            energy_files = [file for file in files if "energy" in file]
            dfs = self.dc1_load(data_dir, power_files, energy_files)
        else:
            for file in files:
                data_file_path = os.path.join(data_dir, file)
                df = pd.read_csv(data_file_path)
                device_id = file.split('_')[0]
                df['Appliance'] = self.appliances_mapping[num_dc - 1].get(device_id, "Unknown")
                dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])

        return combined_df


if __name__ == '__main__':
    df = ApplianceDataLoader().load_data(4)
    print(df.head())
