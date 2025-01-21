# Semantic State Prediction with Power Consumption Data
Semantic state prediction with power consumption data collected by smart plugs.


## 🖥️ Code:: Usage

`pip install -r requirements.txt`

(TBU)

### Directory Structure
The overall structure of the project is as follows:

```
semanticProject/
    ├── channel/
    ├── data/
    ├── dataloader/
    ├── docs/
    ├── models/
    ├── outputs/
    ├── scripts/
    ├── trainer/
    ├── utils/
    ├── config.py
    ├── README.md
    └── requirements.txt
```

- **It is recommended not to modify the directory structure.**
- If you wish to make changes, please ensure to update the paths in the `config.py` file:
  - `self.root_path`
  - `self.save_path`

### Configuration

The `config.py` file allows you to customize various aspects of the project through command-line arguments. Below is a categorized summary of the arguments you can use:

**Environment Arguments**
- `--debug`  
  Enable debug mode. *(default: True)*
  
- `--device`  
  Device to be used for training (e.g., CPU, GPU). *(default: 'cuda')*

- `-rand`, `--random_state`  
  Random state for reproducibility. *(default: 42)*

- `--data_dir_path`  
  Path to the data directory. *(default: `<root_path>/data`)*

- `--save_data_path`  
  Path to save output datasets. *(default: `<save_path>/dataset`)*

- `--save_vis_path`  
  Path to save visualizations. *(default: `<save_path>/visualization`)*

- `--save_model_path`  
  Path to save model checkpoints. *(default: `<save_path>/checkpoints`)*

**Data Arguments**
- `-dc`, `--num_dc`  
  Number of data collections. *(choices: [1, 2, 3, 4, 5], default: 4)*

- `-s`, `--scaler_type`  
  Type of data scaler. *(choices: ["standard", "minmax"], default: "standard")*

- `-eq`, `--make_equal_dist`  
  Ensure equal data distribution. *(default: True)*

- `-agg`, `--aggregate`  
  Use aggregated dataset. *(default: True)*

**Model Arguments**
- `-i`, `--input_size`  
  Input size for the model. *(choices: [2, 7], default: 2)*

- `-c`, `--num_classes`  
  Number of output classes for the model. *(default: 343)*

- `-b`, `--batch_size`  
  Mini-batch size for training. *(default: 32)*

- `-st`, `--stride`  
  Stride length for data. *(default: 1)*

- `-nepoch`, `--num_epochs`  
  Number of training epochs. *(default: 500)*

- `-lr`, `--learning_rate`  
  Learning rate for the optimizer. *(default: 0.001)*

- `-p`, `--patience`  
  Early stopping patience. *(default: 15)*

- `-vs`, `--val_size`  
  Validation set size as a fraction of the dataset. *(default: 0.1)*

- `-ts`, `--test_size`  
  Test set size as a fraction of the dataset. *(default: 0.2)*

- `-seq_len`, `--sequence_length`  
  Sequence length for training. *(default: 50)*

- `--class_names`  
  List of class names. *(default: `[]`)*

- `--model_name`  
  Name of the deep learning model. *(choices: ["CNN-RNN", "CNN-LSTM", "DAE-RNN", "DAE-LSTM"])*

**Embedding Arguments**
- `--embedding_dim`  
  Embedding dimension for semantic embeddings. *(default: 50)*

**Channel Arguments**
- `--snr_db`  
  Signal-to-noise ratio (SNR) in dB. *(default: 20)*

- `--channel_name`  
  Channel environment type. *(choices: ["AWGN", "Rayleigh", "Rician", "Nakagami"], default: "AWGN")*

- `--doppler_freq`  
  Doppler frequency. *(default: 0)*

- `--k_factor`  
  K-factor of the Rician channel. *(default: 1.0)*

- `--variance`  
  Variance of the Rician channel. *(default: 1.0)*

- `--m_factor`  
  M-factor of the Nakagami channel. *(default: 1.0)*

- `--omega`  
  Average power (omega) of the Nakagami channel. *(default: 1.0)*


Make sure to use these arguments as needed when running scripts to configure the project behavior.


## 🖥️ Code:: Explanation
(TBU)


## 🗃️ Data Collection:: 2nd Test Data Acquisition

> ### 🔎 Notes
> 
> **Behavior Details for Data Collection**
> - **Monitor:** Put into sleep mode when not in use.
> - **Fridge:** Simulate real usage by opening and closing at specific intervals repeatedly.
> - **Camera:** Use recordings for labeling purposes.
> 
> **Additional Considerations**
> - Collect data for over an hour to verify energy (Wh) values.
> - Use average power consumption of each user's monitor and PC as a signature.
> - Leverage intermediate data (e.g., door, power) instead of using aggregated power data directly.
> 
> **Issues Identified**
> - Jee-in's PC could not enter sleep mode during data collection, so "working w/o PC" and refrigerator or meeting-related tasks were not power-efficient.
>   - Improvement: Use another PC for data collection in future tests.
> - Hae-won's PC often failed to turn on properly.

### Experimental Setup

| Type            | Content                             |
|-----------------|-------------------------------------|
| Date            | November 28, 2024                   |
| Time            | 3:15 PM - 4:30 PM                   |
| Participants    | Kim Jee-in, Lee Hae-won, Kim Yu-rim |
| Location        | GIST Building C10 Room 207          |
| Behavior Labels | 6 types                             |
| Smart Meters    | 7 units                             |
| Base Sensors    | 2 units                             |
| Reference       | Camera                              |

![Experimental Environment Map](docs%2Fdc2_experimental_environment_map.png)

### Behavior and Device Mapping

| Behavior            | Jee-in      | Hae-won         | Yu-rim         | Notes                                   |
|---------------------|-------------|-----------------|----------------|-----------------------------------------|
| Working with PC     | SMP04 SMP05 | SMP03 SMP14    | SMP12          | Yu-rim: Using a charging laptop.       |
| Working without PC  | -           | -               | -              | Ji-in only, PCs set to sleep mode.     |
| Lunch / Coffee Break| SMP01       | SMP01           | SMP01          | Fridge usage simulation: open 30s, close 1 min, repeat. PCs in sleep mode (except Ji-in). |
| Having Meeting      | SMP02       | SMP02           | SMP02          | 10 min with projector and laptop. PCs in sleep mode (except Ji-in). |
| Go to Work          | Door Sensor | Door Sensor     | Door Sensor    | -                                       |
| Get Off Work        | Door Sensor | Door Sensor     | Door Sensor    | -                                       |


### Scenario

![2nd Test Data Collection](docs%2Fdc2-scenario.png)


## 🗃️ Data Collection:: 4th Test Data Acquisition

> ### 🔎 Notes
>
> - **15:25 - 15:35:** During Jee-in's "Lunch" activity, movement detected near the front bookshelf.
> - **15:35:** Hae-won exits during "Absence" activity, with a slight pause.

### Experimental Setup

| Type            | Content                    |
|-----------------|----------------------------|
| Date            | January 8, 2025            |
| Time            | 2:00 PM - 4:00 PM          |
| Participants    | Kim Jee-in, Lee Hae-won    |
| Location        | GIST Building C10 Room 207 |
| Behavior Labels | 4 types                    |
| Smart Meters    | 5 units                    |
| Base Sensors    | 2 units                    |
| Reference       | Camera                     |

#### States
- S0 ~ S11 (12 states)

| Index | State         |
|-------|---------------|
| S0    | (abs, abs)    |
| S1    | (wpc, abs)    |
| S2    | (abs, wpc)    |
| S3    | (wpc, wpc)    |
| S4    | (wopc, abs)   |
| S5    | (abs, wopc)   |
| S6    | (wopc, wpc)   |
| S7    | (wpc, wopc)   |
| S8    | (l, abs)      |
| S9    | (abs, l)      |
| S10   | (l, wpc)      |
| S11   | (wpc, l)      |

#### State Machine Graph

![State Machine Graph](docs%2Fdc4_state_machine_graph.png)

### Scenario

- **Duration:** 120 minutes (2 hours)
- **Data Collection Per State:** 10 minutes each (to address data imbalance)

| Time (min) | 5   | 5   | 5   | 10  | 10  | 10  | 10  | 10  | 5   | 10  | 10  | 10  | 10  | 5   | 5   |
|------------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| Jee-in     | abs | wpc | wpc | wpc | wpc | abs | abs | abs | wpc | wopc | l   | l   | wopc | wpc | abs |
| Hae-won    | abs | abs | wpc | wopc| l   | l   | wopc| wpc | wpc | wpc | wpc | abs | abs | abs | abs |
| State      | S0  | S1  | S3  | S7  | S11 | S9  | S5  | S2  | S3  | S6  | S10 | S8  | S4  | S1  | S0  |

![4th Test Data Collection](docs%2Fdc4-scenario.png)

### Behavior Details
- Before "Lunch," PCs are set to sleep mode.
- **Working without PC:** PCs are in sleep mode.
- **Absence:** PCs and monitors are powered off.
- **Lunch:** Fridge and freezer doors are opened, kept open for 1 minute, then closed for 4 minutes.
- **Working with PC:** Model training is conducted in the background, utilizing the CPU (Jee-In).

### Data Visualization
![Combined Power Usage](outputs%2Fvisualization%2Fdc4_combined_power_usage.png)


## 🗃️ Data Collection:: 5th Test Data Acquisition

