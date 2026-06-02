import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import os

# NOTE: Functions no longer used by the current analysis pipeline have been
# moved to spiracle_helper_functions_legacy.py (audit 2026-06-01).

#=============================================
# 1. DATA READING AND PREPROCESSING FUNCTIONS
#=============================================

def load_and_reshape_data(file_path, duration=None, sampling_rate=20000):
    """
    Load and reshape data from .mat or binary file.

    Parameters:
    - file_path: Path to the data file
    - duration: Duration in seconds to load (None loads entire file)
    - sampling_rate: Sampling rate of the data

    Returns:
    - data: Reshaped data array (9 x time_points)
    """
    try:
        if file_path.endswith('.mat'):
            mat_data = loadmat(file_path)
            data_var = None
            for key in mat_data.keys():
                if isinstance(mat_data[key], np.ndarray) and len(mat_data[key].shape) == 2:
                    if mat_data[key].shape[0] == 9 or mat_data[key].shape[1] == 9:
                        data_var = key
                        break

            if data_var is None:
                raise ValueError("Could not find appropriate data array in .mat file")

            data = mat_data[data_var]

            if data.shape[0] != 9:
                data = data.T

            if duration is not None:
                samples = int(duration * sampling_rate)
                data = data[:, :samples]

        else:
            if duration is not None:
                num_samples = 9 * int(duration * sampling_rate)
                with open(file_path, 'rb') as fid:
                    data = np.fromfile(fid, dtype=np.float64, count=num_samples)
            else:
                data = np.memmap(file_path, dtype=np.float64, mode='r')

            data = data.reshape((-1, 9)).T

        print(f"Loaded data shape: {data.shape}")
        return data

    except Exception as e:
        print(f"Error loading file: {e}")
        raise

def extract_traces(data):
    """
    Extract relevant traces from the data array.

    Returns:
    - wingbeat_freq: Data[3] containing wingbeat frequency
    - spike_raw: Data[5] containing EMG data
    - opto_stim: Data[6] containing optogenetic stimulus data
    - wing_beat_amplitude: Average of Data[1] and Data[2]
    - x_position: Data[4] containing x position
    """
    wing_beat_amplitude = (data[1] + data[2]) / 2  # Average of rows 1 and 2
    return data[3], data[5], data[6], wing_beat_amplitude, data[4], data[7]  # WBF, spike, opto, amplitude, x_pos, hutchens

def smooth_data(data, window_size=100):
    """
    Smooth data using a sliding window average.

    Parameters:
    - data: Input data array
    - window_size: Size of the smoothing window

    Returns:
    - Smoothed data array
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def transform_opto_stimulus(data7, sampling_rate):
    """
    Transform optogenetic stimulus data into binary signal.

    Parameters:
    - data7: Raw optogenetic stimulus data
    - sampling_rate: Sampling rate in Hz

    Returns:
    - Binary stimulus signal (0 or 1)
    """
    transformed_data7 = np.zeros_like(data7)
    stim_on = False
    stim_start_idx = 0
    max_pulse_gap_samples = int(30 * sampling_rate / 1000)

    block_lengths = []
    block_count = 0
    last_stim_end_idx = -1

    for i in range(1, len(data7)):
        if data7[i] >= 0.9 and not stim_on:
            if stim_start_idx == 0:
                stim_start_idx = i
            stim_on = True
        elif data7[i] < 1 and stim_on:
            stim_on = False
            last_stim_end_idx = i

        if (stim_start_idx != 0 and (last_stim_end_idx != -1) and
            (i - last_stim_end_idx > max_pulse_gap_samples)):
            if last_stim_end_idx > stim_start_idx:
                transformed_data7[stim_start_idx:last_stim_end_idx] = 1
                block_length = last_stim_end_idx - stim_start_idx
                block_lengths.append(block_length)
                block_count += 1
            stim_start_idx = 0

    if stim_start_idx != 0 and last_stim_end_idx > stim_start_idx:
        transformed_data7[stim_start_idx:last_stim_end_idx] = 1
        block_length = last_stim_end_idx - stim_start_idx
        block_lengths.append(block_length)
        block_count += 1

    block_lengths_in_seconds = [length / sampling_rate for length in block_lengths]
    print(f"Number of blocks detected: {block_count}")
    print(f"Lengths of detected blocks (in seconds): {block_lengths_in_seconds}")

    return transformed_data7

def detect_stim_onsets(transformed_opto_stim, sampling_rate):
    """
    Detect actual stimulus onset times from the opto signal.

    Parameters:
    - transformed_opto_stim: Binary stimulus signal
    - sampling_rate: Sampling rate in Hz

    Returns:
    - List of onset times in seconds
    """
    onsets = np.where(np.diff(transformed_opto_stim) > 0)[0]
    onset_times = onsets / sampling_rate
    return onset_times

def load_stim_orders(mat_file):
    """
    Load stimulus orders from .mat file.

    Parameters:
    - mat_file: Path to .mat file

    Returns:
    - List of stimulus durations for each session
    """
    try:
        mat_data = loadmat(mat_file)
        stim_orders = mat_data['allRandomizedStimOrders']

        # Convert to Python list format and flatten the structure
        parsed_orders = []
        for session in stim_orders:
            if isinstance(session[0], np.ndarray):
                # Extract each stimulus duration from the nested structure
                session_stims = [stim for stim in session[0][0]]
                parsed_orders.append(session_stims)
            else:
                parsed_orders.append(session.tolist())

        print(f"Loaded stimulus orders: {parsed_orders}")
        return parsed_orders
    except Exception as e:
        print(f"Error loading stimulus orders: {e}")
        return None

def match_stim_times(detected_onsets, stim_orders, session_duration=90):
    """
    Match detected stimulus onsets with planned stimulus durations and add 0ms stimuli.

    Parameters:
    - detected_onsets: List of detected onset times in seconds
    - stim_orders: List of planned stimulus durations for each session
    - session_duration: Duration of each session in seconds

    Returns:
    - List of tuples (onset_time, duration_ms) for all stimuli
    """
    all_stim_times = []
    detected_idx = 0
    current_time = 0

    for session_idx, session_stims in enumerate(stim_orders):
        n_stims = len(session_stims)
        interval = session_duration / (n_stims + 1)

        for stim_idx, stim_duration in enumerate(session_stims):
            planned_time = current_time + (stim_idx + 1) * interval

            if isinstance(stim_duration, (list, np.ndarray)):
                # If stim_duration is a list/array, extract the first non-zero value or use 0
                actual_duration = next((d for d in stim_duration if d != 0), 0)
            else:
                actual_duration = stim_duration

            if actual_duration == 0:
                # For 0ms stimuli, use the planned time
                all_stim_times.append((planned_time, 0))
            else:
                # For non-zero stimuli, use the nearest detected onset
                if detected_idx < len(detected_onsets):
                    # Find nearest detected onset to planned time
                    detected_time = detected_onsets[detected_idx]
                    if abs(detected_time - planned_time) < interval/2:  # Within half interval
                        all_stim_times.append((detected_time, actual_duration))
                        detected_idx += 1
                    else:
                        print(f"Warning: No matching detected onset for {actual_duration}ms stimulus at {planned_time}s")
                        all_stim_times.append((planned_time, actual_duration))

        current_time += session_duration

    return sorted(all_stim_times, key=lambda x: x[0])

#=============================================
# 2. BOUT DETECTION AND SAVING FUNCTIONS
#=============================================
def extract_bout(data_dict, stim_onset, stim_duration, pre_stim_time=5, post_stim_time=15, sampling_rate=20000):
    """
    Extract a data bout around a stimulus.

    Parameters:
    - data_dict: Dictionary containing all data traces
    - stim_onset: Stimulus onset time in seconds
    - stim_duration: Stimulus duration in seconds
    - pre_stim_time: Time before stimulus to include (seconds)
    - post_stim_time: Time after stimulus to include (seconds)
    - sampling_rate: Sampling rate of the data

    Returns:
    - Dictionary containing the extracted bout data
    """
    start_idx = max(0, int((stim_onset - pre_stim_time) * sampling_rate))
    end_idx = min(
        len(data_dict['Time (s)']),
        int((stim_onset + post_stim_time) * sampling_rate)
    )

    bout_dict = {}
    for key in data_dict:
        bout_dict[key] = data_dict[key][start_idx:end_idx]

    # Adjust time to be relative to stim onset
    bout_dict['Time (s)'] = bout_dict['Time (s)'] - stim_onset
    bout_dict['Stim Duration (s)'] = stim_duration

    return bout_dict

def save_bout_to_csv(bout_dict, output_dir, file_prefix, bout_number):
    """
    Save bout data to CSV file.

    Parameters:
    - bout_dict: Dictionary containing bout data
    - output_dir: Directory to save CSV file
    - file_prefix: Prefix for output filename
    - bout_number: Number of current bout

    Returns:
    - Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    stim_duration_ms = int(bout_dict['Stim Duration (s)'] * 1000)
    output_file = os.path.join(
        output_dir,
        f"{file_prefix}_bout{bout_number}_stim{stim_duration_ms}ms.csv"
    )

    df = pd.DataFrame(bout_dict)
    df.to_csv(output_file, index=False)
    return output_file

#=============================================
# 3. VISUALIZATION FUNCTIONS
#=============================================

def set_plot_style():
    """
    Opt-in matplotlib defaults for editable-text SVGs (Affinity/Illustrator safe).
    Call once near the top of a notebook. Idempotent.
    """
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
