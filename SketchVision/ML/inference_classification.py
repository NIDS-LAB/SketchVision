import os
import re
import glob
import pickle
import struct
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score

# --- Configuration & Constants ---
GPU_INDEX = 0
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
K = "12"

# Paths
MODEL_PATH = "../models/CNN_classifier.joblib"
LABEL_MAP_PATH = "../models/label_map_ex3.pkl"
RESULT_DIR = f"./Result/{K}"
OUTPUT_PKL_DIR = "Resultdf/OneShot_FromNoise"
FIGURE_DIR = "./Figures"
# Root directory containing the flow count binaries (flCnt.bin, atk.bin, ben.bin)
BIN_MAP_ROOT = "../sketch_out/Clean_Stream" 

attack_category_mapping = { 
    "Botnet": ["thbot", "telnetpwdla", "telnetpwdmd"],
    "C2 Com.": ["bitcoinminer", "mazarbot", "ransombo", "wannalocker"], 
    "DDoS": ["ssdprdos", "cldaprdos", "riprdos", "charrdos"],
    "Data Exfiltration": ["dridex", "feiwo", "snojan", "penetho", "koler"],
    "Surveil": ["adload", "webcompanion", "mobidash"],    
}


# Indices for filtering
START_INDICES = START = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 
                         110, 115, 120, 125, 130, 135, 140, 145, 150]


# --- GPU Setup ---
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[GPU_INDEX], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[GPU_INDEX], True)
        print("Using GPU:", gpus[GPU_INDEX])
    except RuntimeError as e:
        print(e)

# --- Helper Functions ---

def read_bin_map(path):
    """Reads binary map file (key -> value)."""
    result = {}
    if not os.path.exists(path):
        print(f"Warning: Binary map not found at {path}")
        return result
        
    with open(path, 'rb') as f:
        size_data = f.read(8)
        if not size_data: return result
        map_size = struct.unpack('Q', size_data)[0]
        
        for _ in range(map_size):
            strsize_data = f.read(8)
            strsize = struct.unpack('Q', strsize_data)[0]
            key = f.read(strsize).decode('utf-8')
            value_data = f.read(4)
            value = struct.unpack('I', value_data)[0]
            result[key] = value
    return result

def get_id_cnt(ben_atk_filename, atk_name):
    """
    Merges flCnt.bin (key->cnt) with ID map (key->id) 
    to produce a map of {id: count}.
    """
    base_path = os.path.join(BIN_MAP_ROOT, atk_name)
    
    # 1. Read Count Map (key -> cnt)
    cnt_map_path = os.path.join(base_path, "flCnt.bin")
    a_map = read_bin_map(cnt_map_path)
    
    # 2. Read ID Map (key -> id) e.g., atk.bin or ben.bin
    id_map_path = os.path.join(base_path, f"{ben_atk_filename}.bin")
    b_map = read_bin_map(id_map_path)
    
    # 3. Merge: id -> cnt
    merged = {}
    for key in a_map:
        cnt = a_map[key]
        if key in b_map:
            id_val = b_map[key]
            merged[id_val] = cnt
            
    return merged

def preprocess_image_tf(path, label):
    """TensorFlow image loading and preprocessing pipeline."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img, label

def extract_metadata(path):
    """Extracts metadata (num, start_index, is_attack) from filename."""
    pattern = re.compile(r'(Atk|Ben)_(\d+)_(\d+)\.png')
    filename = os.path.basename(path)
    match = pattern.match(filename)
    if match:
        type_str, num, start = match.groups()
        is_attack = 1 if type_str == "Atk" else 0
        return int(num), int(start), is_attack
    return None

def select_col(row, probability_columns):
    """Sliding window logic for prediction smoothing."""
    win = 5
    values = row[probability_columns].values
    max_idx = len(values) - win
    
    # Sliding window from left to right
    for i in range(max_idx + 1):
        window = values[i:i+win]
        avg = window.mean()
        if avg >= 0.9:
            return pd.Series([avg, int(probability_columns[i].split('_')[0])])
    
    # Fallback: average of last window
    if len(values) >= win:
        last_window = values[-win:]
        fallback_avg = last_window.mean()
        fallback_start = int(probability_columns[-win].split('_')[0])
        return pd.Series([fallback_avg, fallback_start])
    else:
        # Fallback if sequence is shorter than window
        return pd.Series([values.mean() if len(values) > 0 else 0, 0])

# --- Plotting Functions ---

def plot_results(atk_ratio, atk_ratio_c=None):
    """Generates CDF plots for packet ratios."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # Styling
    sns.set_theme(style="ticks", context="paper", font="Arial")
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 1

    # Colors
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # --- Plot 1: CDF per type (Denoised) ---
    if atk_ratio:
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        color_idx = 0
        for type_name, values in atk_ratio.items():
            if len(values) == 0: continue
            
            values = np.clip(values, None, 1)
            sorted_vals = np.sort(values)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
            
            # Extend to x=1
            sorted_vals = np.append(sorted_vals, 1)
            cdf = np.append(cdf, 1)
            
            curr_color = colors[color_idx % len(colors)]
            ax1.step(sorted_vals, cdf, where='post', linewidth=3, label=type_name, color=curr_color)
            color_idx += 1

        ax1.set_xlim(0, 1.01)
        ax1.set_ylim(0.1, 1.01)
        ax1.set_xlabel('% of Packets', fontsize=28)
        ax1.set_ylabel('CDF (%)', fontsize=28)
        ax1.tick_params(axis='both', labelsize=28, length=0)
        ax1.legend(fontsize=10, loc='upper center', ncols=2, bbox_to_anchor=(0.8, 0.6))
        ax1.grid(True, linestyle=':', linewidth=1, alpha=0.7)
        
        plot_path = os.path.join(FIGURE_DIR, 'CDF_per_Type_atkSeen_oneshot.pdf')
        fig1.tight_layout()
        fig1.savefig(plot_path, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved Plot 1: {plot_path}")

    # --- Plot 2: Combined CDFs (Denoised vs Noise-free) ---
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    
    # Process Denoised
    all_values = np.concatenate([np.clip(v, None, 1) for v in atk_ratio.values()]) if atk_ratio else np.array([])
    if len(all_values) > 0:
        sorted_all = np.sort(all_values)
        cdf_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
        sorted_all = np.append(sorted_all, 1)
        cdf_all = np.append(cdf_all, 1)
        ax2.step(sorted_all, cdf_all, where='post', linewidth=3, color='red', label='Denoised')

    # Process Clean (if available)
    if atk_ratio_c is not None and len(atk_ratio_c) > 0:
        all_values_c = np.concatenate([np.clip(v, None, 1) for v in atk_ratio_c.values()])
        if len(all_values_c) > 0:
            sorted_all_c = np.sort(all_values_c)
            cdf_all_c = np.arange(1, len(sorted_all_c) + 1) / len(sorted_all_c)
            sorted_all_c = np.append(sorted_all_c, 1)
            cdf_all_c = np.append(cdf_all_c, 1)
            ax2.step(sorted_all_c, cdf_all_c, where='post', linewidth=3, color='blue', label='Noise-free')

    ax2.set_xlim(0, 1.01)
    ax2.set_ylim(0.1, 1.01)
    ax2.set_xlabel('% of Packets', fontsize=28)
    ax2.set_ylabel('CDF (%)', fontsize=28)
    ax2.tick_params(axis='both', labelsize=28, length=0)
    ax2.legend(fontsize=22, loc='lower right', handlelength=1, handletextpad=0.5)
    
    plot_path_2 = os.path.join(FIGURE_DIR, 'CDF_Combined_packetSeen_oneshot.pdf')
    fig2.tight_layout()
    fig2.savefig(plot_path_2, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved Plot 2: {plot_path_2}")


# --- Main Execution ---

def main():
    # 1. Load Resources
    print("Loading resources...")
    try:
        with open(LABEL_MAP_PATH, 'rb') as file:
            mp = pickle.load(file)
    except FileNotFoundError:
        print(f"Label map not found at {LABEL_MAP_PATH}")
        return

    model = joblib.load(MODEL_PATH)
    
    # Configure Classes
    allS = int(len(mp)/2)
    NUM_CLASSES = allS + 1
    BENIGN_CLASS_IDX = NUM_CLASSES - 1
    
    # Initialize trackers
    mp[19] = "Ben"
    y_atk = {'Ben': {}}
    
    # Initialize DataFrame
    columns = ["name", "AUC", "ACC", "Recall", "Precision", "F1 Score", "CM"]
    df_r = pd.DataFrame(columns=columns)
    
    # Storage for ratio plots
    atk_ratio = {} 

    # 2. Dataset Loop
    dirs = glob.glob(f"{RESULT_DIR}/*")
    ben_imgs_cache = None 
    ben_meta_cache = None # Also cache metadata for benign to persist counts

    for d in dirs:
        type_ = os.path.basename(d)
        if type_ not in mp:
            continue

        print(f"\nProcessing Dataset: {type_}")
        label_idx = mp[type_]
        if type_ not in y_atk: y_atk[type_] = {}
        atk_ratio[type_] = []

        # --- A. Load Packet Count Maps ---
        # Get mapping: flow_id -> total_packets (cnt)
        # Note: 'atk' maps to atk.bin, 'ben' maps to ben.bin
        try:
            cnt_map_atk = get_id_cnt("atk", type_) 
            cnt_map_ben = get_id_cnt("ben", type_)
        except Exception as e:
            print(f"  Error loading binary maps for {type_}: {e}")
            cnt_map_atk = {}
            cnt_map_ben = {}

        # --- B. Collect Images ---
        img_dir_atk = os.path.join(d, "Atk", "Pred_fromDenoise")
        img_dir_ben = os.path.join(d, "Ben", "Pred_fromDenoise")

        atk_imgs = glob.glob(os.path.join(img_dir_atk, "*.png"))
        ben_imgs = glob.glob(os.path.join(img_dir_ben, "*.png"))

        image_paths = []
        metadata = [] # Stores (num, start, is_atk, total_cnt)
        labels = []

        # Process Attack Images
        for p in atk_imgs:
            meta = extract_metadata(p) # returns (num, start, is_atk)
            if meta and meta[1] in START_INDICES:
                num_id = meta[0]
                # Lookup total packet count
                total_cnt = int(cnt_map_atk.get(num_id, 0))
                
                image_paths.append(p)
                metadata.append((num_id, meta[1], 1, total_cnt)) # Added total_cnt
                
                lbl = np.zeros(NUM_CLASSES)
                lbl[label_idx] = 1
                labels.append(lbl)

        for p in ben_imgs:
            meta = extract_metadata(p)
            if meta and meta[1] in START_INDICES:
                num_id = meta[0]
                total_cnt = int(cnt_map_ben.get(num_id, 0))
                
                image_paths.append(p)
                metadata.append((num_id, meta[1], 0, total_cnt))
                
                lbl = np.zeros(NUM_CLASSES)
                lbl[BENIGN_CLASS_IDX] = 1
                labels.append(lbl)

        if len(image_paths) == 0:
            print("  No valid images found.")
            continue

        print(f"  Total Samples: {len(image_paths)}")

        # 3. TensorFlow Inference
        path_tensor = tf.constant(image_paths)
        
        # meta_np shape: (N, 4) -> num, start, is_attack, cnt
        meta_np = np.array(metadata) 

        dataset = tf.data.Dataset.from_tensor_slices((path_tensor, tf.constant(labels)))
        dataset = dataset.map(preprocess_image_tf, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        preds = model.predict(dataset, verbose=1)

        # 4. Binary Probabilities Logic (One-Shot)
        y_true = np.argmax(np.array(labels), axis=1)
        y_pred_indices = np.argmax(preds, axis=1)
        binary_probs = np.zeros(len(preds))
        
        for i in range(len(preds)):
            
            ####### OR, when targeting range few attacks only in multi-class #######
            #intrest = np.array([mp["telnetpwdla"], mp["mazarbot"]])
            selected = preds[i][label_idx]
            p_target = np.sum(selected)
            p_benign = preds[i, BENIGN_CLASS_IDX]
            binary_probs[i] = p_target / (p_target + p_benign + 1e-9)

            ### OR:
            #binary_probs[i] = 1 - preds[i, BENIGN_CLASS_IDX]

        # 5. Grouping & Ratio Calculation
        grouped_preds = defaultdict(dict)
        grouped_cnts = {} # Store total count per flow

        # Zip includes 'cnt' from meta_np column 3
        for prob, (num, start, is_atk, cnt) in zip(binary_probs, meta_np):
            row_key = f"{num}_{is_atk}" 
            col_key = f"{start}"
            grouped_preds[row_key][col_key] = prob
            grouped_cnts[row_key] = cnt

        df_flow = pd.DataFrame.from_dict(grouped_preds, orient='index')
        
        # Add Count Column
        df_flow['cnt'] = df_flow.index.map(grouped_cnts)

        # Identify probability columns (start indices)
        prob_cols = [c for c in df_flow.columns if c.isdigit()]
        prob_cols = sorted(prob_cols, key=lambda x: int(x))

        # Sliding Window
        df_flow[['max_prob', 'selected_start']] = df_flow.apply(
            lambda row: select_col(row, prob_cols), axis=1
        )
        
        # Ground Truth
        df_flow['act_label'] = df_flow.index.map(lambda x: int(x.split('_')[1]))

        # --- RATIO CALCULATION ---
        # Ratio = selected_start / total_packets (clipped to 1.0)
        # Note: Avoid division by zero
        df_flow['cnt'] = df_flow['cnt'].replace(0, 1) 
        df_flow['ratio'] = (df_flow['selected_start'] / df_flow['cnt']).clip(upper=1.0)

        # Store ratios for attack flows only for plotting
        atk_flows = df_flow[df_flow['act_label'] == 1]
        atk_ratio[type_] = atk_flows['ratio'].tolist()

        # 6. Metrics
        act = df_flow['act_label'].values
        y_pred_final = (df_flow['max_prob'] >= 0.9).astype(int).values
        y_scores = df_flow['max_prob'].values

        try:
            auc = roc_auc_score(act, y_scores)
        except ValueError:
            auc = 0.0

        row_metrics = {
            "name": type_,
            "AUC": auc,
            "ACC": accuracy_score(act, y_pred_final),
            "Recall": recall_score(act, y_pred_final, zero_division=0),
            "Precision": precision_score(act, y_pred_final, zero_division=0),
            "F1 Score": f1_score(act, y_pred_final, zero_division=0),
            "CM": confusion_matrix(act, y_pred_final)
        }
        
        row_df = pd.DataFrame([row_metrics])
        df_r = pd.concat([df_r, row_df], ignore_index=True)
        print(row_df[["name", "ACC", "Recall", "F1 Score"]])

    # 7. Save Results
    os.makedirs(OUTPUT_PKL_DIR, exist_ok=True)
    df_r.to_pickle(f"{OUTPUT_PKL_DIR}/multi_{K}.pkl")
    with open(f"{OUTPUT_PKL_DIR}/multi_{K}.txt", 'w') as f:
        f.write(df_r.to_string(index=False))
    print(f"\nResults saved to {OUTPUT_PKL_DIR}/multi_{K}.pkl")

    # 8. Plotting
    # Check if atk_ratio_c exists (from clean data), otherwise None
    atk_ratio_c = locals().get('atk_ratio_c', None)
    
    if atk_ratio:
        print("\nGenerating Plots...")
        plot_results(atk_ratio, atk_ratio_c)

if __name__ == "__main__":
    main()