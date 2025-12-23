import os
import glob
import time
import gc
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Custom Modules (Assumed to be in the same directory)
from GaussianDiffusionClAdvStream import GaussianDiffusionModel, get_beta_schedule, space_timesteps
from UNetClStream import UNetModel
from helpers import gridify_output

# --- Configuration & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
ARG_NUM = 99
T = 2000
K = 12 ## Memory
DDIM = False
DDIM_STEP = "ddim24"
SCHED = "cosine"

# Paths
MODEL_PATH = f'../models/params-{ARG_NUM}-Denoise.pt'
SRC_ROOT = "../sketch_out/Clean_Stream/"
DST_ROOT = f"./Result/{K}"
F_DIR = "Denoise_img_for_Pred"

# Indices to extract from video
SELECTED_INDICES = [10, 11, 15, 16, 20, 21, 25, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]

# --- Helper Functions ---

def mirror_first_level_dirs(src_root, dst_root):
    """Creates a matching folder structure in destination."""
    os.makedirs(dst_root, exist_ok=True)
    for item in os.listdir(src_root):
        src_path = os.path.join(src_root, item)
        if os.path.isdir(src_path):
            dst_path = os.path.join(dst_root, item)
            os.makedirs(dst_path, exist_ok=True)
            print(f"Created/Verified: {dst_path}")

def get_video_writers(fnames, fps=30):
    """Initializes VideoWriters for a batch of files."""
    video_writers = []
    H, W = IMG_SIZE
    # Use FFV1 codec (lossless) for MKV format
    fourcc = cv2.VideoWriter_fourcc('F', 'F', 'V', '1')
    
    for i in range(len(fnames)):
        path = f"{fnames[i]}.mkv"
        writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
        video_writers.append(writer)
    return video_writers

# --- Dataset Class ---

class AttackImageDataset(Dataset):
    def __init__(self, clean_dir, adv_dir, transform=None, image_size=IMG_SIZE):
        self.clean_dir = clean_dir
        self.adv_dir = adv_dir
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.pairs = []
        self.video_frames_cache = {} 
        
        if not os.path.exists(clean_dir):
            print(f"[Error] Clean directory not found: {clean_dir}")
            return

        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.mkv')])
        
        print(f"Loading dataset from: {clean_dir}")
        #xi = 0
        for clean_file in tqdm(clean_files):
         #   if(xi == 10): break
            base_name = os.path.splitext(clean_file)[0]
            video_file = base_name + ".mkv"
            video_path = os.path.join(adv_dir, video_file)
          #  xi += 1
            if os.path.exists(video_path):
                self.pairs.append((clean_file, video_file, video_path))
                # Cache video frames upfront
                self.video_frames_cache[video_path] = self._load_video_frames(video_path)
            else:
                # Optional: Uncomment to debug missing files
                print(f"[Warning] Video not found: {video_path}")
                pass

    def __len__(self):
        return len(self.pairs)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        
        while cap.isOpened():
            if count >= 250: # Limit max frames loaded
                break
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            transformed_frame = self.transform(pil_img)
            frames.append(transformed_frame)
            count += 1
            
        cap.release()
        return frames

    def __getitem__(self, idx):
        clean_file, video_file, video_path = self.pairs[idx]
        clean_path = os.path.join(self.clean_dir, clean_file)
        
        # Load clean image
        #clean_image = Image.open(clean_path).convert("RGB")
        #clean_tensor = self.transform(clean_image)

        # Retrieve cached frames
        adv_frames = self.video_frames_cache[video_path]
        last_frame_idx = len(adv_frames) - 1
        
        # Determine valid upper bound index for fallback
        valid_upper_bound_idx = max([ix for ix in SELECTED_INDICES if ix <= last_frame_idx], default=0)
        
        predefined_frames = []
        indices = []
        
        for ssidx in SELECTED_INDICES:
            if ssidx <= last_frame_idx:
                predefined_frames.append(adv_frames[ssidx-1])
                indices.append(ssidx)
            else:
                predefined_frames.append(adv_frames[valid_upper_bound_idx-1])
                indices.append(valid_upper_bound_idx)
        
        sample = {
            'clean': clean_path, 
            "filenames": clean_path, 
            "pNum": torch.tensor(indices), 
            "pdef": torch.stack(predefined_frames)
        }
        return sample

# --- Model Loading ---

def load_diffusion_model():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    try:
        loaded_model = torch.load(MODEL_PATH, map_location=DEVICE)
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return None, None

    # Setup Timestep Map
    betas = get_beta_schedule(T, SCHED)
    timestep_map = list(range(T))

    if DDIM:
        use_timesteps = space_timesteps(int(T // 4), DDIM_STEP)
        timestep_map = []
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod_ in enumerate(alphas_cumprod):
            if i in use_timesteps:
                new_betas.append(1 - alpha_cumprod_ / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod_
                timestep_map.append(i)
        betas = np.array(new_betas)
        print(f"DDIM Timestep Map: {timestep_map}")

    # Initialize Diffusion
    diffusion = GaussianDiffusionModel(
        IMG_SIZE, betas, loss_weight="none", loss_type="l2", 
        noise="gauss", img_channels=3, timestep_map=np.array(timestep_map), num_heads=4
    )

    # Initialize UNet
    model = UNetModel(
        IMG_SIZE[0], 64, channel_mults="", dropout=0, 
        n_heads=4, n_head_channels=-1, in_channels=3, num_classes=20
    )

    # Load Weights
    if "unet" in loaded_model:
        model.load_state_dict(loaded_model["unet"])
    elif "ema" in loaded_model:
        model.load_state_dict(loaded_model["ema"])
    
    model.to(DEVICE)
    model.eval()
    
    return diffusion, model

# --- Main Execution ---

def main():
    # 1. Setup Directories
    mirror_first_level_dirs(SRC_ROOT, DST_ROOT)
    
    # 2. Load Model
    diffusion, model = load_diffusion_model()
    if model is None: 
        return

    # 3. Get list of sub-datasets
    atk_list = glob.glob(f"{DST_ROOT}/*")
    
    # 4. Processing Loop
    current_batch_size = BATCH_SIZE
    
    for img_dir in atk_list:
        # Skip non-directory files
        if any(x in img_dir for x in ["joblib", "pkl", "pth"]):
            continue

        fname = os.path.basename(img_dir)
        print(f"Processing: {fname}")
        
        # Define Input Paths (Adjust these based on exact folder structure)
        video_paths = f"../sketch_out/Sketch_Stream_{K}/{fname}" 
        clean_img_paths = f"../sketch_out/Clean_Stream/{fname}"
        
        # Initialize Dataset & Loader
        dataset = AttackImageDataset(clean_img_paths, video_paths)
        if len(dataset) == 0:
            print(f"Skipping {fname} (No data found)")
            continue

        data_loader = DataLoader(dataset, batch_size=current_batch_size, shuffle=False, num_workers=0, persistent_workers=False)
        
        start_time = time.time()
        print(f"Denoising: {fname}")
        with torch.no_grad():
            for data in tqdm(data_loader):
                
                # Prepare Output Paths
                clean_sample_paths = []
                for filename in data["filenames"]:
                    # Create directory structure for predictions
                    base_dir = os.path.dirname(filename) # Roughly getting root
                    
                    # Ensure output directories exist
                    save_atk_dir = os.path.join(img_dir, "Atk", F_DIR)
                    save_ben_dir = os.path.join(img_dir, "Ben", F_DIR)
                    os.makedirs(save_atk_dir, exist_ok=True)
                    os.makedirs(save_ben_dir, exist_ok=True)

                    ffname = os.path.basename(filename)
                    if "Atk" in ffname:
                        target_path = os.path.join(save_atk_dir, ffname)
                    else:
                        target_path = os.path.join(save_ben_dir, ffname)
                    
                    # Strip extension for MKV writing later
                    clean_sample_paths.append(os.path.splitext(target_path)[0])

                # Prepare Inputs
                pdef_imgs_tensor = data["pdef"] # [Batch, SeqLen, C, H, W]
                pnum_tensor = data["pNum"]      # [Batch, SeqLen]
                
                # Initialize Video Writers
                writers = get_video_writers(clean_sample_paths)

                # Iterate over the sequence length (frames selected)
                seq_len = len(SELECTED_INDICES)
                for seq_idx in range(seq_len):
                    xx = pdef_imgs_tensor[:, seq_idx].to(DEVICE)
                    pNum = pnum_tensor[:, seq_idx].to(DEVICE)

                    out, msk_out = diffusion.forward_backward(
                         model, xx, see_whole_sequence="half", t_distance=10, 
                         label=None, pnum=pNum, guidance_kwargs=None
                     )
                    
                    imG = out[-1].detach().cpu() 
                    # Write frames
                    batch_size_used = imG.shape[0]
                    for b_idx in range(batch_size_used):
                        img = gridify_output(imG[b_idx], 1, -1).numpy()
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        writers[b_idx].write(img_bgr)

                # Close writers
                for wr in writers:
                    wr.release()

        print(f"  Finished {fname} in {time.time() - start_time:.2f}s")
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
