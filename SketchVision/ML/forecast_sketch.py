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

# Custom Modules
from GaussianDiffusionClAdvStream import GaussianDiffusionModel, get_beta_schedule, space_timesteps
from UNetClStream import UNetModel  # Note: Using 5Sam version as per imports
from helpers import gridify_output

# --- Configuration & Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
ARG_NUM = 100
T = 2000
K = 12
DDIM = False
DDIM_STEP = "ddim24"
SCHED = "cosine"

# Paths
MODEL_PATH = f'../models/params-{ARG_NUM}-Forecast.pt'
SRC_ROOT = "../sketch_out/Clean_Stream/"
DST_ROOT = f"./Result/{K}"

# Frame Indices (Forecast targets)
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

# --- Dataset Class ---

class AttackImageDataset(Dataset):
    def __init__(self, clean_dir, root_adv_dir, transform=None, image_size=IMG_SIZE):
        """
        clean_dir: Directory containing clean references.
        root_adv_dir: Directory containing 'Atk' and 'Ben' subdirectories with videos (MKVs).
        """
        self.clean_dir = clean_dir
        self.root_adv_dir = root_adv_dir
        
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.pairs = []            # List of (clean_full_path, video_full_path, sub_type)
        self.video_frames_cache = {} 
        
        # Scan both subdirectories
        sub_types = ['Atk', 'Ben']
        
        print(f"Scanning {root_adv_dir} for videos...")
        
        for sub_type in sub_types:
            sub_dir_path = os.path.join(root_adv_dir, sub_type, "Denoise_img_for_Pred")
            if not os.path.exists(sub_dir_path):
                continue
                
            # Find all MKV files in the subdirectory
            video_files = sorted([f for f in os.listdir(sub_dir_path) if f.endswith('.mkv')])
            
            for video_file in tqdm(video_files, desc=f"Loading {sub_type}"):
                base_name = os.path.splitext(video_file)[0]
                
                clean_file_name = base_name + ".mkv"
                clean_full_path = os.path.join(clean_dir, clean_file_name)
                video_full_path = os.path.join(sub_dir_path, video_file)

                # Verify clean reference exists
                if os.path.exists(clean_full_path):
                    self.pairs.append((clean_full_path, video_full_path, sub_type))
                    # Cache frames immediately
                    self.video_frames_cache[video_full_path] = self._load_video_frames(video_full_path)
                else:
                    print(f"[Warning] Video not found: {clean_full_path}")
                    # Optional: Print warning only once per missing file type to avoid spam
                    pass

    def __len__(self):
        return len(self.pairs)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        
        while cap.isOpened():
            if count == 300: # Limit max frames
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
        clean_path, video_path, sub_type = self.pairs[idx]

        # Load clean image
        #clean_image = Image.open(clean_path).convert("RGB")
        #clean_tensor = self.transform(clean_image)

        # Retrieve cached frames
        adv_frames = self.video_frames_cache[video_path]
        
        # Select specific forecast frames
        # Safety check: ensure the requested index exists in the video
        selected_frames = []
        for i in range(len(SELECTED_INDICES)):
            # FRAME_MAP maps frame number to a smaller index, but we need the actual frame from the list
            # Logic from original code: directly accessing adv_frames
            if i < len(adv_frames):
                selected_frames.append(adv_frames[i])
            else:
                # Fallback: repeat last frame if video is shorter than expected
                selected_frames.append(adv_frames[-1])

        # Pack data
        # "pdef" is stack of frames we want to predict/use
        # "pNum" is the tensor of frame indices
        sample = {
            "filenames": clean_path,  # Use clean path for naming logic later
            "pNum": torch.tensor(SELECTED_INDICES), 
            "pdef": torch.stack(selected_frames),
            "subtype": sub_type # Helper to know if it came from Atk or Ben
        }
        return sample

# --- Model Loading ---

def load_model():
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...")
    loaded_model = torch.load(MODEL_PATH, map_location=DEVICE)

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

    if "unet" in loaded_model:
        model.load_state_dict(loaded_model["unet"])
    else:
        model.load_state_dict(loaded_model["ema"])
    
    model.to(DEVICE)
    model.eval()
    
    return diffusion, model

# --- Main Execution ---

def main():
   
    diffusion, model = load_model()
    F_DIR = "Pred_fromDenoise"
    # Ensure directories exist
    mirror_first_level_dirs(SRC_ROOT, DST_ROOT)
    
    # Get list of datasets (subfolders in Result/K/)
    atk_list = glob.glob(f"{DST_ROOT}/*")
    
    for img_dir in atk_list:
        # Skip non-directories or system files
        if any(x in img_dir for x in ["joblib", "pkl", "pth"]) or not os.path.isdir(img_dir):
            continue

        fname = os.path.basename(img_dir)
        print(f"Processing: {fname} (K={K})")
        
        # Paths
        video_root_path = f"{DST_ROOT}/{fname}/"  # Expected to contain Atk/ and Ben/
        clean_img_path = f"../sketch_out/Clean_Stream/{fname}"
        
        if not os.path.exists(video_root_path):
            print(f"Skipping {fname} - Video path not found: {video_root_path}")
            continue

        # Initialize Dataset
        # The dataset will scan video_root_path/Atk and video_root_path/Ben
        dataset = AttackImageDataset(clean_img_path, video_root_path)
        
        if len(dataset) == 0:
            print(f"No valid data found for {fname}")
            continue

        # Benign typically uses larger batch size in orig code, but unified here for simplicity. 
        # Adjust if memory is an issue.
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        start_time = time.time()
        print("Forecasting ...")
        with torch.no_grad():
            for data in tqdm(data_loader):
                pdef_imgs = data["pdef"] # [Batch, SeqLen, C, H, W]
                pnums = data["pNum"]     # [Batch, SeqLen]
                subtypes = data["subtype"] # List of 'Atk' or 'Ben'
                clean_paths = data["filenames"]
                
                # Iterate over the sequence of selected frames
                seq_len = pdef_imgs.shape[1]
                
                for seq_idx in range(seq_len):
                    # Prepare input for this specific frame index across the batch
                    x = pdef_imgs[:, seq_idx].to(DEVICE)
                    p_num = pnums[:, seq_idx].to(DEVICE)

                    # Inference
                    # forward_backward returns a list of frames, we take the last one (fully denoised)
                    out, _ = diffusion.forward_backward(
                        model, x, see_whole_sequence="half", label=None, 
                        t_distance=1, pnum=p_num, guidance_kwargs=None
                    )
                    
                    # Process Output
                    imG_batch = out[-1].detach().cpu()
                    
                    # Save Images
                    batch_size_used = imG_batch.shape[0]
                    for b in range(batch_size_used):
                        # Determine output path based on subtype (Atk/Ben)
                        # img_dir is "./Result/K/fname"
                        subtype = subtypes[b] # 'Atk' or 'Ben'
                        save_dir = os.path.join(img_dir, subtype, F_DIR)
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Construct Filename
                        # Original: basename_frameNum.ext
                        clean_name = os.path.basename(clean_paths[b])
                        basename, ext = os.path.splitext(clean_name)
                        frame_num = SELECTED_INDICES[seq_idx]
                        new_name = f"{basename}_{frame_num}.png"
                        
                        save_path = os.path.join(save_dir, new_name)
                        
                        # Convert and Save
                        img_np = gridify_output(imG_batch[b], 1, -1).numpy()
                        img_pil = Image.fromarray(img_np)
                        img_pil.save(save_path)

        print(f"  Finished {fname} in {time.time() - start_time:.2f}s")
        
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    main()
