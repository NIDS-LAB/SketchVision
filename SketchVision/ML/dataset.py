import os
import random
import bisect
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def init_datasets(ROOT_DIR, args):
    # Default to 'denoise' if not specified in args
    mode = args.get('dataset_mode', 'denoise') 
    
    clean_path_train = f'{ROOT_DIR}/Dataset_mixSK/train/clean'
    adv_path_train = f'{ROOT_DIR}/Dataset_mixSK/train/adv'
    
    clean_path_test = f'{ROOT_DIR}/Dataset_mixSK/test/clean'
    adv_path_test = f'{ROOT_DIR}/Dataset_mixSK/test/adv'

    training_dataset = AttackImageDataset(clean_path_train, adv_path_train, mode=mode)
    testing_dataset = AttackImageDataset(clean_path_test, adv_path_test, mode=mode)
    
    return training_dataset, testing_dataset

def init_dataset_loader(dataset, args, shuffle=True):
    dataset_loader = cycle(
        DataLoader(
            dataset,
            batch_size=args['Batch_Size'], 
            shuffle=shuffle,
            num_workers=0, 
            drop_last=True
        )
    )
    return dataset_loader

class AttackImageDataset(Dataset):
    def __init__(self, clean_dir, adv_dir, mode='denoise', transform=None, image_size=(64, 64)):
        self.clean_dir = clean_dir
        self.adv_dir = adv_dir
        self.mode = mode
        
        # Define transformations
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.pairs = []
        self.video_frames_cache = {} 
        
        # Range logic for denoise-foreacsting and early detection. 
        self.forecast_range = [10, 11, 15, 16, 20, 21, 25, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
                                       110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200,
                                       205, 210] ## Increase as needed

        clean_files = sorted([f for f in os.listdir(clean_dir) if f.endswith('.mkv')])
        
        print(f"Initializing Dataset in [{self.mode.upper()}] mode...")

        for clean_file in tqdm(clean_files):
            base_name = os.path.splitext(clean_file)[0]
            video_file = base_name + ".mkv"
            
            if self.mode == 'denoise':
                # In denoise mode, we need the paired adversarial video
                target_video_path = os.path.join(adv_dir, video_file)
            else:
                # In forecast mode, we use the clean video itself for temporal prediction
                target_video_path = os.path.join(clean_dir, video_file)

            if os.path.exists(target_video_path):
                self.pairs.append((clean_file, video_file, target_video_path))
                # Cache the video frames from the target path
                self.video_frames_cache[target_video_path] = self._load_video_frames(target_video_path)
            else:
                print(f"[Warning] Video not found: {target_video_path}")

    def __len__(self):
        return len(self.pairs)

    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        
        while cap.isOpened():
            if count == 1100:
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

    def _load_single_frame(self, video_path, frame_idx):
        """Fetch a specific frame from disk on demand."""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        return self.transform(pil_img)

    def _find_max_smaller_than(self, sorted_list, X):
        index = bisect.bisect_left(sorted_list, X)
        if index == 0:
            return None
        return index - 1

    def __getitem__(self, idx):
        clean_file, video_file, cached_video_path = self.pairs[idx]
        clean_path_full = os.path.join(self.clean_dir, clean_file)
        
        # Retrieve cached frames
        cached_frames = self.video_frames_cache[cached_video_path]
        total_frames = len(cached_frames)
        last_frame_idx = total_frames - 1

        selected_idx = 0
        clean_tensor = None
        adv_tensor = None

        if self.mode == 'denoise':

            max_range_idx = self._find_max_smaller_than(self.forecast_range, total_frames)
            selected_range_idx = random.randint(0, max_range_idx)
            pkt_num = self.forecast_range[selected_range_idx]      # e.g. 20
            frame_idx = pkt_num - 1 # 0-based
           
            adv_tensor = cached_frames[frame_idx]
            clean_tensor = self._load_single_frame(clean_path_full, frame_idx)
            
        elif self.mode == 'forecast':

            max_range_idx = self._find_max_smaller_than(self.forecast_range, total_frames)
            selected_range_idx = random.randint(0, max_range_idx)
            pkt_num = self.forecast_range[selected_range_idx]      # e.g. 20
            frame_idx = pkt_num - 1                  # 0-based
            
            adv_tensor = cached_frames[frame_idx]
            clean_tensor = cached_frames[-1]

        # Extract Label from filename
        name = clean_file
        try:
            label = int(name.split('_')[1])
        except (IndexError, ValueError):
            label = 0 

        sample = {
            'clean': clean_tensor, 
            'adv': adv_tensor, 
            "filenames": clean_path_full, 
            "label": label, 
            "pNum": pkt_num
        }
        
        return sample

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing DENOISE mode:")
    args_denoise = {"Batch_Size": 4, "dataset_mode": "denoise"}
    # Ensure directories exist before running this block in production
    try:
        d_set, _ = init_datasets("./", args_denoise)
        loader = init_dataset_loader(d_set, args_denoise)
        new = next(loader)
        print("Batch loaded successfully. Shape:", new['clean'].shape)
    except Exception as e:
        print("Skipping execution (check directory paths):", e)