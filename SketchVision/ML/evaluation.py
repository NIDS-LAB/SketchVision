import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import auc, roc_curve

from helpers import gridify_output

def testing(testing_dataset_loader, diffusion, args, ema, model):
    """
    Samples videos on test set & calculates metrics (MSE).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup Output Directories
    save_dir = f'./diffusion-videos/ARGS={args["arg_num"]}/test-set/'
    os.makedirs(save_dir, exist_ok=True)

    ema.eval()
    model.eval()

    plt.rcParams['figure.dpi'] = 200

    # 1. Visual Evaluation (Video Sampling)
    # Generate samples at intervals defined by 'sample_distance'
    start_t = 100
    step_t = 100
    # Adjust for small T values
    if args['sample_distance'] < 100:
        start_t = 10
        step_t = 10
        
    for i in range(start_t, args['sample_distance'] // 4, step_t):
        try:
            data = next(testing_dataset_loader)
        except StopIteration:
            break
            
        # Unified Dictionary Handling (from dataset.py)
        x = data["adv"].to(device)
        pNum = data["pNum"].to(device)
        row_size = min(5, args['Batch_Size'])
        # Sampling
        # Assumes diffusion.forward_backward returns (sequence_of_frames, ...)
        out, _ = diffusion.forward_backward(ema, x, see_whole_sequence="half", label=None, t_distance=i, pnum=pNum, guidance_kwargs=None)

        if (args.get('save_vids', False)):
            # Create Animation
            fig, ax = plt.subplots()
            imgs = [[ax.imshow(gridify_output(frame, row_size), animated=True)] for frame in out]
            ani = animation.ArtistAnimation(fig, imgs, interval=200, blit=True, repeat_delay=1000)
            
            # Save with unique counter
            existing_files = len(os.listdir(save_dir))
            ani.save(f'{save_dir}/t={i}-sample={existing_files}.mp4')
            plt.close(fig)

    # 2. Quantitative Evaluation (MSE)
    print("Running Quantitative Evaluation...")
    
    test_iters = 10  # Number of batches to evaluate
    vlb_stats = []
    
    for _ in range(test_iters):
        try:
            data = next(testing_dataset_loader)
        except StopIteration:
            break

        clean = data["clean"].to(device)
        adv = data["adv"].to(device)
        pNum = data["pNum"].to(device) if "pNum" in data else None
        
        # Calculate VLB (Variational Lower Bound)
        vlb_terms = diffusion.calc_total_vlb(clean, adv, model, args, None, pNum)
        vlb_stats.append(vlb_terms)

        # We reconstruct from T/2 (common heuristic for denoising/restoration)
        t_dist = args["T"] // 2
        out_recon, _ = diffusion.forward_backward(ema, x, see_whole_sequence="half", label=None, t_distance=t_dist, pnum=pNum, guidance_kwargs=None)
        
        # If forward_backward returns a sequence, take the final denoised frame
        if isinstance(out_recon, list):
             final_img = out_recon[-1]
        elif isinstance(out_recon, torch.Tensor) and out_recon.ndim == 5:
             final_img = out_recon[-1]
        else:
             final_img = out_recon

    # 3. Report Results
    def get_stat_mean_std(key, index=None):
        """Helper to extract mean/std from list of dicts"""
        values = []
        for item in vlb_stats:
            val = item[key]
            if index is not None:
                # Ensure index is within bounds of T
                idx = min(index, val.shape[1]-1)
                val = val[0][idx] 
            values.append(val.mean().item())
        return np.mean(values), np.std(values)

    print("\n=== Test Results ===")

    # Inspect specific timestep (e.g. t=200 or max T)
    check_t = min(199, args['T'] - 1)
    
    m, s = get_stat_mean_std('x_0_mse', check_t)
    print(f"x_0 MSE @ t={check_t}: {m:.4f} ± {s:.4f}")

    m, s = get_stat_mean_std('mse', check_t)
    print(f"MSE @ t={check_t}: {m:.4f} ± {s:.4f}")

if __name__ == '__main__':
    print("This module is designed to be called by train.py.")
