import os
import sys
import time
import json
import copy
import collections
import numpy as np
import torch
from torch import optim
from random import seed, random
import matplotlib.pyplot as plt
from matplotlib import animation

# Custom Modules
import dataset 
import evaluation 
from GaussianDiffusionClAdvStream import GaussianDiffusionModel, get_beta_schedule
from UNetClStream import UNetModel, update_ema_params
from helpers import gridify_output, defaultdict_from_json

torch.cuda.empty_cache()

ROOT_DIR = "./"

def train(training_dataset_loader, testing_dataset_loader, args, resume):
    in_channels = 3
    if args["channels"] != "":
        in_channels = args["channels"]

    # Initialize Model
    model = UNetModel(
        args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'], 
        dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
        in_channels=in_channels, num_classes=args['num_classes']
    )

    betas = get_beta_schedule(args['T'], args['beta_schedule'])

    diffusion = GaussianDiffusionModel(
        args['img_size'], betas, loss_weight=args['loss_weight'],
        loss_type=args['loss-type'], noise=args["noise_fn"], img_channels=in_channels
    )

    # Resume Logic
    if resume:
        if "unet" in resume:
            model.load_state_dict(resume["unet"])
        else:
            model.load_state_dict(resume["ema"])

        ema = UNetModel(
            args['img_size'][0], args['base_channels'], channel_mults=args['channel_mults'],
            dropout=args["dropout"], n_heads=args["num_heads"], n_head_channels=args["num_head_channels"],
            in_channels=in_channels, num_classes=args['num_classes']
        )
        ema.load_state_dict(resume["ema"])
        start_epoch = resume['n_epoch']
    else:
        start_epoch = 0
        ema = copy.deepcopy(model)

    tqdm_epoch = range(start_epoch, args['EPOCHS'] + 1)
    model.to(device)
    ema.to(device)
    
    optimiser = optim.AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'], betas=(0.9, 0.999))
    if resume:
        optimiser.load_state_dict(resume["optimizer_state_dict"])

    del resume
    start_time = time.time()
    vlb = collections.deque([], maxlen=10)
    iters = range(100) # Assuming fixed iterations per epoch based on original code

    print(f"Training Starts on {device}...!!")

    for epoch in tqdm_epoch:
        mean_loss = []

        for i in iters:
            data = next(training_dataset_loader)
            x = data["clean"].to(device)
            xx = data["adv"].to(device)
            label = data["label"].to(device)            
            pNum = data["pNum"].to(device)
            
            # Unconditional training probability
            if random() < 0.25:
                label = None

            loss, estimates = diffusion.p_loss(model, x, xx, args, label, pNum)
        
            noisy, est = estimates[1], estimates[2]
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimiser.step()

            update_ema_params(ema, model)
            mean_loss.append(loss.data.cpu())

            # Save visual progress
            if epoch % 50 == 0 and i == 0:
                row_size = min(8, args['Batch_Size'])
                training_outputs(
                    diffusion, x, est, noisy, epoch, row_size, 
                    save_imgs=args['save_imgs'],
                    save_vids=args['save_vids'], 
                    ema=ema, args=args
                )
        # Periodic VLB Logging
        if epoch % 200 == 0:
            time_taken = time.time() - start_time
            remaining_epochs = args['EPOCHS'] - epoch
            time_per_epoch = time_taken / (epoch + 1 - start_epoch)
            hours = remaining_epochs * time_per_epoch / 3600
            mins = (hours % 1) * 60
            hours = int(hours)
            
            current_mse = np.mean(mean_loss)
            
            print(
                f"epoch: {epoch}, "
                f"Mean MSE (Loss): {current_mse:.6f}, "
                f"time elapsed: {int(time_taken / 3600)}:{((time_taken / 3600) % 1) * 60:02.0f}, "
                f"remaining: {hours}:{mins:02.0f}\r"
            )

        # Checkpoint Saving
        if epoch % 1000 == 0 and epoch >= 0:
            save(unet=model, args=args, optimiser=optimiser, final=False, ema=ema, epoch=epoch)

    # Final Save and Eval
    save(unet=model, args=args, optimiser=optimiser, final=True, ema=ema)
    evaluation.testing(testing_dataset_loader, diffusion, ema=ema, args=args, model=model)


def save(final, unet, optimiser, args, ema, loss=0, epoch=0):
    """Save model checkpoint or final weights."""
    base_path = f'{ROOT_DIR}model/diff-params-ARGS={args["arg_num"]}'
    
    state = {
        'model_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
        'ema': ema.state_dict(),
        'args': args
    }

    if final:
        state['n_epoch'] = args["EPOCHS"]
        torch.save(state, f'{base_path}/params-final.pt')
    else:
        state['n_epoch'] = epoch
        state['loss'] = loss
        torch.save(state, f'{base_path}/checkpoint/diff_epoch={epoch}.pt')


def training_outputs(diffusion, x, est, noisy, epoch, row_size, ema, args, save_imgs=False, save_vids=False):
    """Generates and saves validation images/videos during training."""

    if save_imgs:
        if epoch % 100 == 0:
            # Full sampling visualization
            noise = torch.rand_like(x)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=x.device)
            x_t = diffusion.sample_q(x, t, noise)
            temp = diffusion.sample_p(ema, x_t, t)
            
            out = torch.cat(
                (x[:row_size, ...].cpu(), temp["sample"][:row_size, ...].cpu(),
                 temp["pred_x_0"][:row_size, ...].cpu())
            )
            plt.title(f'real, sample, prediction x_0 - epoch {epoch}')
        else:
            # One-step prediction visualization
            out = torch.cat(
                (x[:row_size, ...].cpu(), noisy[:row_size, ...].cpu(), est[:row_size, ...].cpu(),
                 (est - noisy).square().cpu()[:row_size, ...])
            )
            plt.title(f'real, noisy, prediction, mse - epoch {epoch}')

        plt.rcParams['figure.dpi'] = 150
        plt.grid(False)
        plt.imshow(gridify_output(out, row_size), cmap='gray')
        plt.savefig(f'./diffusion-training-images/ARGS={args["arg_num"]}/EPOCH={epoch}.png')
        plt.clf()

    if save_vids and epoch % 500 == 0:
        fig, ax = plt.subplots()
        plt.rcParams['figure.dpi'] = 200
        
        # Reduced sampling distance for faster video generation on frequent epochs
        dist = args['sample_distance'] // 2 if epoch % 1000 == 0 else args['sample_distance']
        
        out = diffusion.forward_backward(ema, x, "half", dist, denoise_fn="noise_fn")
        imgs = [[ax.imshow(gridify_output(x, row_size), animated=True)] for x in out]
        
        ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True, repeat_delay=1000)
        ani.save(f'{ROOT_DIR}diffusion-videos/ARGS={args["arg_num"]}/sample-EPOCH={epoch}.mp4')
    
    plt.close('all')


def main():

    if len(sys.argv) < 2:
        raise ValueError("Missing file argument")

    files = sys.argv[1:]
    resume = 0

    # Parse Resume Flags
    if files[0] == "RESUME_RECENT":
        resume = 1
        files = files[1:]
    elif files[0] == "RESUME_FINAL":
        resume = 2
        files = files[1:]

    if not files:
        raise ValueError("Missing config file argument")

    # Parse Config File Name
    file = files[0]
    if file.isnumeric():
        file = f"args{file}.json"
    elif not file.endswith(".json"):
        # Handle cases like "args25" -> "args25.json"
        if file.startswith("args"):
             file = f"{file}.json"
        else:
             file = f"args{file}.json"

    # Load Config
    with open(f'{ROOT_DIR}/../configuration/{file}', 'r') as f:
        args = json.load(f)
    
    args['arg_num'] = file.replace("args", "").replace(".json", "")
    args = defaultdict_from_json(args)
    
    # Make Arg-Specific Directories
    os.makedirs("model", exist_ok=True)
    base_model_dir = f'./model/diff-params-ARGS={args["arg_num"]}'
    os.makedirs(f'{base_model_dir}/checkpoint', exist_ok=True)
    if(args['save_vids']):
        os.makedirs(f'./diffusion-videos/ARGS={args["arg_num"]}', exist_ok=True)
    if(args['save_imgs']):
        os.makedirs(f'./diffusion-training-images/ARGS={args["arg_num"]}', exist_ok=True)

    print(f"Config: {file}, Args: {args}")

    # Initialize Datasets (Unified)
    training_dataset, testing_dataset = dataset.init_datasets(ROOT_DIR, args)
    training_dataset_loader = dataset.init_dataset_loader(training_dataset, args)
    testing_dataset_loader = dataset.init_dataset_loader(testing_dataset, args)

    # Load Weights if Resuming
    loaded_model = {}
    if resume:
        if resume == 1:
            checkpoints = sorted(os.listdir(f'{base_model_dir}/checkpoint'), reverse=True)
            for i in checkpoints:
                try:
                    file_dir = f"{base_model_dir}/checkpoint/{i}"
                    loaded_model = torch.load(file_dir, map_location=device)
                    print(f"Resuming from checkpoint: {i}")
                    break
                except RuntimeError:
                    continue
        else:
            file_dir = f'{base_model_dir}/params-final.pt'
            loaded_model = torch.load(file_dir, map_location=device)
            print("Resuming from final params.")

    # Start Training
    train(training_dataset_loader, testing_dataset_loader, args, loaded_model)

    # Cleanup Checkpoints
    checkpoint_dir = f'{base_model_dir}/checkpoint'
    for file_remove in os.listdir(checkpoint_dir):
        os.remove(os.path.join(checkpoint_dir, file_remove))
    os.rmdir(checkpoint_dir)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed(1)
    main()
