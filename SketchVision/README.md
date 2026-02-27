## Project Structure
The SketchVision directory is organized as follows:

| Directory | Description |
| :--- | :--- |
| **[configuration/](./configuration)** | Configurations for diffusion model training |
| **[eBPF/](./eBPF)** | eBPF-enabled version of SketchVision |
| **[include/](./include)** | Header files |
| **[src/](./src)** | Source code |
| **[ML/](./ML)** | Diffusion model training & inference |
| **[script/](./script)** | Utility scripts |
| **[models/](https://zenodo.org/records/17903481)** | Pretrained models |
| **[data/](https://zenodo.org/records/17903481)** | datasets |

## Running SketchVision

1. Download the dataset and pretrained models using the link above.
2. Replace the contents of the `data/` and `models/` directories.
3. Run:
   - `invoke run-test`, or
   - `invoke run-X` where `X ∈ {C2, bot, ddos, exfil, surveil}`.
4. The process generates a set of `.mkv` files in `sketch_out` directory, each representing a single flow with per-packet encoded information over time.
5. Use the scripts in the `ML/` directory for denoising and detection.
6. For training, place training data inside `ML/Dataset_mixSK`.
7. Run `diffusion_training.py 99`. The argument 99 instructs to load configuration/args99.json as training configuration.
