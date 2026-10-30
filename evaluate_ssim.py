import json
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure

from src.diffusion import LinearNoiseScheduler
from src.load_data import TestDataset
from src.unet import UNet
from src.utils import load_checkpoint

## Hyperparameters and Setup
NOISE_STEPS_TO_EVAL = [25, 50, 75, 100, 200, 300, 400]  # Sweep across a few noise levels
README_NOISE_STEP = 75  # Save denoised images ONLY for this step
BATCH_SIZE = 4
IMG_SIZE = 256
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SMOKE_TEST = False

# Fixing Reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

## Load Model and Dataset
print("Loading model and dataset...")
lns = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END, DEVICE)

checkpoint_path = "./outputs/model_epoch_3650.pt"
epoch, model_weights, ema_model_weights, optimizer_state, loss_hist = load_checkpoint(
    checkpoint_path
)

unet = UNet(
    in_channels=3,
    base_channels=64,
    sinusoidal_time_emb_dim=128,
    qkv_dim=256,
    n_heads=4,
    num_groups=32,
    depth=4,
).to(DEVICE)
ema_model = torch.optim.swa_utils.AveragedModel(
    unet, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
)
ema_model.load_state_dict(ema_model_weights)
ema_model.eval()

full_dataset = TestDataset(
    test_img_dir="./data/transistor/test",
    gt_dir="./data/transistor/ground_truth",
    img_size=IMG_SIZE,
)

# SMOKE TEST: Using a subset of 10 images
if SMOKE_TEST:
    indices_by_type = {}
    for i, s in enumerate(full_dataset.samples):
        dtype = s["defect_type"]
        if dtype not in indices_by_type:
            indices_by_type[dtype] = []
        indices_by_type[dtype].append(i)

    smoke_indices = []
    for dtype in ["bent_lead", "cut_lead", "damaged_case", "misplaced"]:
        if indices_by_type[dtype]:
            smoke_indices.append(indices_by_type[dtype][0])
    smoke_indices.extend(indices_by_type["good"][:2])

    dataset = torch.utils.data.Subset(full_dataset, smoke_indices)
    print(f"Running SMOKE TEST on {len(dataset)} images: {smoke_indices}")
else:
    dataset = full_dataset
    print(f"Running FULL EVALUATION on {len(dataset)} images.")

# shuffle=False for reproducability
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

## Shared Reconstruction Function
def reconstruct_and_compute_ssim(x_0, shared_noise, t_step, lns, ema_model, device):
    """
    x_0: [B, C, H, W] tensor in [-1, 1]
    shared_noise: [B, C, H, W] tensor, the EXACT SAME noise pattern used across all t_steps for this batch
    t_step = int, the noise level to add and start denoising from
    """
    batch_size = x_0.shape[0]
    with torch.inference_mode():
        # Add noise using the shared noise tensor
        t_tensor = torch.full((batch_size,), t_step, device=device, dtype=torch.long)
        x_t = lns.add_noise(x_0, shared_noise, t_tensor)

        # Denoise
        # starting_step = t_step + 1 so the loop in sampling starts at t_step and goes down to 0
        denoised = lns.sampling(ema_model, x_t, starting_step=t_step + 1)

        # Compute SSIM and scale to [0, 1] for standard SSIM calculation
        denoised_normalized = (denoised + 1.0) / 2.0
        x_0_normalized = (x_0 + 1.0) / 2.0

        ssim_score_batch, ssim_map = structural_similarity_index_measure(
            denoised_normalized, x_0_normalized, data_range=1.0, return_full_image=True, reduction='none'
        )

        # ssim_map shape is [B, C, H, W]. Average over channels to get [B, H, W]
        ssim_maps = ssim_map.mean(dim=1)
        # per-image score is the mean of the SSIM map for each image, yielding shape [B]
        per_image_ssim = ssim_maps.mean(dim=(1,2))

    return denoised, ssim_maps, per_image_ssim, ssim_score_batch


## Evaluation Loop
output_dir = Path("./eval_outputs")
# Automatically clear the directory to prevent duplicate metadata across runs
if output_dir.exists():
    print(f"Clearing existing {output_dir} to prevent duplicate metadata...")
    shutil.rmtree(output_dir)
output_dir.mkdir(exist_ok=True)

print("Starting evaluation...")
start_time = time.time()
batch_count = 0

for batch in dataloader:
    batch_start = time.time()
    images = batch["image"].to(DEVICE)
    labels = batch["label"]
    defect_types = batch["defect_type"]
    masks = batch["mask"]
    paths = batch["img_path"]
    dataset_indices = batch["idx"]  # Original dataset indices

    # Pre-generate shared initial noise for this batch. For reproducible noise.
    shared_noise = torch.randn_like(images, device=DEVICE)

    for t_step in NOISE_STEPS_TO_EVAL:
        t_dir = output_dir / f"t_{t_step}"
        t_dir.mkdir(exist_ok=True)

        denoised, ssim_maps, per_image_ssim, ssim_score_batch = reconstruct_and_compute_ssim(
            images, shared_noise, t_step, lns, ema_model, DEVICE
        )

        denoised_cpu = denoised.cpu()
        ssim_maps_cpu = ssim_maps.cpu()
        per_image_ssim_cpu = per_image_ssim.cpu()
        masks_cpu = masks.cpu()

        # Save incrementally per batch, using dataset_idx for filenames
        meta_file = t_dir / "metadata.json"
        existing_meta = []
        if meta_file.exists():
            with open(meta_file, "r") as f:
                existing_meta = json.load(f)

        for i in range(len(images)):
            d_idx = dataset_indices[i].item()

            map_path = t_dir / f"map_{d_idx:03d}.pt"
            torch.save(ssim_maps_cpu[i], map_path)

            mask_path = t_dir / f"mask_{d_idx:03d}.pt"
            torch.save(masks_cpu[i], mask_path)

            meta_entry = {
                "dataset_idx": d_idx,
                "path": paths[i],
                "label": labels[i].item(),
                "defect_type": defect_types[i],
                "ssim_score": per_image_ssim_cpu[i].item(),
                "map_path": str(map_path),
                "mask_path": str(mask_path),  # Save mask path for scoring script
            }

            if t_step == README_NOISE_STEP:
                denoised_path = t_dir / f"denoised_{d_idx:03d}.pt"
                torch.save(denoised_cpu[i], denoised_path)
                meta_entry["denoised_path"] = str(denoised_path)

            existing_meta.append(meta_entry)

        # Overwrite metadata.json with the updated list
        with open(meta_file, "w") as f:
            json.dump(existing_meta, f, indent=2)

    batch_count += 1
    batch_time = time.time() - batch_start
    print(
        f"  Processed batch {batch_count}/{len(dataloader)} in {batch_time:.2f}s. "
        f"Estimated remaining time: {((len(dataloader) - batch_count) * batch_time) / 60:.1f} mins"
    )

total_time = time.time() - start_time
print(f"\nEvaluation complete! Total time: {total_time / 60:.1f} mins.")
