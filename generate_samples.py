from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2

from src.diffusion import LinearNoiseScheduler
from src.unet import UNet
from src.utils import load_checkpoint

# Hyperparameters (must match training)
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
IMG_SIZE = 256
IN_CHANNELS = 3
BASE_CHANNELS = 64
DEPTH = 4
SINUSOIDAL_TIME_EMB_DIM = 128
EMB_DIM = 256
N_HEADS = 4
NUM_GROUPS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def main():
    print("Loading model...")
    lns = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END, DEVICE)
    checkpoint_path = "./outputs/model_epoch_3650.pt"
    _, _, ema_model_weights, _, _ = load_checkpoint(checkpoint_path)

    unet = UNet(
        IN_CHANNELS,
        BASE_CHANNELS,
        SINUSOIDAL_TIME_EMB_DIM,
        EMB_DIM,
        N_HEADS,
        NUM_GROUPS,
        DEPTH,
    ).to(DEVICE)
    ema_model = torch.optim.swa_utils.AveragedModel(
        unet, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
    )
    ema_model.load_state_dict(ema_model_weights)
    ema_model.eval()

    # 1. Generate Samples (seeded)
    num_samples = 8
    print(f"Generating {num_samples} samples with seed {SEED}...")
    with torch.inference_mode():
        noise = torch.randn(
            (num_samples, IN_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE
        )
        generated = lns.sampling(ema_model.module, noise)
        generated = (generated + 1.0) / 2.0  # Scale to [0, 1]
        generated = generated.clamp(0, 1).cpu()

    # 2. Load ALL training images + their horizontal flips
    train_dir = Path("./data/transistor/train/good")
    train_imgs = []
    transform = v2.Compose(
        [v2.Resize((IMG_SIZE, IMG_SIZE)), v2.Lambda(lambda x: x.float() / 255.0)]
    )

    print("Loading full training set with horizontal flips...")
    for img_path in train_dir.glob("*.png"):
        img = torchvision.io.decode_image(str(img_path))
        img_transformed = transform(img)
        train_imgs.append(img_transformed)
        # Add horizontally flipped version (training used RandomHorizontalFlip)
        train_imgs.append(v2.functional.hflip(img_transformed))

    train_tensor = torch.stack(train_imgs).to(DEVICE)  # [2*213, 3, 256, 256]
    num_train = train_tensor.shape[0]
    print(f"Loaded {num_train} training images (including flips)")

    # 3. Load test-good images for distance comparison
    test_dir = Path("./data/transistor/test/good")
    test_imgs = []
    print("Loading test-good images for distance comparison...")
    for img_path in test_dir.glob("*.png"):
        img = torchvision.io.decode_image(str(img_path))
        test_imgs.append(transform(img))

    test_tensor = torch.stack(test_imgs).to(DEVICE)  # [60, 3, 256, 256]
    num_test = test_tensor.shape[0]
    print(f"Loaded {num_test} test-good images")

    # 4. Find Nearest Neighbors (L2 distance in pixel space)
    print("Computing nearest training neighbors for generated samples...")
    gen_flat = generated.view(num_samples, -1).to(DEVICE)
    train_flat = train_tensor.view(num_train, -1)

    # Compute pairwise L2 distance: [num_samples, num_train]
    distances_gen_to_train = torch.cdist(gen_flat, train_flat, p=2)
    min_distances_gen, min_indices_gen = torch.min(distances_gen_to_train, dim=1)

    nearest_train_imgs = train_tensor[min_indices_gen].cpu()
    min_distances_gen = min_distances_gen.cpu().numpy()

    # 5. Compute test-good-to-train distances for comparison
    print("Computing test-good-to-train distances for baseline comparison...")
    test_flat = test_tensor.view(num_test, -1)
    distances_test_to_train = torch.cdist(test_flat, train_flat, p=2)
    min_distances_test, _ = torch.min(distances_test_to_train, dim=1)

    # Get statistics
    gen_mean_dist = min_distances_gen.mean()
    gen_std_dist = min_distances_gen.std()
    test_mean_dist = min_distances_test.mean()
    test_std_dist = min_distances_test.std()

    print(f"\n{'=' * 60}")
    print("DISTANCE ANALYSIS (L2 in pixel space)")
    print(f"{'=' * 60}")
    print(f"Generated → Train:  mean={gen_mean_dist:.2f}, std={gen_std_dist:.2f}")
    print(f"Test-Good → Train:  mean={test_mean_dist:.2f}, std={test_std_dist:.2f}")
    print(f"{'=' * 60}")

    # 6. Visualize
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        # Generated
        axes[i, 0].imshow(generated[i].permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Generated Sample")
        axes[i, 0].axis("off")

        # Nearest Training
        axes[i, 1].imshow(nearest_train_imgs[i].permute(1, 2, 0).numpy())
        axes[i, 1].set_title("Nearest Training Neighbor")
        axes[i, 1].axis("off")

        # Difference
        diff_img = torch.abs(generated[i] - nearest_train_imgs[i])
        l1_diff = diff_img.mean().item()
        axes[i, 2].imshow(diff_img.permute(1, 2, 0).numpy())
        axes[i, 2].set_title(f"L2: {min_distances_gen[i]:.2f}\nL1: {l1_diff:.4f}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("./outputs/generation_and_nn_check.png", dpi=150)
    print("Saved generation_and_nn_check.png")

    # Save distance stats to file for README reference
    with open("./outputs/distance_stats.txt", "w") as f:
        f.write(
            f"Generated → Train: mean={gen_mean_dist:.2f}, std={gen_std_dist:.2f}\n"
        )
        f.write(
            f"Test-Good → Train: mean={test_mean_dist:.2f}, std={test_std_dist:.2f}\n"
        )
    print("Saved distance_stats.txt")


if __name__ == "__main__":
    main()
