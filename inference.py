from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.transforms import v2
from torchmetrics.image import StructuralSimilarityIndexMeasure
from src.unet import UNet
from src.utils import load_checkpoint
from src.diffusion import LinearNoiseScheduler

NOISE_STEPS = 300
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02
IMG_SIZE = 256
IN_CHANNELS = 3
BASE_CHANNELS = 64  # paper used 128
DEPTH = 4
SINUSOIDAL_TIME_EMB_DIM = 128
EMB_DIM = 256
N_HEADS = 4
NUM_GROUPS = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

lns = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END, DEVICE)

# loading checkpoint data
epoch, model_weights, ema_model_weights, optimizer_state, loss_hist = load_checkpoint("./outputs/model_epoch_3650.pt")

# loading model
unet = UNet(IN_CHANNELS, BASE_CHANNELS, SINUSOIDAL_TIME_EMB_DIM,
            EMB_DIM, N_HEADS, NUM_GROUPS, DEPTH).to(DEVICE)
ema_model = torch.optim.swa_utils.AveragedModel(
    unet,
    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
)

ema_model.load_state_dict(ema_model_weights)
ema_model.eval()

ssim_metric = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0), return_full_image=True).to(DEVICE)

def load_image(img_dir, IMG_SIZE=IMG_SIZE):
    r"""
    Loads a random image from given directory with all required transforms.
    """
    img_dir = Path(img_dir)
    img_list = [f for f in img_dir.rglob("*.png") if f.is_file()]
    idx = torch.randint(0, len(img_list), (1,)).item()
    print("File Path: ", str(img_list[idx]))

    image = torchvision.io.decode_image(str(img_list[idx]))
    image = image.float() / 255.0

    transform = v2.Compose([
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.Lambda(lambda x: x*2 - 1)
    ])
    image = transform(torch.unsqueeze(image, dim=0))

    return image

def calculate_ssim(x_0):

    with torch.inference_mode():
        x_0 = x_0.to(DEVICE)
        noise = torch.randn_like(x_0).to(DEVICE)
        t = torch.tensor([NOISE_STEPS,], dtype=torch.int64, device=DEVICE)

        # Adding noise till NOISE_STEPS steps -
        x_t = lns.add_noise(x_0, noise, t)
        print("Noising Process Complete! Starting Denoising...")

        # Reverse diffusion using UNet -
        denoised_img = lns.sampling(ema_model, x_t, NOISE_STEPS)
        print("\nDenoising Complete! Calculating anomaly map and SSIM Score...")

        # Calculating SSIM Score -
        score, sim_map = ssim_metric(denoised_img, x_0)

        anomaly_map = 1 - sim_map   # 1 - similarity gives heatmap for defectiveness (high values mean not similar)
        anomaly_map = torch.mean(anomaly_map, dim=1, keepdim=True)  # mean along rgb to convert to single channel

    return x_0, x_t, denoised_img, anomaly_map, score.item()

def tensor_to_pil(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(dim=0)  # remove batch dim

    # move to CPU, detach, change back to [0, 1]
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.) / 2.
    tensor = torch.clamp(tensor, 0., 1.)

    # convert from [C, H, W] to [H, W, C]
    if tensor.shape[0] == 1:
        img_np = tensor.squeeze(0).numpy()
    else:
        img_np = tensor.permute(1, 2, 0).numpy()

    return img_np

def visualize_results(x_0, x_t, denoised_img, anomaly_map, score):
    img_x0 = tensor_to_pil(x_0)
    img_xt = tensor_to_pil(x_t)
    img_denoised = tensor_to_pil(denoised_img)
    img_anomaly = tensor_to_pil(anomaly_map)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_x0)
    axes[0].set_title("Original ($x_0$)")
    axes[0].axis("off")

    axes[1].imshow(img_xt)
    axes[1].set_title(f"Noisy ($x_t$ at $t={NOISE_STEPS}$)")
    axes[1].axis("off")

    axes[2].imshow(img_denoised)
    axes[2].set_title("Denoised Image")
    axes[2].axis("off")

    axes[3].imshow(img_anomaly, cmap="jet")
    axes[3].set_title("Anomaly Map")
    axes[3].axis("off")

    fig.suptitle(f"SSIM Score: {score:.5f}", fontsize=14)
    plt.show()

if __name__=="__main__":
    x_0 = load_image("./data/transistor/test/damaged_case/")
    x_0, x_t, denoised_img, anomaly_map, score = calculate_ssim(x_0)
    visualize_results(x_0, x_t, denoised_img, anomaly_map, score)
