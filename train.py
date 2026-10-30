import os

import torch
from dotenv import load_dotenv
from torch.amp import autocast
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import make_grid

import wandb
from src.diffusion import LinearNoiseScheduler
from src.load_data import TransistorDataset
from src.unet import UNet
from src.utils import save_model_checkpoint

load_dotenv()

# Hyperparameters
BATCH_SIZE = 16
LR = 1e-4  # paper default is 2e-4
EPOCHS = 6500
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
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

config = {
    "BATCH_SIZE": BATCH_SIZE,
    "LR": LR,
    "EPOCHS": EPOCHS,
    "NUM_TIMESTEPS": NUM_TIMESTEPS,
    "BETA_START": BETA_START,
    "BETA_END": BETA_END,
    "IMG_SIZE": IMG_SIZE,
    "IN_CHANNELS": IN_CHANNELS,
    "BASE_CHANNELS": BASE_CHANNELS,
    "DEPTH": DEPTH,
    "SINUSOIDAL_TIME_EMB_DIM": SINUSOIDAL_TIME_EMB_DIM,
    "EMB_DIM": EMB_DIM,
    "N_HEADS": N_HEADS,
    "NUM_GROUPS": NUM_GROUPS,
    "DEVICE": DEVICE,
}

wandb.login(key=os.getenv("WANDB_API_KEY"))
wandb.init(project="DDPM-MVTecAD", config=config)

# Loading our dataset
transform = v2.Compose(
    [
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.RandomHorizontalFlip(),
        v2.Lambda(lambda x: x * 2 - 1),
    ]
)

dataset = TransistorDataset(
    img_dir="./data/transistor/train/good/", transform=transform
)

# Objects
lns = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END, DEVICE)
model = UNet(
    IN_CHANNELS,
    BASE_CHANNELS,
    SINUSOIDAL_TIME_EMB_DIM,
    EMB_DIM,
    N_HEADS,
    NUM_GROUPS,
    DEPTH,
).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
ema_model = torch.optim.swa_utils.AveragedModel(
    model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
)
criterion = torch.nn.MSELoss()

# Create DataLoader
train_loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, num_workers=8)

# Training starts
fixed_noise = torch.randn(
    (8, IN_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE, requires_grad=False
)  # Fixed noise to test generated samples
best_loss = 1000
losses = []

for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    ema_model.train()
    train_loss = 0.0
    total_imgs = 0

    for batch_idx, x_0 in enumerate(train_loader):
        optimizer.zero_grad()
        x_0 = x_0.to(DEVICE)

        # t ~ Uniform ({1...T})
        t = torch.randint(low=0, high=NUM_TIMESTEPS, size=(x_0.size(0),)).to(DEVICE)

        # noise ~ N(0, I)
        noise = torch.randn_like(x_0).to(DEVICE)

        # forward diffusion process
        x_t = lns.add_noise(x_0, noise, t)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            # Loss -> ||noise - unet_pred(add_noise function to timesteps t, timestep t itself)|| ** 2
            predicted_noise = model(x_t, t)  # getting model prediction
            loss = criterion(predicted_noise, noise)

        loss.backward()  # calculate gradients
        optimizer.step()  # update model parameters

        ema_model.update_parameters(model)

        train_loss += loss.item() * x_0.size(0)
        total_imgs += x_0.size(0)

        if (batch_idx) % 8 == 0:
            print(f"Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.5f}")

        wandb.log({"batch_loss": loss.item(), "batch_idx": batch_idx})

    # In case the total dataset size is not divisible by batch_size, we explicitly count number of images in last batch and add them for correct division to get epoch_loss
    epoch_train_loss = train_loss / total_imgs

    # Saving best performing model
    if epoch_train_loss < best_loss:
        best_loss = epoch_train_loss
        save_model_checkpoint(
            "./outputs/best.pt", epoch, model, ema_model, optimizer, losses
        )

    losses.append(epoch_train_loss)
    print("-" * 60)
    print(f"Epoch: {epoch + 1}/{EPOCHS} | Loss: {epoch_train_loss:.5f}")
    print("-" * 60)

    if epoch == 0:
        img_grid = make_grid(fixed_noise, 4).to(torch.device("cpu"))
        wandb.log(
            {
                "epoch_loss": epoch_train_loss,
                "epoch": epoch,
                "validation_samples": wandb.Image(
                    img_grid, caption=f"Sampled at epoch: {epoch + 1}"
                ),
            }
        )
    elif (epoch + 1) % 50 == 0:
        save_model_checkpoint(
            f"./outputs/model_epoch_{epoch + 1}.pt",
            epoch,
            model,
            ema_model,
            optimizer,
            losses,
        )
        # sampling logic
        ema_model.eval()
        with torch.inference_mode():
            generated = lns.sampling(ema_model.module, fixed_noise)
            generated = (generated + 1) / 2
            img_grid = make_grid(generated, 4).to(torch.device("cpu"))
            # log generated samples to w&b
            wandb.log(
                {
                    "epoch_loss": epoch_train_loss,
                    "epoch": epoch,
                    "validation_samples": wandb.Image(
                        img_grid, caption=f"Sampled at epoch: {epoch + 1}"
                    ),
                }
            )
    else:
        wandb.log({"epoch_loss": epoch_train_loss, "epoch": epoch})
