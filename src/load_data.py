# Custom Datasets for MVTec AD dataset.
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2


class TransistorDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Create a list of all img paths -
        self.img_list = [f for f in self.img_dir.glob("*.png") if f.is_file()]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = (
            decode_image(str(self.img_list[idx])).float() / 255.0
        )
        if self.transform:
            image = self.transform(image)
        return image


class TestDataset(Dataset):
    def __init__(self, test_img_dir, gt_dir, img_size=256):
        super().__init__()
        self.img_dir = Path(test_img_dir)
        self.gt_dir = Path(gt_dir)
        self.img_size = img_size

        # Image transform: Resize and normalize to [-1, 1]
        self.transform = v2.Compose(
            [
                v2.Resize(
                    (img_size, img_size), interpolation=v2.InterpolationMode.BILINEAR
                ),
                v2.Lambda(lambda x: x * 2 - 1),
            ]
        )

        # Mask transform:
        # 1. Grayscale: Ensures single channel (safe no-op if already 1-channel)
        # 2. Resize with NEAREST. Bilinear creates fractional edge pixels (e.g., 0.3),
        #    ruining the strict 0/1 requirement for pixel-level AUROC.
        # 3. ToDtype(scale=True): Takes the raw uint8 [0, 255] from decode_image and
        #    correctly scales it to float32 [0.0, 1.0].
        #    (Calling .float() BEFORE this breaks the scaling)
        self.mask_transform = v2.Compose(
            [
                v2.Grayscale(num_output_channels=1),
                v2.Resize(
                    (img_size, img_size), interpolation=v2.InterpolationMode.NEAREST
                ),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        self.samples = []

        for defect_type_dir in sorted(self.img_dir.iterdir()):
            if not defect_type_dir.is_dir():
                continue

            defect_type = defect_type_dir.name
            label = 0 if defect_type == "good" else 1

            for img_path in sorted(defect_type_dir.glob("*.png")):
                if defect_type == "good":
                    mask_path = None
                else:
                    stem = img_path.stem
                    mask_filename = f"{stem}_mask.png"
                    mask_path = self.gt_dir / defect_type / mask_filename

                    # Hard fail on missing masks.
                    # Silent zeros for defective images will tank Pixel AUROC silently.
                    if not mask_path.exists():
                        raise FileNotFoundError(
                            f"Critical Data Error: Mask not found for defective image '{img_path.name}'. "
                            f"Expected at: {mask_path}"
                        )

                self.samples.append(
                    {
                        "idx": len(
                            self.samples
                        ),  # Dataset index for reproducible noise/batching
                        "img_path": str(img_path),  # File path for tracing results
                        "label": label,
                        "defect_type": defect_type,
                        "mask_path": str(mask_path) if mask_path else None,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load and transform image
        # decode_image returns uint8 [0, 255]. We normalize to [0, 1] first, then transform to [-1, 1]
        image = decode_image(str(sample["img_path"])).float() / 255.0
        image = self.transform(image)

        # 2. Load and transform mask
        if sample["mask_path"] is not None:
            # Pass raw uint8 tensor directly to transform. ToDtype(scale=True) handles the [0, 255] -> [0.0, 1.0] conversion.
            mask = decode_image(str(sample["mask_path"]))
            mask = self.mask_transform(mask)
        else:
            # For 'good' images, return a strict 0/1 zero tensor
            mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        return {
            "idx": sample["idx"],
            "img_path": sample["img_path"],
            "image": image,  # [3, H, W], float32, [-1, 1]
            "label": sample["label"],  # int (0 or 1)
            "defect_type": sample["defect_type"],
            "mask": mask,  # [1, H, W], float32, [0.0, 1.0]
        }
