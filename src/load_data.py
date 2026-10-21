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

# ==========================================
# SANITY CHECKLIST (Run this to verify everything)
# ==========================================
# if __name__ == "__main__":
#     IMG_SIZE = 256
#
#     test_dataset = TestDataset(
#         test_img_dir="./data/transistor/test",
#         gt_dir="./data/transistor/ground_truth",
#         img_size=IMG_SIZE,
#     )
#
#     # 1. Check length and distribution
#     print(f"1. Dataset length: {len(test_dataset)} (Expected: 100)")
#     counts = Counter([s["defect_type"] for s in test_dataset.samples])
#     print(f"   Breakdown: {dict(counts)}\n")
#
#     # 2. Check image properties
#     sample_0 = test_dataset[0]
#     img = sample_0["image"]
#     print("2. Image properties (sample 0):")
#     print(f"   Shape: {list(img.shape)} (Expected: [3, 256, 256])")
#     print(f"   Dtype: {img.dtype} (Expected: torch.float32)")
#     print(
#         f"   Range: [{img.min().item():.2f}, {img.max().item():.2f}] (Expected: ~[-1.0, 1.0])\n"
#     )
#
#     # 3. Check mask properties for one image per defect type
#     print("3. Mask properties per defect type:")
#     checked_defects = set()
#     for sample in test_dataset:
#         dtype = sample["defect_type"]
#         if dtype in checked_defects:
#             continue
#         checked_defects.add(dtype)
#
#         mask = sample["mask"]
#         unique_vals = sorted(torch.unique(mask).tolist())
#         print(
#             f"   {dtype:12} | Shape: {str(list(mask.shape)):12} | Unique: {unique_vals} | Sum: {mask.sum().item():.1f}"
#         )
#
#         # Assert strict 0/1 for defective masks
#         if dtype != "good":
#             assert set(unique_vals).issubset({0.0, 1.0}), (
#                 f"FAIL: Mask for {dtype} has non-binary values!"
#             )
#             assert mask.sum().item() > 0, f"FAIL: Mask for {dtype} is empty!"
#     print("   -> All mask checks passed!\n")
#
#     # 4. Visual Overlay Check
#     print("4. Generating visual overlay check for 2 defective samples...")
#     defective_samples = [s for s in test_dataset if s["defect_type"] != "good"]
#
#     fig, axes = plt.subplots(2, 3, figsize=(12, 8))
#     for i, sample in enumerate(defective_samples[:2]):
#         # Image is [-1, 1], convert to [0, 1] for visualization
#         img_viz = (sample["image"] + 1) / 2.0
#         img_viz = img_viz.permute(1, 2, 0).numpy()  # [H, W, 3]
#
#         # Mask is [1, H, W], convert to [H, W]
#         mask_viz = sample["mask"].squeeze(0).numpy()
#
#         # Create overlay: highlight mask region in red
#         overlay = img_viz.copy()
#         mask_3d = np.stack([mask_viz] * 3, axis=-1)
#         # Blend original image with red where mask is present
#         overlay = np.where(
#             mask_3d > 0.5, overlay * 0.6 + np.array([1.0, 0.0, 0.0]) * 0.4, overlay
#         )
#
#         axes[i, 0].imshow(img_viz)
#         axes[i, 0].set_title(f"Original: {sample['defect_type']}")
#         axes[i, 0].axis("off")
#
#         axes[i, 1].imshow(mask_viz, cmap="gray")
#         axes[i, 1].set_title("Ground Truth Mask")
#         axes[i, 1].axis("off")
#
#         axes[i, 2].imshow(overlay)
#         axes[i, 2].set_title("Overlay (Red = Defect)")
#         axes[i, 2].axis("off")
#
#     plt.tight_layout()
#     plt.savefig("mask_overlay_check.png")
#     print(
#         "   Saved overlay check to 'mask_overlay_check.png'. Please open and visually verify alignment!"
#     )
#     plt.show()
