# Writing an eval class to run Image Level AUROC and Per-Pixel AUROC
# Evaluating AUROC based on SSIM only.
# TODO:
# Write a test dataset for evaluation
# Each item from test datset needs to return the image, mask, label, and mask (gt)
# For good images we return an all black mask
#
#

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2


class TestDataset(Dataset):
    def __init__(self, test_img_dir, gt_dir, img_size=256):
        super().__init__()
        self.img_dir = Path(test_img_dir)   # img dir
        self.gt_dir = Path(gt_dir)  # mask dir
        self.img_size = img_size    # img size (same as the one used during training)

        # Same transforms
        self.transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.Lambda(lambda x: x*2 - 1)
        ])

        self.mask_transform = v2.Compose([
            v2.Resize((img_size, img_size)),
            v2.Grayscale(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        self.samples = []

        for defect_type_dir in sorted(self.img_dir.iterdir()):
            if not defect_type_dir.is_dir():
                continue

            defect_type = defect_type_dir.name
            label = 0 if defect_type == "good" else 1

            for img_path in sorted(defect_type_dir.glob("*.png")):
                # Construct expected mask path
                if defect_type == "good":
                    mask_path = None
                else:
                    stem = img_path.stem    # "000.png" becomes "000"
                    mask_filename = f"{stem}_mask.png"
                    mask_path = self.gt_dir / defect_type / mask_filename

                    if not mask_path.exists():
                        print(f"Warning: Mask not found for {img_path.name}, expected at {mask_path}")
                        mask_path = None

                self.samples.append({
                    "img_path": img_path,
                    "label": label,
                    "defect_type": defect_type,
                    "mask_path": mask_path
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and transform the image
        image = decode_image(str(sample["img_path"]))   # returns byte tensor with range (0, 255)
        image = image.float() / 255.0   # Converting to float and normalizing between 0 and 1.
        print("Image type and range --> ", image.type(), image.min(), image.max())
        image = self.transform(image)

        # Load or generate the mask
        # For pixel level AUROC, mask must be strictly 0/1 with the right shape.
        if sample["mask_path"] is not None and sample["mask_path"].exists():
            mask = decode_image(str(sample["mask_path"])).float()
            print("Mask type and range --> ", mask.type(), mask.min(), mask.max())
            mask = self.mask_transform(mask)
            print("Mask (After Transform) type and range --> ", mask.type(), mask.min(), mask.max(), mask.unique())
        else:
            # For 'good' images or missing masks, return a zero tensor (black mask)
            mask = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float32)

        return {
            "image": image,
            "label": sample["label"],
            "defect_type": sample["defect_type"],
            "mask": mask
        }



if __name__ == "__main__":
    IMG_SIZE = 256

    test_dataset = TestDataset(
        test_img_dir="./data/transistor/test",
        gt_dir="./data/transistor/ground_truth",
        img_size=IMG_SIZE
    )

    print(len(test_dataset))
    out_dict = next(iter(test_dataset))
