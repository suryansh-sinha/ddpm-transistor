# Custom Dataset for MVTec AD dataset.
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image

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
        image = decode_image(str(self.img_list[idx])).float() / 255.0   # Potential issue -> loads uint8 tensor. Might need to convert to torch.float32
        if self.transform:
            image = self.transform(image)
        return image