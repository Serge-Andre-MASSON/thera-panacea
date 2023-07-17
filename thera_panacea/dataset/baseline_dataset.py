import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize


class BaselineDS(Dataset):
    def __init__(self, dataset_df: pd.DataFrame, device: str) -> None:
        super().__init__()
        self.dataset_df = dataset_df
        self.to_tensor = ToTensor()
        self.device = device

        self.preprocess = transforms.Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def set_preprocess(self, preprocess):
        self.preprocess = preprocess

    def __getitem__(self, index):
        path, label = self.dataset_df.iloc[index]

        with Image.open(path) as img:
            t: torch.Tensor = self.preprocess(img)

        return t.to(self.device), torch.tensor(label, device=self.device)

    def __len__(self):
        return len(self.dataset_df)
