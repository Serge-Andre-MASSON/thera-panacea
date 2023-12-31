import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class TherasDS(Dataset):
    def __init__(self, dataset_df: pd.DataFrame, device: str, preprocess) -> None:
        super().__init__()
        self.dataset_df = dataset_df
        self.device = device
        self.preprocess = preprocess

    def __getitem__(self, index):
        label = None
        try:
            path, label = self.dataset_df.iloc[index]
        except:
            path = self.dataset_df.iloc[index].item()

        with Image.open(path) as img:
            t: torch.Tensor = self.preprocess(img)
        if label is None:
            return t.to(self.device)
        return t.to(self.device), torch.tensor(label, device=self.device)

    def __len__(self):
        return len(self.dataset_df)
