import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def df_to_arrays(df: pd.DataFrame):
    """Return a numpy array containing a black and white version of the images taken from the "path" column of df."""
    n_samples = len(df)
    X = np.zeros(shape=(n_samples, 64*64))
    y = np.array(df["label"])
    for i in tqdm(range(n_samples), total=n_samples):
        path, _ = df.iloc[i]
        with Image.open(path) as img:
            X[i] = np.array(img.convert("L")).ravel()
    return X, y
