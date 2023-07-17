from pathlib import Path

import pandas as pd


def extract_to_df(img_dir: Path, label_file: Path = None) -> pd.DataFrame:
    """Return a dataframe containing the image path with corresponding labels."""
    paths = get_img_paths(img_dir)
    if label_file is None:
        return pd.DataFrame({"path": paths})
    labels = get_labels(label_file)
    return pd.DataFrame({
        "path": paths,
        "label": labels
    })


def get_img_paths(img_dir: Path) -> list[Path]:
    paths = list(img_dir.iterdir())
    paths.sort(key=lambda p: p.stem)
    return paths


def get_labels(train_label_file: Path) -> list[int]:
    with open(train_label_file) as f:
        labels = [int(line.strip()) for line in f]
    return labels
