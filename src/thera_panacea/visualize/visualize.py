from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd


def visualize_sample(df: pd.DataFrame, n_rows: int = 3, n_cols: int = 5, title="label"):
    n_samples = n_rows * n_cols
    if n_samples <= 1:
        raise ValueError("Sample size (n_rows*n_cols) should be at least 2.")
    if len(df) == n_samples:
        sample_df = df
    else:
        sample_df = df.sample(n_samples)

    fig, axes = plt.subplots(n_rows, n_cols)
    if n_rows == 1:
        axes = axes.reshape(1, n_cols)
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    fig.set_size_inches(16, 9)

    for i in range(n_rows):
        for j in range(n_cols):
            ax: plt.Axes = axes[i, j]
            idx = i * n_cols + j
            img_path, label = sample_df.iloc[idx]
            with Image.open(img_path) as img:
                ax.imshow(img)
                # img = np.asarray(f)
            ax.set_title(f"label: {label} ({img_path.name})")
            ax.set_axis_off()
