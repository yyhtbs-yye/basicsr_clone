import numpy as np
import pywt
import plotly.subplots as sp
import plotly.graph_objects as go
from PIL import Image

def load_image(image_path):
    """Load an image and convert it to grayscale."""
    image = Image.open(image_path).convert('L')
    return np.array(image)

def compute_multilevel_wavelet_transform(image, wavelet='haar', level=2):
    """Compute multi-level wavelet decomposition of an image."""
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    return coeffs

def compute_difference(coeffs1, coeffs2):
    """Compute the difference in wavelet coefficients between two images."""
    diff_coeffs = []
    for (c1, c2) in zip(coeffs1[1:], coeffs2[1:]):  # Skip the approximation coefficients
        # cH, cV, cD differences
        diff_horizontal = c1[0][0] - c2[0][0]
        diff_vertical = c1[0][1] - c2[0][1]
        diff_diagonal = c1[0][2] - c2[0][2]
        diff_coeffs.append((diff_horizontal, diff_vertical, diff_diagonal))
    return diff_coeffs

def plot_wavelet_difference(image1_path, image2_path, wavelet='haar', level=2):
    """Plot the differences in multi-level wavelet transforms of two images."""
    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Compute multi-level wavelet decomposition
    coeffs1 = compute_multilevel_wavelet_transform(image1, wavelet, level)
    coeffs2 = compute_multilevel_wavelet_transform(image2, wavelet, level)

    # Compute differences
    diff_coeffs = compute_difference(coeffs1, coeffs2)

    subplot_titles = []
    for i in range(level):
        subplot_titles += [
            f"Level {i+1} - Horizontal Diff", f"Level {i+1} - Vertical Diff", f"Level {i+1} - Diagonal Diff"
        ]

    fig = sp.make_subplots(
        rows=level * 2, cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
    )

    # Add difference heatmaps for each level
    for i, (dh, dv, dd) in enumerate(diff_coeffs):
        fig.add_trace(go.Heatmap(z=dh, colorscale="Viridis", colorbar=dict(title="Horizontal Diff Intensity")), row=i+1, col=1)
        fig.add_trace(go.Heatmap(z=dv, colorscale="Viridis", colorbar=dict(title="Vertical Diff Intensity")), row=i+1, col=2)
        fig.add_trace(go.Heatmap(z=dd, colorscale="Viridis", colorbar=dict(title="Diagonal Diff Intensity")), row=i+1, col=3)

    # Update layout
    fig.update_layout(
        height=600 * level,  # Adjust height dynamically based on the number of levels
        width=1200,
        title_text=f"Wavelet Transform Differences ({wavelet.capitalize()} Wavelet, {level} Levels)",
        showlegend=False,
    )

    # Show figure
    fig.show()

# Example usage
image1_path = '/home/admyyh/python_workspace/BasicSR/experiments/train_HMA_SRx2_from_DF2K_250k_smaller/visualization/baby/baby_18500.png'
image2_path = '/home/admyyh/python_workspace/BasicSR/datasets/Set5/GTmod12/baby.png'
plot_wavelet_difference(image1_path, image2_path, wavelet='haar', level=3)
