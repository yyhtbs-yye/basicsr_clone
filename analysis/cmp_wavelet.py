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

def plot_multilevel_wavelet_comparison(image1_path, image2_path, wavelet='haar', level=2):
    """Plot multi-level Wavelet Transform results for two images using Plotly."""
    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Compute multi-level wavelet decomposition
    coeffs1 = compute_multilevel_wavelet_transform(image1, wavelet, level)
    coeffs2 = compute_multilevel_wavelet_transform(image2, wavelet, level)

    # Plot horizontal, vertical, and diagonal details for each level
    subplot_titles = []
    for i in range(level):
        subplot_titles += [
            f"Level {i+1} - Horizontal (Image 1)", f"Level {i+1} - Vertical (Image 1)", f"Level {i+1} - Diagonal (Image 1)",
            f"Level {i+1} - Horizontal (Image 2)", f"Level {i+1} - Vertical (Image 2)", f"Level {i+1} - Diagonal (Image 2)"
        ]

    fig = sp.make_subplots(
        rows=level * 2, cols=3,
        subplot_titles=subplot_titles,
    )

    # Add wavelet components for each level
    for i in range(level):
        # Get detail coefficients for Image 1 and Image 2
        cH1, cV1, cD1 = coeffs1[i + 1]
        cH2, cV2, cD2 = coeffs2[i + 1]

        # Image 1 details
        fig.add_trace(go.Heatmap(z=cH1, colorscale="Viridis", showscale=False), row=i*2+1, col=1)
        fig.add_trace(go.Heatmap(z=cV1, colorscale="Viridis", showscale=False), row=i*2+1, col=2)
        fig.add_trace(go.Heatmap(z=cD1, colorscale="Viridis", showscale=False), row=i*2+1, col=3)

        # Image 2 details
        fig.add_trace(go.Heatmap(z=cH2, colorscale="Viridis", showscale=False), row=i*2+2, col=1)
        fig.add_trace(go.Heatmap(z=cV2, colorscale="Viridis", showscale=False), row=i*2+2, col=2)
        fig.add_trace(go.Heatmap(z=cD2, colorscale="Viridis", showscale=False), row=i*2+2, col=3)

    # Update layout
    fig.update_layout(
        height=800 * level,  # Adjust height dynamically for the number of levels
        width=1200,
        title_text=f"Multi-Level Wavelet Transform Comparison ({wavelet.capitalize()} Wavelet, {level} Levels)",
        showlegend=False,
    )

    # Show figure
    fig.show()

# Example usage
image1_path = '/home/admyyh/python_workspace/BasicSR/experiments/train_HMA_SRx2_from_DF2K_250k_smaller/visualization/baby/baby_18500.png'
image2_path = '/home/admyyh/python_workspace/BasicSR/datasets/Set5/GTmod12/baby.png'
plot_multilevel_wavelet_comparison(image1_path, image2_path, wavelet='haar', level=4)
