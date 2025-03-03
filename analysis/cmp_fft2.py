import numpy as np
import plotly.subplots as sp
import plotly.graph_objects as go
from PIL import Image

def load_image(image_path):
    """Load an image and convert it to grayscale."""
    image = Image.open(image_path).convert('L')
    return np.array(image)

def compute_fourier_spectrum(image):
    """Compute the Fourier spectrum of an image."""
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)  # Shift zero frequency to center
    magnitude_spectrum = np.log(1 + np.abs(f_shift))  # Log scale for visualization
    return magnitude_spectrum

def plot_fourier_spectrums_plotly(image1_path, image2_path):
    """Plot the Fourier spectrums of two images using Plotly with separate colorbars."""
    # Load images
    image1 = load_image(image1_path)
    image2 = load_image(image2_path)

    # Compute Fourier spectrums
    spectrum1 = compute_fourier_spectrum(image1)
    spectrum2 = compute_fourier_spectrum(image2)

    # Create a Plotly subplot
    fig = sp.make_subplots(
        rows=2, cols=2,
        subplot_titles=("Image 1", "Fourier Spectrum 1", "Image 2", "Fourier Spectrum 2"),
        vertical_spacing=0.1,
    )

    # Add Image 1
    fig.add_trace(go.Image(z=image1), row=1, col=1)

    # Add Fourier Spectrum 1
    fig.add_trace(
        go.Heatmap(
            z=spectrum1,
            colorscale="Viridis",
            colorbar=dict(title="Spectrum 1 Intensity", x=0.48),  # Adjust position
        ),
        row=1, col=2,
    )

    # Add Image 2
    fig.add_trace(go.Image(z=image2), row=2, col=1)

    # Add Fourier Spectrum 2
    fig.add_trace(
        go.Heatmap(
            z=spectrum2,
            colorscale="Viridis",
            colorbar=dict(title="Spectrum 2 Intensity", x=1.02),  # Adjust position
        ),
        row=2, col=2,
    )

    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text="Fourier Spectrum Comparison with Separate Colorbars",
        showlegend=False,
    )

    # Show figure
    fig.show()

# Example usage
image1_path = '/home/admyyh/python_workspace/BasicSR/experiments/train_HMA_SRx2_from_DF2K_250k_smaller/visualization/baby/baby_18500.png'
image2_path = '/home/admyyh/python_workspace/BasicSR/datasets/Set5/GTmod12/baby.png'
plot_fourier_spectrums_plotly(image1_path, image2_path)
