# image_filters.py
"""
Pre-defined image processing functions.
Students: Use these functions to process images in your pipeline.
"""

from PIL import Image, ImageFilter, ImageStat
import os


def load_image(path):
    """
    Load an image from disk.

    Args:
        path: Path to image file (.png, .jpg, etc.)

    Returns:
        PIL Image in RGB mode
    """
    return Image.open(path).convert('RGB')


def to_grayscale(img):
    """
    Convert a color image to grayscale.
    Each pixel becomes a single brightness value (0 = black, 255 = white).

    Args:
        img: PIL Image

    Returns:
        PIL Image in grayscale ('L' mode)
    """
    return img.convert('L')


def apply_blur(img, radius=2):
    """
    Apply Gaussian blur to smooth the image and reduce noise.

    Args:
        img: PIL Image
        radius: Blur strength (default 2 — higher = more blurring)

    Returns:
        Blurred PIL Image
    """
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def detect_edges(img):
    """
    Detect edges in an image using a convolution filter.
    Pixels on sharp color boundaries become bright; smooth regions become dark.
    Converts to grayscale internally if the image is color.

    Args:
        img: PIL Image

    Returns:
        Edge-detected PIL Image (grayscale)
    """
    gray = img.convert('L')
    return gray.filter(ImageFilter.FIND_EDGES)


def compute_stats(img):
    """
    Compute three statistics that characterize an image:

        mean_brightness — average pixel brightness (0–255).
                          High value = bright image; low value = dark image.

        std_brightness  — standard deviation of pixel brightness.
                          High value = high contrast (lots of variation);
                          low value = uniform / flat image.

        edge_density    — fraction of pixels detected as edges (0.0–1.0).
                          High value = many edges (sharp shapes or noise);
                          low value = smooth, featureless image.

    Args:
        img: PIL Image (color or grayscale)

    Returns:
        dict with keys: 'mean_brightness', 'std_brightness', 'edge_density'
    """
    gray = img.convert('L')
    stat = ImageStat.Stat(gray)

    edge_img = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edge_img)

    return {
        'mean_brightness': round(stat.mean[0], 2),
        'std_brightness':  round(stat.stddev[0], 2),
        'edge_density':    round(edge_stat.mean[0] / 255.0, 4),
    }


def save_image(img, path):
    """
    Save an image to disk. Creates any missing directories automatically.

    Args:
        img: PIL Image
        path: Destination file path (e.g. 'output/batch_A/img_001.png')
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
