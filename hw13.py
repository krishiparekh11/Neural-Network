"""
HW - Chapter 13 “More Convolutions and Transfer Learning”
Programming Assignment:
Develop a program for depthwise and pointwise convolutions. The program should take an initial image, kernel (filter), and produce the convoluted image based on the external flag that indicates either the depthwise or pointwise convolution.
"""

import argparse

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image


# ---------------------------------------------------------------------------
# Kernels (defined here so no separate files are needed)
# ---------------------------------------------------------------------------

# Depthwise kernel: one 5 x 5 kernel for each of the 3 channels -> shape (5, 5, 3)
DEPTHWISE_KERNEL = np.random.default_rng(1).standard_normal((5, 5, 3))

# Pointwise kernel: 256 kernels of size 1 x 1 x 3 -> shape (3, 256)
POINTWISE_KERNEL = np.random.default_rng(2).standard_normal((3, 256))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def make_sample_image(height=12, width=12):
    """Create a random RGB image (H, W, 3) as float values."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (height, width, 3)).astype(np.float64)


def load_image(path):
    """Load an image file as a float array of shape (H, W, C)."""
    img = Image.open(path)
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr


def to_uint8(arr):
    """Rescale values to 0-255 so the result can be saved as an image."""
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------

def depthwise_convolution(image, kernel):
    """
    Each channel is convolved with its own kernel. The channel count stays the same.
    image  : (H, W, C)
    kernel : (kh, kw, C)
    return : (H - kh + 1, W - kw + 1, C)   (no padding, stride 1)
    """
    H, W, C = image.shape
    kh, kw, kc = kernel.shape
    if kc != C:
        raise ValueError(f"Depthwise kernel needs {C} channels, got {kc}")
    # windows[i, j, c] is the kh x kw patch of channel c starting at (i, j)
    windows = sliding_window_view(image, (kh, kw), axis=(0, 1))
    return np.einsum("hwcij,ijc->hwc", windows, kernel)


def pointwise_convolution(image, kernel):
    """
    1 x 1 kernels that span all input channels. The channel count changes.
    image  : (H, W, C)
    kernel : (C, F)
    return : (H, W, F)
    """
    H, W, C = image.shape
    if kernel.shape[0] != C:
        raise ValueError(f"Pointwise kernel needs {C} input channels, got {kernel.shape[0]}")
    return np.tensordot(image, kernel, axes=([2], [0]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Depthwise or pointwise convolution")
    parser.add_argument("--mode", choices=["depthwise", "pointwise"], default="depthwise",
                        help="Type of convolution (default: depthwise)")
    parser.add_argument("--image", default=None,
                        help="Input image file (optional; a sample image is used if omitted)")
    parser.add_argument("--output", default="output.png", help="Output image file")
    args = parser.parse_args()

    if args.image:
        image = load_image(args.image)
    else:
        image = make_sample_image()

    if args.mode == "depthwise":
        result = depthwise_convolution(image, DEPTHWISE_KERNEL)
        kernel_shape = DEPTHWISE_KERNEL.shape
    else:
        result = pointwise_convolution(image, POINTWISE_KERNEL)
        kernel_shape = POINTWISE_KERNEL.shape

    # Save as an image (grey-scale for 1 channel, RGB for 3, first channel otherwise)
    C = result.shape[2]
    if C == 3:
        out_img = Image.fromarray(to_uint8(result), mode="RGB")
    else:
        out_img = Image.fromarray(to_uint8(result[:, :, 0]), mode="L")
    out_img.save(args.output)

    print(f"Mode:         {args.mode}")
    print(f"Input shape:  {image.shape}")
    print(f"Kernel shape: {kernel_shape}")
    print(f"Output shape: {result.shape}")
    print(f"Saved to:     {args.output}")


if __name__ == "__main__":
    main()