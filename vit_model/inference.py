# vit_model/inference.py

import cv2
import numpy as np
import os

def vit_denoise(img, save_path=None):
    """
    Vision Transformer (simulated using Non-Local Means)

    Input  : BGR image (OpenCV)
    Output : BGR denoised image
    """

    if img is None:
        raise ValueError("Input image is None")

    # Ensure uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Apply Non-Local Means (color)
    denoised = cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=10,
        hColor=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # Save output if path provided (dataset mode)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, denoised)

    return denoised