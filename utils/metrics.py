# metrics.py

import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

CLEAN_DIR = "dataset/clean"

METHODS = {
    "Gaussian Filter": "outputs/filter_results/gaussian",
    "Bilateral Filter": "outputs/filter_results/bilateral",
    "Median Filter": "outputs/filter_results/median",
    "Vision Transformer (NLM)": "outputs/vit_results"
}


def compute_metrics(clean_img, denoised_img):
    """
    Computes PSNR & SSIM for COLOR images
    """

    clean = clean_img.astype(np.float32)
    den = denoised_img.astype(np.float32)

    psnr = peak_signal_noise_ratio(clean, den, data_range=255)

    ssim = structural_similarity(
        clean,
        den,
        channel_axis=-1,
        data_range=255
    )

    return psnr, ssim


for method, result_dir in METHODS.items():
    psnr_list = []
    ssim_list = []

    for img_name in os.listdir(CLEAN_DIR):
        clean_path = os.path.join(CLEAN_DIR, img_name)
        denoised_path = os.path.join(result_dir, img_name)

        if not os.path.exists(denoised_path):
            continue

        clean = cv2.imread(clean_path)
        denoised = cv2.imread(denoised_path)

        if clean is None or denoised is None:
            continue

        # Resize safety
        if clean.shape != denoised.shape:
            denoised = cv2.resize(
                denoised,
                (clean.shape[1], clean.shape[0])
            )

        psnr, ssim = compute_metrics(clean, denoised)

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print(f"\n📊 {method}")
    if psnr_list:
        print(f"Average PSNR : {sum(psnr_list)/len(psnr_list):.2f} dB")
        print(f"Average SSIM : {sum(ssim_list)/len(ssim_list):.4f}")
    else:
        print("⚠️ No valid images found")