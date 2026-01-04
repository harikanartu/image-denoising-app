import os
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

CLEAN_DIR = "dataset/clean"

METHODS = {
    "Gaussian Filter": "outputs/filter_results/gaussian",
    "Bilateral Filter": "outputs/filter_results/bilateral",
    "Median Filter": "outputs/filter_results/median",
    "Vision Transformer": "outputs/vit_results"
}

def compute_metrics(clean_img, denoised_img):
    psnr = peak_signal_noise_ratio(clean_img, denoised_img, data_range=255)
    ssim = structural_similarity(clean_img, denoised_img, data_range=255)
    return psnr, ssim


for method, result_dir in METHODS.items():
    psnr_list = []
    ssim_list = []

    for img_name in os.listdir(CLEAN_DIR):
        clean_path = os.path.join(CLEAN_DIR, img_name)
        denoised_path = os.path.join(result_dir, img_name)

        if not os.path.exists(denoised_path):
            continue

        clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        denoised = cv2.imread(denoised_path, cv2.IMREAD_GRAYSCALE)

        if clean is None or denoised is None:
            continue

        psnr, ssim = compute_metrics(clean, denoised)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    if len(psnr_list) > 0:
        print(f"\n📊 {method}")
        print(f"Average PSNR : {sum(psnr_list)/len(psnr_list):.2f} dB")
        print(f"Average SSIM : {sum(ssim_list)/len(ssim_list):.4f}")
    else:
        print(f"\n⚠️ No valid images found for {method}")