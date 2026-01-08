import os
import cv2
from inference import vit_denoise

NOISY_DIR = "dataset/noisy"
OUT_DIR = "outputs/vit_results"

# 🔹 THIS LINE CREATES THE FOLDER
os.makedirs(OUT_DIR, exist_ok=True)

for img_name in os.listdir(NOISY_DIR):
    noisy_path = os.path.join(NOISY_DIR, img_name)

    img = cv2.imread(noisy_path)
    if img is None:
        continue

    denoised = vit_denoise(img)

    out_path = os.path.join(OUT_DIR, img_name)
    cv2.imwrite(out_path, denoised)

print("✅ outputs/vit_results created and filled")