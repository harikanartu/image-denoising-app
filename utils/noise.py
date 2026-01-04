import cv2
import os
import numpy as np

CLEAN_DIR = "dataset/clean"
NOISY_DIR = "dataset/noisy"

os.makedirs(NOISY_DIR, exist_ok=True)

def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

for img_name in os.listdir(CLEAN_DIR):
    img_path = os.path.join(CLEAN_DIR, img_name)
    img = cv2.imread(img_path)

    if img is None:
        continue

    noisy_img = add_gaussian_noise(img)
    cv2.imwrite(os.path.join(NOISY_DIR, img_name), noisy_img)

print("✅ Noisy images generated successfully")