import cv2
import os

# Input and output directories
INPUT_DIR = "dataset/noisy"
OUTPUT_DIR = "outputs/filter_results/gaussian"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_gaussian(img, kernel_size=(5, 5), sigma=0):
    """
    Apply Gaussian filter to an image

    Parameters:
    img : Grayscale image
    kernel_size : Size of Gaussian kernel
    sigma : Standard deviation (0 = auto)

    Returns:
    Denoised image
    """
    return cv2.GaussianBlur(img, kernel_size, sigma)

def process_dataset():
    """
    Apply Gaussian filter to all images in dataset/noisy
    """
    for img_name in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        denoised = apply_gaussian(img)

        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, denoised)

    print("✅ Gaussian filtering completed")

if __name__ == "__main__":
    process_dataset()