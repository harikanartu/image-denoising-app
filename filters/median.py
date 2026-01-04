import cv2
import os

# Input and output directories
INPUT_DIR = "dataset/noisy"
OUTPUT_DIR = "outputs/filter_results/median"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_median(img, kernel_size=5):
    """
    Apply Median filter to an image

    Parameters:
    img : Grayscale image
    kernel_size : Size of the median filter window (must be odd)

    Returns:
    Denoised image
    """
    return cv2.medianBlur(img, kernel_size)

def process_dataset():
    """
    Apply Median filter to all images in dataset/noisy
    """
    for img_name in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        denoised = apply_median(img)

        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, denoised)

    print("✅ Median filtering completed")

if __name__ == "__main__":
    process_dataset()