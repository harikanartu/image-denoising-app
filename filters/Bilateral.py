import cv2
import os

# Input and output directories
INPUT_DIR = "dataset/noisy"
OUTPUT_DIR = "outputs/filter_results/bilateral"

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_bilateral(img, diameter=9, sigma_color=75, sigma_space=75):
    """
    Apply Bilateral filter to an image

    Parameters:
    img : Grayscale image
    diameter : Diameter of pixel neighborhood
    sigma_color : Filter sigma in color space
    sigma_space : Filter sigma in coordinate space

    Returns:
    Edge-preserving denoised image
    """
    return cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)

def process_dataset():
    """
    Apply Bilateral filter to all images in dataset/noisy
    """
    for img_name in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        denoised = apply_bilateral(img)

        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, denoised)

    print("✅ Bilateral filtering completed")

if __name__ == "__main__":
    process_dataset()