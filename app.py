from flask import Flask, render_template, request
import cv2
import os
import time

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Filters
from filters.gaussian import apply_gaussian
from filters.median import apply_median
from filters.Bilateral import apply_bilateral

# ViT
from vit_model.inference import vit_denoise

app = Flask(__name__)

UPLOAD_PATH = "static/input.jpg"
OUTPUT_PATH = "static/output.jpg"
HISTORY_DIR = "static/history"

os.makedirs("static", exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

# Store history in memory
history = []


def compute_metrics(img1, img2):
    psnr = peak_signal_noise_ratio(img1, img2, data_range=255)
    ssim = structural_similarity(
        img1, img2,
        channel_axis=-1,
        data_range=255
    )
    return psnr, ssim


@app.route("/", methods=["GET", "POST"])
def index():
    global history

    if request.method == "POST":
        image_file = request.files["image"]
        method = request.form["method"]

        image_file.save(UPLOAD_PATH)
        img_color = cv2.imread(UPLOAD_PATH)

        # Apply method
        if method == "gaussian":
            output = apply_gaussian(img_color)
            method_name = "Gaussian Filter"

        elif method == "median":
            output = apply_median(img_color)
            method_name = "Median Filter"

        elif method == "bilateral":
            output = apply_bilateral(img_color)
            method_name = "Bilateral Filter"

        elif method == "vit":
            output = vit_denoise(img_color)
            method_name = "Vision Transformer"

        # Save current output
        cv2.imwrite(OUTPUT_PATH, output)

        # Save history image
        timestamp = int(time.time())
        history_img_path = f"{HISTORY_DIR}/output_{timestamp}.jpg"
        cv2.imwrite(history_img_path, output)

        # Metrics (only output metrics)
        psnr, ssim = compute_metrics(img_color, output)

        # Store history (latest first)
        history.insert(0, {
            "image": history_img_path,
            "method": method_name,
            "psnr": f"{psnr:.2f}",
            "ssim": f"{ssim:.4f}"
        })

        # Limit history to last 5 results
        history = history[:5]

        return render_template(
            "index.html",
            show=True,
            output_psnr=f"{psnr:.2f}",
            output_ssim=f"{ssim:.4f}",
            history=history
        )

    return render_template("index.html", show=False, history=history)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)