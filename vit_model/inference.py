import torch
import timm
import cv2
import numpy as np
from torchvision import transforms

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained Vision Transformer (feature extractor)
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=True,
    num_classes=0   # no classification head
)

model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def vit_denoise(img):
    """
    Apply Vision Transformer based enhancement to image

    Parameters:
    img : BGR image (OpenCV format)

    Returns:
    output : BGR image
    """

    # Save original size
    h, w = img.shape[:2]

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess
    x = transform(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(x)

    # Simple enhancement using global feature strength
    scale = torch.sigmoid(features.mean())
    enhanced = x * scale

    # Convert back to image
    out = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # De-normalize
    out = out * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    out = (out * 255).clip(0, 255).astype(np.uint8)

    # Resize back to original size
    out = cv2.resize(out, (w, h))

    # Convert RGB → BGR
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    return out