# views/display_results.py

import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image

def imshow(img, title=None, figsize=(6,6)):
    """
    Display a high-resolution image.
    Accepts:
    - img: PIL.Image or torch.Tensor
    - title: optional title
    - figsize: tuple (width, height) in inches
    """
    # Convert tensor to PIL if needed
    if isinstance(img, torch.Tensor):
        # Detach if requires grad
        if img.requires_grad:
            img = img.detach()
        # If batch dimension exists, remove it
        if img.ndim == 4:
            img = img.squeeze(0)
        # Unnormalize if values in [-1,1]
        if img.min() < 0:
            img = img / 2 + 0.5
        img = to_pil_image(img)

    plt.figure(figsize=figsize)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def show_confidences(confidences):
    """
    Print class probabilities in readable format
    """
    for cls, conf in confidences.items():
        print(f"{cls}: {conf:.4f}")
