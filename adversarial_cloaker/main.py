# main.py

import torch
import warnings
import os
from PIL import Image
from controllers.cloak_controller import AdversarialCloaker
from models.cnn_model import CNN
from views.display_results import show_confidences
from torchvision.transforms.functional import to_pil_image
import matplotlib

matplotlib.use("TkAgg")  # ensures interactive plotting on Mac
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

# Suppress warnings
warnings.filterwarnings("ignore")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_trained_model(path='models/trained_cnn.pth'):
    model = CNN(num_classes=3)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Loaded trained model from {path}")
    else:
        print(f"No trained model found at {path}. Using untrained model.")
    model.to(device)
    model.eval()
    return model


def show_side_by_side(orig_img, cloaked_img, conf_orig=None, conf_cloak=None,
                      orig_title='Original', cloaked_title='Cloaked'):
    """
    Display original and cloaked images side by side with prediction confidences below.
    """
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[8, 1])  # first row images, second row texts

    # --- Original Image ---
    ax_img1 = fig.add_subplot(gs[0, 0])
    if isinstance(orig_img, torch.Tensor):
        orig_img = orig_img.detach().cpu()
        if orig_img.ndim == 4:
            orig_img = orig_img.squeeze(0)
        if orig_img.min() < 0:
            orig_img = orig_img / 2 + 0.5
        orig_img = to_pil_image(orig_img)
    ax_img1.imshow(orig_img)
    ax_img1.axis('off')
    ax_img1.set_title(orig_title)

    # Prediction text below original image
    ax_text1 = fig.add_subplot(gs[1, 0])
    ax_text1.axis('off')
    if conf_orig:
        pred_text = " | ".join([f"{cls}: {conf * 100:.1f}%" for cls, conf in conf_orig.items()])
        ax_text1.text(0.5, 0.5, pred_text, ha='center', va='center', fontsize=12, color='red')

    # --- Cloaked Image ---
    ax_img2 = fig.add_subplot(gs[0, 1])
    if isinstance(cloaked_img, torch.Tensor):
        cloaked_img = cloaked_img.detach().cpu()
        if cloaked_img.ndim == 4:
            cloaked_img = cloaked_img.squeeze(0)
        if cloaked_img.min() < 0:
            cloaked_img = cloaked_img / 2 + 0.5
        cloaked_img = to_pil_image(cloaked_img)
    ax_img2.imshow(cloaked_img)
    ax_img2.axis('off')
    ax_img2.set_title(cloaked_title)

    # Prediction text below cloaked image
    ax_text2 = fig.add_subplot(gs[1, 1])
    ax_text2.axis('off')
    if conf_cloak:
        pred_text = " | ".join([f"{cls}: {conf * 100:.1f}%" for cls, conf in conf_cloak.items()])
        ax_text2.text(0.5, 0.5, pred_text, ha='center', va='center', fontsize=12, color='red')

    plt.tight_layout()
    plt.show(block=True)


def predict_and_cloak_custom_images(model, image_dir='custom_images'):
    """
    Run predictions and generate adversarial cloaks for all images in custom_images folder,
    including images inside subdirectories.
    """
    os.makedirs(image_dir, exist_ok=True)  # ensure folder exists
    class_names = ['airplane', 'automobile', 'bird']
    cloaker = AdversarialCloaker(model=model, class_names=class_names, device=device)

    for root, dirs, files in os.walk(image_dir):
        for file_name in sorted(files):
            if not file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                continue

            img_path = os.path.join(root, file_name)
            highres_img = Image.open(img_path).convert("RGB")

            # Predict original
            orig_tensor = cloaker.preprocess(highres_img)
            conf_before = cloaker.predict(orig_tensor)
            print(f"\nOriginal Image: {file_name}")
            show_confidences(conf_before)

            # Generate cloaked image
            cloaked_img = cloaker.pgd_cloak(highres_img)
            cloaked_tensor = cloaker.preprocess(cloaked_img)
            conf_after = cloaker.predict(cloaked_tensor)
            print(f"Cloaked Image: {file_name}")
            show_confidences(conf_after)

            # Display side by side with confidences
            show_side_by_side(
                highres_img,
                cloaked_img,
                conf_orig=conf_before,
                conf_cloak=conf_after,
                orig_title=f"Original: {file_name}",
                cloaked_title=f"Cloaked: {file_name}"
            )


def main():
    model = load_trained_model()
    print("\nRunning predictions and cloaking on custom images...")
    predict_and_cloak_custom_images(model)


if __name__ == '__main__':
    main()
