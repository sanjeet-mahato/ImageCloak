# controllers/cloak_controller.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import lpips  # perceptual similarity

class AdversarialCloaker:
    """
    Generates high-resolution adversarial cloaks that are visually similar
    to the original image using LPIPS perceptual loss and optional TV smoothing.
    """
    def __init__(self, model, class_names, device='cpu', epsilon=0.05, alpha=0.01, steps=20,
                 tv_weight=0.02, lpips_weight=1.0, model_input_size=(32, 32)):
        self.model = model.to(device)
        self.model.eval()
        self.class_names = class_names
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.tv_weight = tv_weight
        self.lpips_weight = lpips_weight
        self.model_input_size = model_input_size

        # Model preprocessing
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # LPIPS model
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
        lpips.LPIPS.verbose = False  # suppress LPIPS messages

        # For converting high-res images to tensors without normalization
        self.to_tensor = transforms.ToTensor()

    def load_image(self, path):
        """Load a high-resolution image as PIL.Image"""
        img = Image.open(path).convert("RGB")
        return img

    def predict(self, img_tensor):
        """Predict class confidences on a tensor"""
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            probs = F.softmax(outputs, dim=1).squeeze(0)
        return {cls: float(probs[idx]) for idx, cls in enumerate(self.class_names)}

    def total_variation(self, x):
        """Total Variation (TV) loss for smoothness"""
        diff_h = torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]))
        diff_w = torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
        return diff_h + diff_w

    def pgd_cloak(self, img_pil, target_class=None):
        """
        Generate a high-resolution adversarial cloak.
        The perturbation is computed on the model-sized image and upsampled to original resolution.
        """
        # --- Prepare tensors ---
        # High-resolution tensor for final cloaked image
        img_highres_tensor = self.to_tensor(img_pil).to(self.device).unsqueeze(0)
        img_highres_tensor.requires_grad = False

        # Resize for model input
        img_small = img_pil.resize(self.model_input_size)
        img_tensor = self.preprocess(img_small).to(self.device).unsqueeze(0)
        img_orig = img_tensor.clone()

        img_tensor.requires_grad = True

        # Determine target class
        if target_class is None:
            with torch.no_grad():
                target_class_idx = self.model(img_tensor).argmax(dim=1).item()
        else:
            target_class_idx = target_class

        # --- PGD Loop ---
        for _ in range(self.steps):
            outputs = self.model(img_tensor)
            loss = F.cross_entropy(outputs, torch.tensor([target_class_idx], device=self.device))
            # LPIPS perceptual loss
            lpips_loss = self.lpips_model(img_tensor, img_orig).mean()
            loss += self.lpips_weight * lpips_loss
            # TV smoothing
            loss += self.tv_weight * self.total_variation(img_tensor)
            self.model.zero_grad()
            loss.backward()
            grad_sign = img_tensor.grad.sign()
            img_tensor = img_tensor + self.alpha * grad_sign
            # Clip to epsilon ball
            img_tensor = torch.max(torch.min(img_tensor, img_orig + self.epsilon), img_orig - self.epsilon)
            img_tensor = torch.clamp(img_tensor, -1, 1)  # normalized range
            img_tensor.detach_()
            img_tensor.requires_grad = True

        # --- Upsample perturbation to high-resolution ---
        perturbation = img_tensor.detach() - img_orig  # small perturbation
        perturbation_highres = F.interpolate(perturbation, size=img_highres_tensor.shape[2:], mode='bilinear', align_corners=False)

        # Apply perturbation to high-res image
        cloaked_highres_tensor = torch.clamp(img_highres_tensor + perturbation_highres, 0, 1)
        cloaked_pil = to_pil_image(cloaked_highres_tensor.squeeze(0).cpu())

        return cloaked_pil
