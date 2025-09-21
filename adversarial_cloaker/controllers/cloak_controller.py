# controllers/cloak_controller.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import lpips  # perceptual similarity

class AdversarialCloaker:
    """
    Generates visually imperceptible adversarial cloaks using LPIPS + optional TV smoothing.
    Works with high-resolution images.
    """
    def __init__(self, model, class_names, device='cpu', epsilon=0.05, alpha=0.015, steps=30,
                 tv_weight=0.02, lpips_weight=1.0):
        self.model = model.to(device)
        self.model.eval()
        self.class_names = class_names
        self.device = device
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.tv_weight = tv_weight
        self.lpips_weight = lpips_weight

        # Preprocess for model input
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # LPIPS model for perceptual similarity
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        self.lpips_model.eval()
        lpips.LPIPS.verbose = False  # suppress LPIPS messages

    def load_image(self, path):
        """Load high-resolution image as PIL.Image"""
        img = Image.open(path).convert("RGB")
        self.original_size = img.size
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
        """Generate LPIPS-guided PGD adversarial cloak"""
        # Preprocess for model
        model_input_size = self.model_input_size()
        img_small = img_pil.resize(model_input_size)
        img_tensor = self.preprocess(img_small).to(self.device)
        img_orig = img_tensor.clone()
        img_tensor.requires_grad = True

        # Determine target class
        if target_class is None:
            with torch.no_grad():
                target_class_idx = self.model(img_tensor.unsqueeze(0)).argmax(dim=1).item()
        else:
            target_class_idx = target_class

        # PGD loop
        for _ in range(self.steps):
            outputs = self.model(img_tensor.unsqueeze(0))
            loss = F.cross_entropy(outputs, torch.tensor([target_class_idx]).to(self.device))

            # LPIPS perceptual loss
            lpips_loss = self.lpips_model(img_tensor.unsqueeze(0), img_orig.unsqueeze(0)).mean()
            loss += self.lpips_weight * lpips_loss

            # Optional TV smoothness
            loss += self.tv_weight * self.total_variation(img_tensor.unsqueeze(0))

            self.model.zero_grad()
            loss.backward()
            grad_sign = img_tensor.grad.sign()
            img_tensor = img_tensor + self.alpha * grad_sign

            # Clip epsilon-ball and clamp
            img_tensor = torch.max(torch.min(img_tensor, img_orig + self.epsilon), img_orig - self.epsilon)
            img_tensor = torch.clamp(img_tensor, 0, 1)

            img_tensor.detach_()
            img_tensor.requires_grad = True

        # Resize to original high-resolution and convert to PIL
        cloaked_tensor = transforms.functional.resize(img_tensor.detach(), img_pil.size[::-1])
        cloaked_pil = to_pil_image(cloaked_tensor)
        return cloaked_pil

    def model_input_size(self):
        """Return model input size (H, W) for prediction"""
        return (32, 32)  # match your trained model
