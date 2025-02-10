from .compute_score import compute
import torch
from PIL import Image
import numpy as np

def pil_image_from_pixels(pixels: torch.Tensor):
    img = 255. * pixels.cpu().numpy()
    image = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return image

class ComputeScoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE", {}),
                "image2": ("IMAGE", {}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT")
    FUNCTION = "compute"
    CATEGORY = "loaders"

    