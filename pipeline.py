import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from PIL import Image
from rembg import remove


DEFAULT_PROMPT = (
    "a stuffed animal toy pixel art character sprite, "
    "32-bit pixel art style, super deformed chibi proportions, "
    "large rounded head, chubby plush body, short stubby limbs, "
    "soft warm color palette, 2-tone shading, "
    "thin dark brown outline, dot eyes with highlight, tiny nose, soft smile, "
    "pink blush cheeks, white background, full body, front-facing, standing"
)

DEFAULT_NEGATIVE_PROMPT = (
    "realistic, 3d render, blurry, photograph, smooth illustration, "
    "painterly, watercolor, sketch, sitting, side view, angled, "
    "pure black outline, deformed, extra limbs"
)


class Mongle32BitPipeline:
    def __init__(
        self,
        lora_path: Optional[str] = None,
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet_model: str = "lllyasviel/sd-controlnet-canny",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device in ("cuda", "mps") else torch.float32)

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=self.dtype,
        )

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=self.dtype,
            safety_checker=None,
        )

        if lora_path and os.path.isfile(os.path.join(lora_path, "pytorch_lora_weights.safetensors")):
            self.pipe.load_lora_weights(
                lora_path,
                weight_name="pytorch_lora_weights.safetensors",
            )
            print(f"LoRA loaded from {lora_path}")
        else:
            print("No LoRA found — running base model only (style test mode)")

        self.pipe.to(self.device)
        self.pipe.enable_attention_slicing()

    @staticmethod
    def remove_background(image: Image.Image) -> Image.Image:
        removed = remove(image.convert("RGBA"))
        white_bg = Image.new("RGBA", removed.size, (255, 255, 255, 255))
        white_bg.paste(removed, mask=removed.split()[3])
        return white_bg.convert("RGB")

    @staticmethod
    def rembg_succeeded(image: Image.Image, threshold: float = 0.40) -> bool:
        arr = np.array(image.convert("RGB"))
        bg_mask = (arr[:, :, 0] >= 245) & (arr[:, :, 1] >= 245) & (arr[:, :, 2] >= 245)
        return float(bg_mask.sum()) / bg_mask.size >= threshold

    @staticmethod
    def extract_canny(image: Image.Image, low: int = 80, high: int = 180) -> Image.Image:
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        return Image.fromarray(np.stack([edges] * 3, axis=-1))

    def prepare_image(
        self,
        image: Image.Image,
        size: Tuple[int, int] = (512, 512),
        bg_min_ratio: float = 0.40,
        canny_low: int = 80,
        canny_high: int = 180,
    ) -> Tuple[Image.Image, Image.Image, bool]:
        original = image.convert("RGB").resize(size)
        bg_removed = self.remove_background(original)

        if self.rembg_succeeded(bg_removed, bg_min_ratio):
            source = bg_removed
            rembg_ok = True
        else:
            source = original
            rembg_ok = False

        canny = self.extract_canny(source, canny_low, canny_high)
        return source, canny, rembg_ok

    def __call__(
        self,
        image: Image.Image,
        prompt: str = DEFAULT_PROMPT,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 0.8,
        strength: float = 0.6,
        seed: Optional[int] = None,
    ) -> dict:
        source, canny, rembg_ok = self.prepare_image(image)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            control_image=canny,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            strength=strength,
            generator=generator,
        ).images[0]

        # 생성된 output에 rembg 재적용 → 배경 날아다니는 픽셀 제거
        result_cleaned = self.remove_background(result)

        return {
            "image": result_cleaned,
            "image_raw": result,
            "source_image": source,
            "canny_image": canny,
            "rembg_ok": rembg_ok,
        }


def load_pipeline(lora_path: Optional[str] = None) -> Mongle32BitPipeline:
    return Mongle32BitPipeline(lora_path=lora_path or os.getenv("LORA_PATH"))
