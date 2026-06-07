import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from PIL import Image
from rembg import remove


# 스타일 고정 부분 (VLM이 앞부분을 채워줌)
STYLE_SUFFIX = (
    "pixel art character sprite, "
    "32-bit pixel art style, super deformed chibi proportions, "
    "large rounded head, chubby plush body, short stubby limbs, "
    "soft warm color palette, 2-tone shading, "
    "thin dark brown outline, dot eyes with highlight, tiny nose, soft smile, "
    "pink blush cheeks, white background, full body, front-facing, standing"
)

DEFAULT_NEGATIVE_PROMPT = (
    "realistic, 3d render, blurry, photograph, smooth illustration, "
    "painterly, watercolor, sketch, sitting, side view, angled, "
    "pure black outline, deformed, extra limbs, ugly, duplicate"
)

VLM_QUESTION = (
    "This is a stuffed animal toy. "
    "In one short sentence, describe: what animal it is and its main color. "
    "Example: 'a pink bear with a cream belly'"
)


class VLMDescriptor:
    """Moondream2로 인형 사진 특징 추출"""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("VLM(Moondream2) 로딩 중...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
        self.model.eval()
        print("VLM 로딩 완료")

    def describe(self, image: Image.Image) -> str:
        enc = self.model.encode_image(image)
        result = self.model.answer_question(enc, VLM_QUESTION, self.tokenizer)
        return result.strip().rstrip(".")


class Mongle32BitPipeline:
    def __init__(
        self,
        lora_path: Optional[str] = None,
        base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        controlnet_model: str = "lllyasviel/sd-controlnet-canny",
        use_vlm: bool = True,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device in ("cuda", "mps") else torch.float32)

        # VLM 로드 (선택)
        self.vlm = VLMDescriptor(device=self.device) if use_vlm else None

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=self.dtype,
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
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
            print("No LoRA found — base model only")

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
        source = bg_removed if self.rembg_succeeded(bg_removed, bg_min_ratio) else original
        rembg_ok = source is bg_removed
        canny = self.extract_canny(source, canny_low, canny_high)
        return source, canny, rembg_ok

    @staticmethod
    def to_pixel_art(
        image: Image.Image,
        pixel_size: int = 96,
        n_colors: int = 20,
    ) -> Image.Image:
        """
        치비 이미지 → 픽셀아트 변환
        pixel_size: 중간 해상도 (낮을수록 픽셀이 굵어짐, 권장 64~128)
        n_colors: 색상 팔레트 수 (낮을수록 단순, 권장 16~32)
        """
        w, h = image.size

        # 핵심: 색상 팔레트 제한 (이게 없으면 그냥 블러처럼 보임)
        quantized = image.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
        rgb = quantized.convert("RGB")

        # 다운스케일 → 업스케일 (NEAREST = 픽셀 블록감)
        small = rgb.resize((pixel_size, pixel_size), Image.NEAREST)
        result = small.resize((w, h), Image.NEAREST)

        return result

    def build_prompt(self, source: Image.Image) -> str:
        if self.vlm is not None:
            description = self.vlm.describe(source)
            print(f"VLM 설명: {description}")
            return f"{description}, {STYLE_SUFFIX}"
        return f"a cute stuffed animal toy, {STYLE_SUFFIX}"

    def __call__(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
        controlnet_conditioning_scale: float = 0.8,
        seed: Optional[int] = None,
    ) -> dict:
        source, canny, rembg_ok = self.prepare_image(image)

        final_prompt = prompt if prompt else self.build_prompt(source)
        print(f"최종 프롬프트: {final_prompt}")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            image=canny,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            height=512,
            width=512,
        ).images[0]

        result_cleaned = self.remove_background(result)

        # 픽셀아트 후처리: 색상 팔레트 제한 + 픽셀화
        pixel_art = self.to_pixel_art(result_cleaned)

        return {
            "image": pixel_art,
            "image_chibi": result_cleaned,
            "image_raw": result,
            "source_image": source,
            "canny_image": canny,
            "prompt_used": final_prompt,
            "rembg_ok": rembg_ok,
        }


def load_pipeline(lora_path: Optional[str] = None, use_vlm: bool = True) -> Mongle32BitPipeline:
    return Mongle32BitPipeline(
        lora_path=lora_path or os.getenv("LORA_PATH"),
        use_vlm=use_vlm,
    )
