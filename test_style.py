"""
Quick style test — runs base SD 1.5 + ControlNet WITHOUT a LoRA.
Use this to verify the SD 1.5 approach produces better pixel art style
before committing to a full retraining run.

Usage:
    python test_style.py --input ../mongle_32bit_window/image/01.jpg
"""
import argparse
from pathlib import Path
from PIL import Image
from pipeline import load_pipeline, DEFAULT_NEGATIVE_PROMPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/style_test")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--controlnet-scale", type=float, default=0.8)
    parser.add_argument("--guidance-scale", type=float, default=9.0)
    parser.add_argument("--lora-path", default=None, help="LoRA 경로. 없으면 base 모델만 사용.")
    parser.add_argument("--no-vlm", action="store_true", help="VLM 없이 실행")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(lora_path=args.lora_path, use_vlm=not args.no_vlm)

    image = Image.open(args.input).convert("RGB")
    stem = Path(args.input).stem

    result = pipe(
        image,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        seed=args.seed,
    )

    suffix = "lora" if args.lora_path else "base"
    result["image"].save(out_dir / f"{stem}_pixel.png")        # 최종 픽셀아트
    result["image_chibi"].save(out_dir / f"{stem}_chibi.png")  # 픽셀화 전 치비
    result["source_image"].save(out_dir / f"{stem}_source.png")
    result["canny_image"].save(out_dir / f"{stem}_canny.png")
    print(f"픽셀아트 → {stem}_pixel.png")
    print(f"치비(전단계) → {stem}_chibi.png")
    print(f"프롬프트: {result.get('prompt_used', 'N/A')}")
    print(f"rembg_ok: {result['rembg_ok']}")


if __name__ == "__main__":
    main()
