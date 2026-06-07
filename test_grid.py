import argparse
import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from pipeline import DEFAULT_NEGATIVE_PROMPT, DEFAULT_PROMPT, load_pipeline


PROMPT_PRESETS = {
    "default": DEFAULT_PROMPT,
    "soft": (
        "a stuffed animal toy transformed into a cute 32-bit pixel art character sprite, "
        "modern Korean mobile RPG style, soft rounded chibi proportions, "
        "large rounded head, plush body, tiny limbs, warm pastel colors, "
        "gentle 2-tone shading, thin dark brown outline, shiny dot eyes, "
        "tiny nose, soft smile, pink blush cheeks, white background, full body, front-facing"
    ),
    "sprite": (
        "front-facing full body stuffed animal character sprite, "
        "high quality 32-bit pixel art, cozy mobile RPG game asset, "
        "clean pixel edges, readable silhouette, compact chibi body, "
        "soft fur shading, warm colors, subtle highlights, thin brown outline, "
        "cute face, dot eyes with highlights, pink blush, white background"
    ),
    "reference": (
        "cute plush toy, high resolution pixel art character, large visible pixels, "
        "rounded chibi body, front-facing full body, warm soft shading, "
        "brown pixel outline, glossy dot eyes, tiny nose, small smile, "
        "pink cheek blush, clean white background"
    ),
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def collect_images(input_path: str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path]

    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in path.iterdir() if p.suffix.lower() in extensions])


def save_grid(items: list[tuple[str, Image.Image]], output_path: Path, columns: int = 4):
    if not items:
        return

    rows = (len(items) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))

    if rows == 1:
        axes = [axes] if columns == 1 else axes
    else:
        axes = [ax for row in axes for ax in row]

    for ax, (title, image) in zip(axes, items):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for ax in axes[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run a parameter grid for Mongle 32-bit LoRA.")
    parser.add_argument("--input", default="image", help="Input image file or folder.")
    parser.add_argument("--output", default="outputs/grid_test", help="Output folder.")
    parser.add_argument("--lora-path", default=os.getenv("LORA_PATH", "."), help="LoRA repo/path.")
    parser.add_argument("--strengths", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--controlnet-scales", default="0.6,0.8,1.0")
    parser.add_argument("--guidance-scales", default="7.5")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--prompt-presets", default="default,soft,sprite")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of input images. 0 means all.")
    parser.add_argument("--save-individual", action="store_true", help="Also save every generated image.")
    args = parser.parse_args()

    images = collect_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No input images found: {args.input}")

    strengths = parse_csv_floats(args.strengths)
    controlnet_scales = parse_csv_floats(args.controlnet_scales)
    guidance_scales = parse_csv_floats(args.guidance_scales)
    prompt_keys = [key.strip() for key in args.prompt_presets.split(",") if key.strip()]

    unknown = [key for key in prompt_keys if key not in PROMPT_PRESETS]
    if unknown:
        raise SystemExit(f"Unknown prompt preset(s): {', '.join(unknown)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args.lora_path)

    for image_path in images:
        original = Image.open(image_path).convert("RGB")
        stem = image_path.stem
        image_output_dir = output_dir / stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        grid_items = [("input", original.resize((512, 512)))]

        for prompt_key, strength, controlnet_scale, guidance_scale in product(
            prompt_keys,
            strengths,
            controlnet_scales,
            guidance_scales,
        ):
            result = pipe(
                original,
                prompt=PROMPT_PRESETS[prompt_key],
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                num_inference_steps=args.steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                strength=strength,
                seed=args.seed,
            )

            filename = (
                f"{stem}_{prompt_key}"
                f"_st{strength:.2f}_cn{controlnet_scale:.2f}_g{guidance_scale:.1f}.png"
            )
            output_path = image_output_dir / filename
            if args.save_individual:
                result["image"].save(output_path)
                print(f"saved {output_path}")

            title = f"{prompt_key}\nst={strength:.2f} cn={controlnet_scale:.2f} g={guidance_scale:.1f}"
            grid_items.append((title, result["image"]))

        grid_path = image_output_dir / f"{stem}_grid.png"
        save_grid(grid_items, grid_path, columns=4)
        print(f"saved grid {grid_path}")

    print(f"done: {output_dir}")


if __name__ == "__main__":
    main()
