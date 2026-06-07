#!/bin/bash
# SD 1.5 DreamBooth LoRA training script
# Run this on RunPod or Windows RTX 3060

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --instance_data_dir="data/dataset" \
  --output_dir="models/lora_sd15_v1" \
  --instance_prompt="a mongle pixel art character sprite" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --max_train_steps=2000 \
  --rank=32 \
  --mixed_precision="fp16" \
  --checkpointing_steps=500 \
  --seed=42
