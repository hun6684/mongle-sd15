#!/bin/bash
# RunPod 초기 세팅 - mongle SD 1.5 픽셀아트 파이프라인
# RunPod 터미널에서 실행: bash setup_runpod.sh

set -e

echo "=== mongle SD 1.5 RunPod 세팅 ==="
echo ""

# .env 로드
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
    echo ".env 로드 완료"
fi

HF_REPO="${HF_REPO:-}"
LORA_DIR="models/lora_sd15_v1"
LORA_FILE="$LORA_DIR/pytorch_lora_weights.safetensors"

# ================================================================
# 1. 패키지 설치
# ================================================================
echo "[1/4] 패키지 설치 중..."
pip install -q \
    diffusers==0.31.0 \
    accelerate \
    transformers \
    peft \
    xformers \
    rembg \
    onnxruntime-gpu \
    opencv-python-headless \
    matplotlib \
    huggingface_hub \
    safetensors

# 학습용 추가 패키지
pip install -q bitsandbytes
echo "패키지 설치 완료"
echo ""

# ================================================================
# 2. 디렉토리 생성
# ================================================================
echo "[2/4] 디렉토리 생성 중..."
mkdir -p "$LORA_DIR"
mkdir -p image
mkdir -p data/dataset
mkdir -p outputs/style_test
mkdir -p outputs/grid_test
echo "디렉토리 생성 완료"
echo ""

# ================================================================
# 3. LoRA 가중치 다운로드 (학습 완료 후)
# ================================================================
echo "[3/4] LoRA 가중치 확인 중..."
if [ -f "$LORA_FILE" ]; then
    SIZE=$(du -sh "$LORA_FILE" | cut -f1)
    echo "이미 존재함 ($SIZE) - 스킵"
elif [ -n "$HF_REPO" ] && [ -n "$HF_TOKEN" ]; then
    echo "HuggingFace Hub에서 다운로드 중: $HF_REPO"
    huggingface-cli download \
        "$HF_REPO" \
        pytorch_lora_weights.safetensors \
        --local-dir "$LORA_DIR" \
        --token "$HF_TOKEN"
    echo "LoRA 다운로드 완료"
else
    echo "LoRA 없음 - 스타일 테스트(base model only)로 진행하거나"
    echo "학습 후 $LORA_FILE 에 가중치를 두세요."
fi
echo ""

# ================================================================
# 4. 환경 확인
# ================================================================
echo "[4/4] 환경 확인..."
python3 - <<'PYEOF'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA:    {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  VRAM:    {vram:.1f} GB")
    if vram < 8:
        print("  [경고] VRAM 8GB 미만 - SD 1.5 실행에 부족할 수 있음")

for pkg in ["diffusers", "rembg", "cv2"]:
    try:
        import importlib
        m = importlib.import_module(pkg)
        ver = getattr(m, "__version__", "ok")
        print(f"  {pkg}: {ver}")
    except ImportError:
        print(f"  [오류] {pkg} 설치 실패")
PYEOF

echo ""
echo "=== 세팅 완료 ==="
echo ""
echo "다음 단계:"
echo "  [스타일 테스트]  python test_style.py --input image/01.jpg"
echo "  [LoRA 학습]      bash train_lora_sd15.sh"
echo "  [그리드 테스트]  python test_grid.py --lora-path $LORA_DIR"
echo ""
