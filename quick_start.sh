#!/bin/bash

# Quick Start Script - 快速啟動訓練
# 自動檢查依賴並啟動訓練

echo "================================"
echo "Project Physics-AGI - 快速啟動"
echo "================================"
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $python_version"

# Check if torch is installed
if python -c "import torch" 2>/dev/null; then
    torch_version=$(python -c "import torch; print(torch.__version__)")
    echo "✓ PyTorch 已安裝: $torch_version"
    
    # Check CUDA
    if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "✓ CUDA 可用"
        cuda_version=$(python -c "import torch; print(torch.version.cuda)")
        echo "  CUDA 版本: $cuda_version"
    else
        echo "⚠ CUDA 不可用，將使用 CPU 訓練"
    fi
else
    echo "✗ PyTorch 未安裝"
    echo "請運行: pip install -r requirements.txt"
    exit 1
fi

echo ""

# Ask for training configuration
echo "選擇訓練配置："
echo "1) 快速測試 (10K steps)"
echo "2) 標準訓練 (1M steps)"
echo "3) 自定義步數"
read -p "輸入選項 [1-3]: " choice

case $choice in
    1)
        steps=10000
        echo "選擇: 快速測試模式"
        ;;
    2)
        steps=1000000
        echo "選擇: 標準訓練模式"
        ;;
    3)
        read -p "輸入訓練步數: " steps
        echo "選擇: 自定義 $steps 步"
        ;;
    *)
        echo "無效選項，使用默認 (1M steps)"
        steps=1000000
        ;;
esac

echo ""
echo "================================"
echo "開始訓練"
echo "================================"
echo "總步數: $steps"
echo "配置文件: config.yaml"
echo "日誌目錄: logs/"
echo ""

# Create logs directory
mkdir -p logs

# Start training
python train.py --config config.yaml --steps $steps

echo ""
echo "================================"
echo "訓練完成"
echo "================================"
echo ""
echo "查看結果："
echo "  tensorboard --logdir logs/"
echo ""
echo "評估模型："
echo "  python evaluate.py --config config.yaml --checkpoint logs/physics_agi_v1/checkpoints/checkpoint_*.pt --episodes 10"
