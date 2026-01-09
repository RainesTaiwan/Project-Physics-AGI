#!/bin/bash
# System Verification Script
# 系統驗證腳本

echo "=========================================="
echo "🔍 Project Physics-AGI 系統驗證"
echo "=========================================="
echo ""

# Check Python version
echo "📌 檢查 Python 版本..."
python --version
echo ""

# Check PyTorch
echo "📌 檢查 PyTorch..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Check dependencies
echo "📌 檢查核心依賴..."
python -c "
import sys
deps = ['numpy', 'scipy', 'yaml', 'tqdm', 'mujoco', 'dm_control', 'gymnasium']
missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f'✓ {dep}')
    except ImportError:
        print(f'✗ {dep} (缺失)')
        missing.append(dep)
if missing:
    print(f'\n⚠️  缺失依賴: {missing}')
    sys.exit(1)
else:
    print('\n✅ 所有核心依賴已安裝')
"
echo ""

# Check EGL/OpenGL
echo "📌 檢查 OpenGL 支援..."
if ldconfig -p | grep -q libegl; then
    echo "✓ EGL 已安裝"
else
    echo "✗ EGL 未安裝"
fi
echo ""

# Run module tests
echo "📌 運行模組測試..."
./test_modules.sh
echo ""

# Test training script
echo "📌 測試訓練腳本（50 步快速測試）..."
timeout 120 python train.py --config config.yaml --steps 50 > /tmp/train_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ 訓練腳本測試通過"
else
    echo "❌ 訓練腳本測試失敗"
    echo "查看日誌: /tmp/train_test.log"
    tail -20 /tmp/train_test.log
fi
echo ""

echo "=========================================="
echo "✅ 系統驗證完成！"
echo "=========================================="
echo ""
echo "🚀 準備開始訓練："
echo "   python train.py --config config.yaml --steps 10000"
echo ""
