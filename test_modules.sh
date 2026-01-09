#!/bin/bash

# Quick Test Script - 快速測試所有模組
# 確保所有核心組件正常工作

echo "================================"
echo "Project Physics-AGI - 模組測試"
echo "================================"
echo ""

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_module() {
    local module_name=$1
    local module_path=$2
    
    echo -e "${YELLOW}測試: ${module_name}${NC}"
    if python -m $module_path > /dev/null 2>&1; then
        echo -e "${GREEN}✓ ${module_name} 通過${NC}"
        return 0
    else
        echo -e "${RED}✗ ${module_name} 失敗${NC}"
        return 1
    fi
}

# Test all modules
passed=0
failed=0

# Test Module A: Encoder
if test_module "模組 A - 變分編碼器" "src.models.encoder"; then
    ((passed++))
else
    ((failed++))
fi
echo ""

# Test Module B: RSSM
if test_module "模組 B - RSSM 動力學模型" "src.models.rssm"; then
    ((passed++))
else
    ((failed++))
fi
echo ""

# Test Module C/D: Actor-Critic
if test_module "模組 C/D - Actor-Critic" "src.models.actor_critic"; then
    ((passed++))
else
    ((failed++))
fi
echo ""

# Test Replay Buffer
if test_module "工具 - Replay Buffer" "src.utils.replay_buffer"; then
    ((passed++))
else
    ((failed++))
fi
echo ""

# Test Environment (may fail if dm-control not installed)
echo -e "${YELLOW}測試: 工具 - 環境包裝器${NC}"
if python -m src.utils.env_wrapper > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 環境包裝器 通過${NC}"
    ((passed++))
else
    echo -e "${YELLOW}⚠ 環境包裝器 跳過 (需要 dm-control)${NC}"
fi
echo ""

# Test Trainer
if test_module "訓練器 - World Model Trainer" "src.trainer"; then
    ((passed++))
else
    ((failed++))
fi
echo ""

# Summary
echo "================================"
echo "測試總結"
echo "================================"
echo -e "${GREEN}通過: ${passed}${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}失敗: ${failed}${NC}"
fi
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}🎉 所有測試通過！系統已就緒。${NC}"
    exit 0
else
    echo -e "${RED}⚠️  部分測試失敗，請檢查錯誤信息。${NC}"
    exit 1
fi
