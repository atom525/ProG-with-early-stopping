#!/bin/bash
# MUTAG 全流程：预训练 -> 下游任务
# 使用 RecBole 风格配置（config/config.yaml）
set -e
cd "$(dirname "$0")/.."

# 激活 conda 环境（若有）
if command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate Graphpro 2>/dev/null || true
    # 使用 conda 的 libstdc++，避免 GLIBCXX_3.4.29 缺失错误
    export LD_LIBRARY_PATH=${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}
fi

echo "=========================================="
echo "Step 0: 确保 MUTAG 已下载"
echo "=========================================="
python scripts/download_data.py --datasets MUTAG

echo ""
echo "=========================================="
echo "Step 1: 预训练 (GraphMultiGprompt on MUTAG)"
echo "=========================================="
python pre_train.py \
    --pretrain_task GraphMultiGprompt \
    --dataset_name MUTAG \
    --gnn_type GIN \
    --hid_dim 128 \
    --num_layer 2 \
    --epochs 500 \
    --device 0 \
    --config_file config/config.yaml

PRETRAIN_MODEL=$(ls checkpoints/pretrain/MUTAG/MultiGprompt_*.pth 2>/dev/null | head -1)
if [ -z "$PRETRAIN_MODEL" ] || [ ! -f "$PRETRAIN_MODEL" ]; then
    echo "预训练模型未生成，请检查 checkpoints/pretrain/MUTAG/"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: 下游任务 (GraphTask, Gprompt, 5-shot)"
echo "=========================================="
python downstream_task.py \
    --downstream_task GraphTask \
    --pre_train_model_path "$PRETRAIN_MODEL" \
    --dataset_name MUTAG \
    --gnn_type GIN \
    --prompt_type Gprompt \
    --shot_num 5 \
    --split_ratio 0.8 0.1 0.1 \
    --patience 20 \
    --epochs 500 \
    --device 0 \
    --config_file config/config.yaml

echo ""
echo "=========================================="
echo "全流程完成！"
echo "=========================================="
