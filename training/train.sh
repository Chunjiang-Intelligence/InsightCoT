#!/bin/bash

# =====================================================================================
# InsightCoT Fine-Tuning Script
#
# This script launches the LoRA fine-tuning process using Hugging Face Accelerate.
# It allows for easy configuration of the model, dataset, and training parameters.
#
# Usage:
# ./train.sh [OPTIONS]
#
# Example:
# ./train.sh --model_name "microsoft/phi-3-mini-4k-instruct" --epochs 2 --batch_size 4
# =====================================================================================

MODEL_NAME="meta-llama/Llama-3-8B-Instruct"
DATASET_PATH="synthetic_data.jsonl"
OUTPUT_DIR_BASE="./results"
NUM_EPOCHS=1
BATCH_SIZE=2
GRAD_ACCUM=2
LEARNING_RATE="2e-4"
SEQ_LENGTH=2048

show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo "-------------------------------------------------------------------------------------"
    echo "Options:"
    echo "  --model_name <name>         Hugging Face model ID."
    echo "                              (default: ${MODEL_NAME})"
    echo "  --dataset <path>            Path to the JSONL dataset file."
    echo "                              (default: ${DATASET_PATH})"
    echo "  --output_dir <path>         Base directory to save training results."
    echo "                              (default: ${OUTPUT_DIR_BASE})"
    echo "  --epochs <num>              Number of training epochs."
    echo "                              (default: ${NUM_EPOCHS})"
    echo "  --batch_size <num>          Per-device training batch size."
    echo "                              (default: ${BATCH_SIZE})"
    echo "  --grad_accum <num>          Gradient accumulation steps."
    echo "                              (default: ${GRAD_ACCUM})"
    echo "  --lr <rate>                 Learning rate."
    echo "                              (default: ${LEARNING_RATE})"
    echo "  --seq_len <num>             Maximum sequence length."
    echo "                              (default: ${SEQ_LENGTH})"
    echo "  -h, --help                  Show this help message and exit."
    echo "-------------------------------------------------------------------------------------"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        --dataset) DATASET_PATH="$2"; shift ;;
        --output_dir) OUTPUT_DIR_BASE="$2"; shift ;;
        --epochs) NUM_EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --grad_accum) GRAD_ACCUM="$2"; shift ;;
        --lr) LEARNING_RATE="$2"; shift ;;
        --seq_len) SEQ_LENGTH="$2"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) echo "Unknown parameter passed: $1"; show_help; exit 1 ;;
    esac
    shift
done

if ! accelerate env > /dev/null 2>&1; then
    echo "Error: Hugging Face Accelerate is not configured."
    echo "Please run 'accelerate config' to set up your environment."
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found at '${DATASET_PATH}'"
    echo "Please generate the dataset first using 'generate_data.py'."
    exit 1
fi

SANITIZED_MODEL_NAME=$(echo "$MODEL_NAME" | tr '/' '_')
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR_BASE}/${SANITIZED_MODEL_NAME}-insightcot-${TIMESTAMP}"

echo "==================================================="
echo "🚀 Starting InsightCoT LoRA Fine-Tuning"
echo "==================================================="
echo "  Model:           ${MODEL_NAME}"
echo "  Dataset:         ${DATASET_PATH}"
echo "  Output Dir:      ${OUTPUT_DIR}"
echo "  Epochs:          ${NUM_EPOCHS}"
echo "  Batch Size:      ${BATCH_SIZE}"
echo "  Grad Accum:      ${GRAD_ACCUM}"
echo "  Learning Rate:   ${LEARNING_RATE}"
echo "  Sequence Length: ${SEQ_LENGTH}"
echo "==================================================="
echo ""

export TOKENIZERS_PARALLELISM=false

accelerate launch train_lora.py \
    --model_name "$MODEL_NAME" \
    --dataset_name "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --max_seq_length "$SEQ_LENGTH"

echo ""
echo "==================================================="
echo "✅ Training Finished!"
echo "LoRA adapters saved in: ${OUTPUT_DIR}"
echo "==================================================="
