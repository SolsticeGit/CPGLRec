#!/bin/bash

DATASETS=("Beauty" "CDs_and_Vinyl")

ALPHA_VALUES=(0.1 0.3 0.5 0.7 0.9)

GAMMA=1.0
ALPHA_WEIGHT=0.6

DEVICE="cuda"

TOTAL_TASKS=$((${#DATASETS[@]} * ${#ALPHA_VALUES[@]}))
CURRENT_TASK=0

TOTAL_TASKS_STR="$TOTAL_TASKS"
CURRENT_TASK_STR="$CURRENT_TASK"
DEVICE_STR="$DEVICE"
GAMMA_STR="$GAMMA"
ALPHA_WEIGHT_STR="$ALPHA_WEIGHT"
DATASETS_STR="${DATASETS[*]}"
ALPHA_VALUES_STR="${ALPHA_VALUES[*]}"

TOTAL_TASKS_STR="$TOTAL_TASKS_STR"
CURRENT_TASK_STR="$CURRENT_TASK_STR"
DEVICE_STR="$DEVICE_STR"
GAMMA_STR="$GAMMA_STR"
ALPHA_WEIGHT_STR="$ALPHA_WEIGHT_STR"
DATASETS_STR="$DATASETS_STR"
ALPHA_VALUES_STR="$ALPHA_VALUES_STR"

for DATASET in "${DATASETS[@]}"; do
    EMBEDDINGS_FILE="mydata/$DATASET/process_data/item_embeddings.pt"
    POPULARITY_FILE="mydata/$DATASET/process_data/popularity.json"
    TRAIN_FILE="mydata/$DATASET/process_data/train.json"

    if [ ! -f "$EMBEDDINGS_FILE" ]; then
        CURRENT_TASK=$((CURRENT_TASK + ${#ALPHA_VALUES[@]}))
        continue
    fi

    if [ ! -f "$POPULARITY_FILE" ] || [ ! -f "$TRAIN_FILE" ]; then
        CURRENT_TASK=$((CURRENT_TASK + ${#ALPHA_VALUES[@]}))
        continue
    fi

    for ALPHA in "${ALPHA_VALUES[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))

        ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA")
        ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$ALPHA_WEIGHT")

        P_TARGET_FILE="mydata/$DATASET/process_data/p_target_alpha_${ALPHA_FORMATTED}.json"
        if [ ! -f "$P_TARGET_FILE" ]; then
            continue
        fi

        OUTPUT_FILE="dpo_data/$DATASET/dpo_pairs_alpha_${ALPHA_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}.jsonl"
        if [ -f "$OUTPUT_FILE" ]; then
            continue
        fi

        python dpo_pairs.py \
            --dataset "$DATASET" \
            --alpha "$ALPHA" \
            --gamma "$GAMMA" \
            --alpha_weight "$ALPHA_WEIGHT" \
            --device "$DEVICE"

        if [ $? -ne 0 ]; then
            :
        fi
    done
done

for DATASET in "${DATASETS[@]}"; do
    for ALPHA in "${ALPHA_VALUES[@]}"; do
        ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA")
        ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$ALPHA_WEIGHT")
        OUTPUT_FILE="dpo_data/$DATASET/dpo_pairs_alpha_${ALPHA_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}.jsonl"
        if [ -f "$OUTPUT_FILE" ]; then
            :
        else
            :
        fi
    done
done
