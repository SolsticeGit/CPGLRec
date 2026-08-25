#!/bin/bash

RUN_STAGE1=false
RUN_STAGE2=true
RUN_STAGE3=false

DATASETS=("Beauty" "CDs_and_Vinyl")

LOOP_PARAM="alpha"

ALPHA_VALUES=(0.1 0.3 0.5 0.7 0.9)
ALPHA=0.3

ALPHA_WEIGHT_VALUES=(0.2 0.4 0.6 0.8 1.0)
ALPHA_WEIGHT=0.6

MU=0.3

BASE_MODEL="/home/power/MyLLMs/Qwen2.5-1.5B-Instruct"
USE_4BIT=false

NUM_BEAMS=20
BATCH_SIZE=128
MICRO_BATCH_SIZE=4
MAX_NEW_TOKENS=64
TEMPERATURE=1.0

if [ "$LOOP_PARAM" = "alpha" ]; then
    LOOP_VALUES=("${ALPHA_VALUES[@]}")
    FIXED_PARAM="alpha_weight"
elif [ "$LOOP_PARAM" = "alpha_weight" ]; then
    LOOP_VALUES=("${ALPHA_WEIGHT_VALUES[@]}")
    FIXED_PARAM="alpha"
else
    exit 1
fi

MU_FORMATTED=$(printf "%.2f" "$MU")

TOTAL_TASKS=$((${#DATASETS[@]} * ${#LOOP_VALUES[@]}))
CURRENT_TASK=0

for DATASET in "${DATASETS[@]}"; do
    for LOOP_VALUE in "${LOOP_VALUES[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))

        if [ "$LOOP_PARAM" = "alpha" ]; then
            CURRENT_ALPHA=$LOOP_VALUE
            CURRENT_ALPHA_WEIGHT=$ALPHA_WEIGHT
        else
            CURRENT_ALPHA=$ALPHA
            CURRENT_ALPHA_WEIGHT=$LOOP_VALUE
        fi

        ALPHA_FORMATTED=$(printf "%.2f" "$CURRENT_ALPHA")
        ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$CURRENT_ALPHA_WEIGHT")

        STAGE2_CHECKPOINT="output_stage2/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}/final"

        if [ ! -d "$STAGE2_CHECKPOINT" ]; then
            continue
        fi

        OUTPUT_DIR="results_stage3/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}"
        for SPLIT in hist test; do
            if [ "$SPLIT" = "hist" ]; then
                OUT_FILE="${OUTPUT_DIR}/hist_predictions.json"
            else
                OUT_FILE="${OUTPUT_DIR}/predictions.json"
            fi
            if [ -f "$OUT_FILE" ]; then
                continue
            fi

            CMD="python inference_stage3.py \
                --dataset $DATASET \
                --alpha $CURRENT_ALPHA \
                --mu $MU \
                --alpha_weight $CURRENT_ALPHA_WEIGHT \
                --base_model $BASE_MODEL \
                --stage2_checkpoint $STAGE2_CHECKPOINT \
                --num_beams $NUM_BEAMS \
                --batch_size $BATCH_SIZE \
                --micro_batch_size $MICRO_BATCH_SIZE \
                --max_new_tokens $MAX_NEW_TOKENS \
                --temperature $TEMPERATURE \
                --split $SPLIT"

        if [ "$USE_4BIT" = "true" ]; then
            CMD="$CMD --use_4bit"
        fi

        eval $CMD

        if [ $? -ne 0 ]; then
            :
        fi
        done
    done
done

for DATASET in "${DATASETS[@]}"; do
    for LOOP_VALUE in "${LOOP_VALUES[@]}"; do
        if [ "$LOOP_PARAM" = "alpha" ]; then
            CURRENT_ALPHA=$LOOP_VALUE
            CURRENT_ALPHA_WEIGHT=$ALPHA_WEIGHT
        else
            CURRENT_ALPHA=$ALPHA
            CURRENT_ALPHA_WEIGHT=$LOOP_VALUE
        fi

        ALPHA_FORMATTED=$(printf "%.2f" "$CURRENT_ALPHA")
        ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$CURRENT_ALPHA_WEIGHT")
        OUTPUT_DIR="results_stage3/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}"

        if [ -f "$OUTPUT_DIR/predictions.json" ]; then
            :
        else
            :
        fi
    done
done
