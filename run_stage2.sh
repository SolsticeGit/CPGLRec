#!/bin/bash

DATASETS=("Beauty" "CDs_and_Vinyl")

LOOP_PARAM="alpha"

ALPHA_VALUES=(0.1 0.3 0.5 0.7 0.9)
ALPHA=0.3

ALPHA_WEIGHT_VALUES=(0.2 0.4 0.6 0.8 1.0)
ALPHA_WEIGHT=0.6

MU=0.3

BASE_MODEL="/home/power/MyLLMs/Qwen2.5-1.5B-Instruct"

NUM_EPOCHS=2
BATCH_SIZE=64
MICRO_BATCH_SIZE=8
LEARNING_RATE=5e-6
BETA=0.1
USE_4BIT=false

LOGGING_STEPS=10
SAVE_STEPS=500

MU_FORMATTED=$(printf "%.2f" "$MU")

GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE / MICRO_BATCH_SIZE))

if [ "$LOOP_PARAM" = "alpha" ]; then
    TOTAL_TASKS=$((${#DATASETS[@]} * ${#ALPHA_VALUES[@]}))
else
    TOTAL_TASKS=$((${#DATASETS[@]} * ${#ALPHA_WEIGHT_VALUES[@]}))
fi
CURRENT_TASK=0

for DATASET in "${DATASETS[@]}"; do
    if [ "$LOOP_PARAM" = "alpha" ]; then
        for ALPHA_VAL in "${ALPHA_VALUES[@]}"; do
            CURRENT_TASK=$((CURRENT_TASK + 1))

            ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA_VAL")
            ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$ALPHA_WEIGHT")

            STAGE1_CHECKPOINT="output_stage1/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}/final"
            if [ ! -d "$STAGE1_CHECKPOINT" ]; then
                continue
            fi

            DPO_DATA_PATH="dpo_data/${DATASET}/dpo_pairs_alpha_${ALPHA_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}.jsonl"
            if [ ! -f "$DPO_DATA_PATH" ]; then
                continue
            fi

            OUTPUT_DIR="output_stage2/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}"
            if [ -d "$OUTPUT_DIR/final" ]; then
                continue
            fi

            CMD="python train_stage2.py \
                --dataset $DATASET \
                --alpha $ALPHA_VAL \
                --mu $MU \
                --alpha_weight $ALPHA_WEIGHT \
                --base_model $BASE_MODEL \
                --stage1_checkpoint $STAGE1_CHECKPOINT \
                --beta $BETA \
                --num_epochs $NUM_EPOCHS \
                --batch_size $MICRO_BATCH_SIZE \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                --learning_rate $LEARNING_RATE \
                --logging_steps $LOGGING_STEPS \
                --save_steps $SAVE_STEPS"

            if [ "$USE_4BIT" = "true" ]; then
                CMD="$CMD --use_4bit"
            fi

            eval $CMD

            if [ $? -ne 0 ]; then
                :
            fi
        done
    else
        for ALPHA_WEIGHT_VAL in "${ALPHA_WEIGHT_VALUES[@]}"; do
            CURRENT_TASK=$((CURRENT_TASK + 1))

            ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA")
            ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$ALPHA_WEIGHT_VAL")

            STAGE1_CHECKPOINT="output_stage1/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}/final"
            if [ ! -d "$STAGE1_CHECKPOINT" ]; then
                continue
            fi

            DPO_DATA_PATH="dpo_data/${DATASET}/dpo_pairs_alpha_${ALPHA_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}.jsonl"
            if [ ! -f "$DPO_DATA_PATH" ]; then
                continue
            fi

            OUTPUT_DIR="output_stage2/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}"
            if [ -d "$OUTPUT_DIR/final" ]; then
                continue
            fi

            CMD="python train_stage2.py \
                --dataset $DATASET \
                --alpha $ALPHA \
                --mu $MU \
                --alpha_weight $ALPHA_WEIGHT_VAL \
                --base_model $BASE_MODEL \
                --stage1_checkpoint $STAGE1_CHECKPOINT \
                --beta $BETA \
                --num_epochs $NUM_EPOCHS \
                --batch_size $MICRO_BATCH_SIZE \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                --learning_rate $LEARNING_RATE \
                --logging_steps $LOGGING_STEPS \
                --save_steps $SAVE_STEPS"

            if [ "$USE_4BIT" = "true" ]; then
                CMD="$CMD --use_4bit"
            fi

            eval $CMD

            if [ $? -ne 0 ]; then
                :
            fi
        done
    fi
done

for DATASET in "${DATASETS[@]}"; do
    if [ "$LOOP_PARAM" = "alpha" ]; then
        for ALPHA_VAL in "${ALPHA_VALUES[@]}"; do
            ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA_VAL")
            ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$ALPHA_WEIGHT")
            OUTPUT_DIR="output_stage2/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}"
            if [ -d "$OUTPUT_DIR/final" ]; then
                :
            else
                :
            fi
        done
    else
        for ALPHA_WEIGHT_VAL in "${ALPHA_WEIGHT_VALUES[@]}"; do
            ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA")
            ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$ALPHA_WEIGHT_VAL")
            OUTPUT_DIR="output_stage2/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}_aw_${ALPHA_WEIGHT_FORMATTED}"
            if [ -d "$OUTPUT_DIR/final" ]; then
                :
            else
                :
            fi
        done
    fi
done
