#!/bin/bash

DATASETS=("Beauty" "CDs_and_Vinyl")

LOOP_PARAM="alpha"

ALPHA_VALUES=(0.1 0.3 0.5 0.7 0.9)
ALPHA=0.3

MU_VALUES=(0.1 0.3 0.5 0.7 1.0)
MU=0.3

BASE_MODEL="/home/power/MyLLMs/Qwen2.5-1.5B-Instruct"
TEMPLATE="qwen"

LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
USE_4BIT=true

NUM_EPOCHS=3
BATCH_SIZE=96
MICRO_BATCH_SIZE=12
LEARNING_RATE=3e-4
MAX_LENGTH=512
TEMPERATURE=1.0

LOGGING_STEPS=10
SAVE_STEPS=500

GRADIENT_ACCUMULATION_STEPS=$((BATCH_SIZE / MICRO_BATCH_SIZE))

if [ "$LOOP_PARAM" = "alpha" ]; then
    TOTAL_TASKS=$((${#DATASETS[@]} * ${#ALPHA_VALUES[@]}))
else
    TOTAL_TASKS=$((${#DATASETS[@]} * ${#MU_VALUES[@]}))
fi
CURRENT_TASK=0

for DATASET in "${DATASETS[@]}"; do
    if [ "$LOOP_PARAM" = "alpha" ]; then
        for ALPHA_VAL in "${ALPHA_VALUES[@]}"; do
            CURRENT_TASK=$((CURRENT_TASK + 1))

            ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA_VAL")
            MU_FORMATTED=$(printf "%.2f" "$MU")

            OUTPUT_DIR="output_stage1/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}"
            if [ -d "$OUTPUT_DIR/final" ]; then
                continue
            fi

            CMD="python train_stage1.py \
                --dataset $DATASET \
                --alpha $ALPHA_VAL \
                --base_model $BASE_MODEL \
                --template $TEMPLATE \
                --lora_r $LORA_R \
                --lora_alpha $LORA_ALPHA \
                --lora_dropout $LORA_DROPOUT \
                --num_epochs $NUM_EPOCHS \
                --batch_size $MICRO_BATCH_SIZE \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                --learning_rate $LEARNING_RATE \
                --max_length $MAX_LENGTH \
                --mu $MU \
                --temperature $TEMPERATURE \
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
        for MU_VAL in "${MU_VALUES[@]}"; do
            CURRENT_TASK=$((CURRENT_TASK + 1))

            ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA")
            MU_FORMATTED=$(printf "%.2f" "$MU_VAL")

            OUTPUT_DIR="output_stage1/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}"
            if [ -d "$OUTPUT_DIR/final" ]; then
                continue
            fi

            CMD="python train_stage1.py \
                --dataset $DATASET \
                --alpha $ALPHA \
                --base_model $BASE_MODEL \
                --template $TEMPLATE \
                --lora_r $LORA_R \
                --lora_alpha $LORA_ALPHA \
                --lora_dropout $LORA_DROPOUT \
                --num_epochs $NUM_EPOCHS \
                --batch_size $MICRO_BATCH_SIZE \
                --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
                --learning_rate $LEARNING_RATE \
                --max_length $MAX_LENGTH \
                --mu $MU_VAL \
                --temperature $TEMPERATURE \
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
            MU_FORMATTED=$(printf "%.2f" "$MU")
            OUTPUT_DIR="output_stage1/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}"
            if [ -d "$OUTPUT_DIR/final" ]; then
                :
            else
                :
            fi
        done
    else
        for MU_VAL in "${MU_VALUES[@]}"; do
            ALPHA_FORMATTED=$(printf "%.2f" "$ALPHA")
            MU_FORMATTED=$(printf "%.2f" "$MU_VAL")
            OUTPUT_DIR="output_stage1/${DATASET}/alpha_${ALPHA_FORMATTED}_mu_${MU_FORMATTED}"
            if [ -d "$OUTPUT_DIR/final" ]; then
                :
            else
                :
            fi
        done
    fi
done
