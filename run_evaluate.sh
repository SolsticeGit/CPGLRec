#!/bin/bash

RUN_STAGE1=false
RUN_STAGE2=true
RUN_STAGE3=false

DATASETS=("Beauty" "CDs_and_Vinyl")

ALPHA_VALUES=(0.1 0.3 0.5 0.7 0.9)

MU=0.3
ALPHA_WEIGHT=0.6

LAMBDA=0.1

K_LIST="1 5 10 20"

NO_RERANK=false

MU_FORMATTED=$(printf "%.2f" "$MU")
ALPHA_WEIGHT_FORMATTED=$(printf "%.2f" "$ALPHA_WEIGHT")

STAGE1_TASKS=0
STAGE2_TASKS=0
STAGE3_TASKS=0

if [ "$RUN_STAGE1" = "true" ]; then
    STAGE1_TASKS=$((${#DATASETS[@]} * ${#ALPHA_VALUES[@]}))
fi
if [ "$RUN_STAGE2" = "true" ]; then
    STAGE2_TASKS=$((${#DATASETS[@]} * ${#ALPHA_VALUES[@]} * 4))
fi
if [ "$RUN_STAGE3" = "true" ]; then
    STAGE3_TASKS=$((${#DATASETS[@]} * ${#ALPHA_VALUES[@]}))
fi

TOTAL_TASKS=$((STAGE1_TASKS + STAGE2_TASKS + STAGE3_TASKS))
CURRENT_TASK=0

if [ "$RUN_STAGE1" = "true" ]; then
    for DATASET in "${DATASETS[@]}"; do
        for ALPHA in "${ALPHA_VALUES[@]}"; do
            CURRENT_TASK=$((CURRENT_TASK + 1))

            CMD="python evaluate.py \
                --stage 1 \
                --dataset $DATASET \
                --alpha $ALPHA \
                --mu $MU \
                --lambda_val $LAMBDA \
                --k_list $K_LIST"

            if [ "$NO_RERANK" = "true" ]; then
                CMD="$CMD --no_rerank"
            fi

            eval $CMD

            if [ $? -ne 0 ]; then
                :
            fi
        done
    done
fi

if [ "$RUN_STAGE2" = "true" ]; then
    for DATASET in "${DATASETS[@]}"; do
        for ALPHA in "${ALPHA_VALUES[@]}"; do
            EXP_TYPES=("standard_dpo" "random" "overpop" "hard")

            for i in {0..3}; do
                CURRENT_TASK=$((CURRENT_TASK + 1))
                EXP_TYPE="${EXP_TYPES[$i]}"

                CMD="python evaluate.py \
                    --stage 2 \
                    --dataset $DATASET \
                    --alpha $ALPHA \
                    --mu $MU \
                    --alpha_weight $ALPHA_WEIGHT \
                    --exp_type $EXP_TYPE \
                    --lambda_val $LAMBDA \
                    --k_list $K_LIST"

                if [ "$NO_RERANK" = "true" ]; then
                    CMD="$CMD --no_rerank"
                fi

                eval $CMD

                if [ $? -ne 0 ]; then
                    :
                fi
            done
        done
    done
fi

if [ "$RUN_STAGE3" = "true" ]; then
    for DATASET in "${DATASETS[@]}"; do
        for ALPHA in "${ALPHA_VALUES[@]}"; do
            CURRENT_TASK=$((CURRENT_TASK + 1))

            CMD="python evaluate.py \
                --stage 3 \
                --dataset $DATASET \
                --alpha $ALPHA \
                --mu $MU \
                --alpha_weight $ALPHA_WEIGHT \
                --lambda_val $LAMBDA \
                --k_list $K_LIST"

            if [ "$NO_RERANK" = "true" ]; then
                CMD="$CMD --no_rerank"
            fi

            eval $CMD

            if [ $? -ne 0 ]; then
                :
            fi
        done
    done
fi
