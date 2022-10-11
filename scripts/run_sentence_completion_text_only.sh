TASK=sentence_completion
DATASET=activitynet
LOG_DIR=log/activitynet

MODEL=gpt2
MODE=vanilla
PORTION=0.01
TRAIN_EPOCH=20
TRAIN_BATCH_SIZE=8
GRAD_ACC=1
WARM_UP=400
LR=2e-5
MAX_INPUT_LENGTH=100
MAX_OUTPUT_LENGTH=100
RANDOM_SEEDS=41,42,43
TEST_SET=test_both


equal_batch_size=$(($TRAIN_BATCH_SIZE*$GRAD_ACC))
echo "batch_size = ${equal_batch_size}"

LABEL=${DATASET}_td${PORTION}_ep${TRAIN_EPOCH}_bs${equal_batch_size}_lr${LR}_warm${WARM_UP}

CUDA_VISIBLE_DEVICES=0 python code/main.py \
--model_type $MODEL \
--mode $MODE \
--task $TASK \
--dataset $DATASET \
--train_ratio $PORTION \
--train_epoch ${TRAIN_EPOCH} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACC} \
--max_input_length ${MAX_INPUT_LENGTH} \
--max_output_length ${MAX_OUTPUT_LENGTH} \
--lr $LR \
--warmup_steps ${WARM_UP} \
--test_set ${TEST_SET} \
--random_seeds ${RANDOM_SEEDS} \
--label $LABEL \
--verbose 0 \
--skip_meteor --skip_spice |& tee ${LOG_DIR}/${MODEL}-$LABEL.out 
