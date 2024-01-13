TASK_NAME=mrpc
MINIBATCH=4
STRATEGY=vanilla
MODEL=bert-base-cased


# for INPUT_LEN_RATE in 1.0 0.7 0.8 0.9; do
#   CUDA_VISIBLE_DEVICES=5 python run_glue.py \
#     --model_name_or_path $MODEL \
#     --task_name $TASK_NAME \
#     --do_train \
#     --do_eval \
#     --max_seq_length 128 \
#     --per_device_train_batch_size 16 \
#     --evaluation_strategy steps \
#     --save_strategy steps \
#     --load_best_model_at_end \
#     --metric_for_best_model accuracy \
#     --logging_strategy steps \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --num_train_epochs 100 \
#     --minibatch $MINIBATCH \
#     --strategy $STRATEGY \
#     --input_len_rate $INPUT_LEN_RATE \
#     --report_to wandb \
#     --logging_steps 100 \
#     --run_name ${TASK_NAME}_${MODEL}_$STRATEGY-$INPUT_LEN_RATE \
#     --output_dir results/$TASK_NAME/$MODEL/$STRATEGY-$INPUT_LEN_RATE/
# done

INPUT_LEN_RATE=1.0
LAYER_SELECTION=RGN
LAYER_THRESHOLD=0.5

for TASK_NAME in mrpc sst2 qqp qnli; do 
for LAYER_THRESHOLD in 0.5 0.6 0.7 0.8 0.9; do
  CUDA_VISIBLE_DEVICES=4 python run_glue.py \
    --model_name_or_path $MODEL \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 16 \
    --evaluation_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --logging_strategy steps \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --num_train_epochs 100 \
    --minibatch $MINIBATCH \
    --strategy $STRATEGY \
    --input_len_rate $INPUT_LEN_RATE \
    --layer_selection $LAYER_SELECTION \
    --layer_threshold $LAYER_THRESHOLD \
    --report_to wandb \
    --logging_steps 100 \
    --run_name ${TASK_NAME}_${MODEL}_${STRATEGY}-${INPUT_LEN_RATE}_$LAYER_SELECTION-$LAYER_THRESHOLD \
    --output_dir results/$TASK_NAME/$MODEL/${STRATEGY}-${INPUT_LEN_RATE}_$LAYER_SELECTION-$LAYER_THRESHOLD/ \
    --overwrite_output_dir
  done
done