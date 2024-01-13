MINIBATCH=4
STRATEGY=entropy
MODEL=bert-base-cased
# TASK_NAME=mrpc

for TASK_NAME in sst2 qqp qnli; do 
  for INPUT_LEN_RATE in 1.0 0.7 0.8 0.9; do
    CUDA_VISIBLE_DEVICES=7 python run_glue.py \
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
      --report_to wandb \
      --logging_steps 100 \
      --run_name ${TASK_NAME}_${MODEL}_$STRATEGY-$INPUT_LEN_RATE \
      --output_dir results/$TASK_NAME/$MODEL/$STRATEGY-$INPUT_LEN_RATE/ \
      --overwrite_output_dir
  done
done

# --max_steps 23000 \
# --num_train_epochs 100 \