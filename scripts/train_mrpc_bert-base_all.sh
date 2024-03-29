TASK_NAME=mrpc
MINIBATCH=4
STRATEGY=all
MODEL=bert-base-cased

CUDA_VISIBLE_DEVICES=0 python run_glue.py \
  --model_name_or_path $MODEL \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 4 \
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
  --report_to wandb \
  --logging_steps 100 \
  --run_name ${TASK_NAME}_${MODEL}_${STRATEGY} \
  --output_dir results/$TASK_NAME/$MODEL/$STRATEGY/ \
  --overwrite_output_dir


# --max_steps 23000 \
# --num_train_epochs 100 \