export WANDB_PROJECT="templm"
DATAPERTYPE=10
for SEED in 1 2 3; do
RUNNAME="d${DATAPERTYPE}s${SEED}"
OUTPUTDIR=$OUT/e2e/infill_models/$RUNNAME
python ./train_infill.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --dataset_name e2e_nlg \
    --output_dir $OUTPUTDIR \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --evaluation_strategy steps \
    --warmup_steps 500 \
    --save_steps 100000 \
    --learning_rate 3e-5 \
    --logging_steps 10 \
    --run_name "$RUNNAME" \
    --eval_steps 1000 \
    --ignore_pad_token_for_loss \
    --fp16 \
    --group_by_length \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --overwrite_output_dir \
    --max_mask_tokens 10 \
    --min_mask_tokens 0 \
    --seed $SEED \
    --dedup_input True \
    --data_per_type $DATAPERTYPE \
    --num_train_epochs 50
  done