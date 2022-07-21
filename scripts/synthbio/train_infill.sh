export WANDB_PROJECT="tempretrain"
for SEED in 1 2 3; do
OUTPUTDIR="$OUT/synthbio/infill_models/s$SEED"
python ./train_infill.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --dataset_name ./datasets/synthbio.py \
    --output_dir $OUTPUTDIR \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=16 \
    --evaluation_strategy steps \
    --num_train_epochs 20 \
    --warmup_steps 500 \
    --save_steps 100000 \
    --learning_rate 3e-5 \
    --logging_steps 10 \
    --run_name bart_synthbio_infill_seed$SEED \
    --eval_steps 100 \
    --ignore_pad_token_for_loss \
    --fp16 \
    --group_by_length \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --overwrite_output_dir \
    --max_mask_tokens 15 \
    --seed $SEED
done