export WANDB_PROJECT="tempretrain"
for SEED in 1 2 3; do
OUTPUTDIR=$OUT/synthbio/seq2seq_models/s$SEED
python ./train_seq2seq.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --dataset_name ./datasets/synthbio.py \
    --output_dir $OUTPUTDIR \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=16 \
    --evaluation_strategy steps \
    --num_train_epochs 5 \
    --warmup_steps 500 \
    --save_steps 500 \
    --learning_rate 3e-5 \
    --logging_steps 10 \
    --run_name bart_synthbio \
    --eval_steps 100 \
    --ignore_pad_token_for_loss \
    --fp16 \
    --group_by_length \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --overwrite_output_dir \
    --dedup_input False \
    --seed $SEED

python ./eval_baselines.py \
    --model_name_or_path $OUTPUTDIR \
    --dataset_name ./datasets/synthbio.py \
    --output_dir $OUTPUTDIR \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 25 \
    --num_beams 5 \
    --length_penalty 1.5 \
    --max_decode_steps 200 \
    --dedup_input False \

python ./eval_baselines.py \
    --model_name_or_path $OUTPUTDIR \
    --dataset_name ./datasets/synthbio.py \
    --output_dir $OUTPUTDIR \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 25 \
    --num_beams 5 \
    --length_penalty 1.5 \
    --max_decode_steps 200 \
    --dedup_input False \
    --evaluation_split test
done
