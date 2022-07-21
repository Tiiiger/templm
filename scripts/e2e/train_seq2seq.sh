export WANDB_PROJECT="templm"
for SEED in 1 2 3; do
DATAPERTYPE=10

RUNNAME="d${DATAPERTYPE}s${SEED}"
OUTPUTDIR=$OUT/e2e/seq2seq_models/$RUNNAME
python ./train_seq2seq.py \
    --model_name_or_path facebook/bart-base \
    --do_train \
    --do_eval \
    --do_predict \
    --dataset_name e2e_nlg \
    --output_dir $OUTPUTDIR \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=32 \
    --overwrite_output_dir \
    --text_column meaning_representation \
    --summary_column human_reference \
    --evaluation_strategy steps \
    --num_train_epochs 10 \
    --warmup_steps 100 \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 3e-5 \
    --logging_steps 50 \
    --run_name $RUNNAME \
    --eval_steps 25 \
    --ignore_pad_token_for_loss \
    --data_per_type 10 \
    --dedup_input True \
    --seed $SEED
done