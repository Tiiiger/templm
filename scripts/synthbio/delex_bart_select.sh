NUMBEAM=5

for SEED in 1 2 3; do

CKPT_PATH="$OUT/synthbio/seq2seq_models/s$SEED"
OUTPUTDIR="$OUT/synthbio/delex_bart/s$SEED"
mkdir -p $OUTPUTDIR
python ./template_search.py \
    --tokenizer_name "facebook/bart-base" \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir $OUTPUTDIR \
    --num_beams ${NUMBEAM} \
    --inference_batch_size 24 \
    --length_penalty 1.0 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --max_decode_steps 200 \
    --overwrite_output_dir \
    --local_search_baseline \
    --local_search_compute_stats \
    --dedup_input True \
    --seed $SEED
cp -r $OUTPUTDIR $OUT/synthbio/delex_bart

python ./eval_synthbio_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --eval_typed_templates \
    --length_penalty 1.0 \
    --local_search_prune_topk 10 \
    --evaluation_split test

python ./eval_synthbio_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --eval_typed_templates \
    --length_penalty 1.0 \
    --local_search_prune_topk 10 \
    --evaluation_split val
done