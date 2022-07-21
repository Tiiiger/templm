for SEED in 1 2 3; do
CKPT_PATH="$OUT/synthbio/seq2seq_models/s$SEED"
OUTPUTDIR="$OUT/synthbio/delex_gt/s$SEED"
python ./template_search.py \
    --tokenizer_name "facebook/bart-base" \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir $OUTPUTDIR \
    --inference_batch_size 24 \
    --length_penalty 0.0 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --max_decode_steps 200 \
    --overwrite_output_dir \
    --dedup_input True

python ./eval_synthbio_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 50 \
    --eval_typed_templates \
    --length_penalty 1.0 \
    --evaluation_split test

python ./eval_synthbio_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 25 \
    --eval_typed_templates \
    --length_penalty 1.0

python ./eval_synthbio_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 25 \
    --eval_typed_templates \
    --length_penalty 1.0 \
    --evaluation_split test \
    --random_selection_inference

python ./eval_synthbio_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 25 \
    --eval_typed_templates \
    --length_penalty 1.0 \
    --random_selection_inference
done