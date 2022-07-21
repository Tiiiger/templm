for SEED in 1 2 3; do
DATAPER=10
CKPT_PATH="$OUT/e2e/seq2seq_models/d${DATAPER}s${SEED}"
OUTPUTDIR="$OUT/e2e/delex_gt/d${DATAPER}s${SEED}"

python ./template_search.py \
    --tokenizer_name "facebook/bart-base" \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --inference_batch_size 24 \
    --length_penalty 1.0 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --overwrite_output_dir \
    --data_per_type $DATAPER \
    --dedup_input True \
    --local_search_compute_stats True \
    --seed $SEED \
    --field_tokens_cutoff 10 \
    --complex_field_transformation False

python ./eval_e2e_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --length_penalty 1.0 \
    --eval_typed_templates \
    --field_tokens_cutoff 10 \
    --dedup_input True

cd $DIR/e2e_metrics
./measure_scores.py $DIR/e2e_data/dev.txt $OUTPUTDIR/dev_output/typed_best_outputs.txt | tee $OUTPUTDIR/dev_output/metrics.txt
cd $DIR

python ./eval_e2e_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --length_penalty 1.0 \
    --eval_typed_templates \
    --field_tokens_cutoff 10 \
    --dedup_input True \
    --evaluation_split test

cd $DIR/e2e_metrics
./measure_scores.py $DIR/e2e_data/test.txt $OUTPUTDIR/test_output/typed_best_outputs.txt | tee $OUTPUTDIR/test_output/metrics.txt
cd $DIR

python ./eval_e2e_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --length_penalty 1.0 \
    --eval_typed_templates \
    --field_tokens_cutoff 10 \
    --dedup_input True \
    --random_selection_inference

cd $DIR/e2e_metrics
./measure_scores.py $DIR/e2e_data/dev.txt $OUTPUTDIR/random_inference/dev_output/typed_best_outputs.txt | tee $OUTPUTDIR/random_inference/dev_output/metrics.txt
cd $DIR

python ./eval_e2e_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --length_penalty 1.0 \
    --eval_typed_templates \
    --field_tokens_cutoff 10 \
    --dedup_input True \
    --evaluation_split test \
    --random_selection_inference

cd $DIR/e2e_metrics
./measure_scores.py $DIR/e2e_data/test.txt $OUTPUTDIR/random_inference/test_output/typed_best_outputs.txt | tee $OUTPUTDIR/random_inference/test_output/metrics.txt
cd $DIR

done