PRUNE=5
DATAPER=10
LENPEN=1.0
for SEED in 1 2 3; do
CKPT_PATH="$OUT/e2e/seq2seq_models/d${DATAPER}s${SEED}"
OUTPUTDIR="$OUT/e2e/delex_bart/d${DATAPER}s${SEED}"

python ./eval_e2e_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --length_penalty $LENPEN \
    --eval_typed_templates \
    --local_search_prune_topk $PRUNE \
    --field_tokens_cutoff 10

cd $DIR/e2e_metrics
./measure_scores.py $DIR/e2e_data/dev.txt $OUTPUTDIR/prune${PRUNE}/dev_output/typed_best_outputs.txt | tee $OUTPUTDIR/prune${PRUNE}/dev_output/metrics.txt
cd $DIR

python ./eval_e2e_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --length_penalty $LENPEN \
    --eval_typed_templates \
    --field_tokens_cutoff 10 \
    --local_search_prune_topk $PRUNE \
    --evaluation_split test

cd $DIR/e2e_metrics
./measure_scores.py $DIR/e2e_data/test.txt $OUTPUTDIR/prune${PRUNE}/test_output/typed_best_outputs.txt | tee $OUTPUTDIR/prune${PRUNE}/test_output/metrics.txt
cd $DIR

done
