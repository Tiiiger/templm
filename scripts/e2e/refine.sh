NUMBEAM=10
LENPEN=1.0

DATAPER=10
for SEED in 1 2 3; do
CKPT_PATH="$OUT/e2e/seq2seq_models/d${DATAPER}s${SEED}"
INFILL_CKPT_PATH="$OUT/e2e/infill_models/d${DATAPER}s${SEED}"
REFINE_DIR="$OUT/e2e/delex_bart/d${DATAPER}s${SEED}"
OUTPUTDIR="$OUT/e2e/refine/d${DATAPER}s${SEED}"

python ./template_refinement.py \
    --tokenizer_name $REFINE_DIR \
    --refinement_infill_model_path $INFILL_CKPT_PATH \
    --refinement_template_dir $REFINE_DIR \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --num_beams ${NUMBEAM} \
    --inference_batch_size 24 \
    --length_penalty 0.5 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --max_decode_steps 200 \
    --overwrite_output_dir \
    --dedup_input True \
    --local_search_baseline True \
    --local_search_augment_type True \
    --local_search_augment 50 \
    --seed $SEED \
    --field_tokens_cutoff 10 \
    --complex_field_transformation False \
    --refinement_const_cutoff -2 \
    --refinement_topn_delex 5

python ./eval_e2e_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name e2e_nlg \
    --output_dir ${OUTPUTDIR} \
    --tokenizer_name ${OUTPUTDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --length_penalty $LENPEN \
    --eval_typed_templates \
    --complex_field_transformation False \
    --field_tokens_cutoff 10

cd $DIR/e2e-metrics
./measure_scores.py $DIR/e2e_data/dev.txt $OUTPUTDIR/dev_output/typed_best_outputs.txt | tee $OUTPUTDIR/dev_output/metrics.txt
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
    --complex_field_transformation False \
    --field_tokens_cutoff 10 \
    --evaluation_split test

cd $DIR/e2e-metrics
./measure_scores.py $DIR/e2e_data/test.txt $OUTPUTDIR/test_output/typed_best_outputs.txt | tee $OUTPUTDIR/test_output/metrics.txt
cd $DIR

done
