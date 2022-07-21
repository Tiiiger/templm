NUMBEAM=5
SEED=1
CKPT_PATH="$OUT/synthbio/seq2seq_models/bart_s$SEED"
INFILL_CKPT_PATH="$OUT/synthbio/infill_models/s$SEED"
REFINE_DIR="$OUT/synthbio/delex_bart/s$SEED"
TOKENIZER_PATH=$REFINE_DIR
OUTPUT_DIR="$OUT/synthbio/refine/s$SEED"
python ./template_refinement.py \
    --tokenizer_name $TOKENIZER_PATH \
    --model_name_or_path $CKPT_PATH \
    --refinement_infill_model_path $INFILL_CKPT_PATH \
    --refinement_template_dir $REFINE_DIR \
    --refinement_topn_delex 10 \
    --refinement_const_cutoff -2 \
    --refinement_lex_only True \
    --dataset_name ./datasets/synthbio.py \
    --output_dir $OUTPUT_DIR \
    --num_beams ${NUMBEAM} \
    --inference_batch_size 24 \
    --length_penalty 0.5 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --max_decode_steps 200

for SEED in 1 2 3; do
CKPT_PATH="$OUT/synthbio/seq2seq_models/bart_s$SEED"
OUTPUT_DIR="$OUT/synthbio/refine/s$SEED"
TOKENIZERDIR="$OUT/synthbio/delex_bart/s$SEED"

python ./eval_synthbio_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir ${OUTPUT_DIR} \
    --tokenizer_name ${TOKENIZERDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --eval_typed_templates \
    --length_penalty 1.0

CKPT_PATH="$OUT/synthbio/seq2seq_models/s$SEED"
OUTPUT_DIR="$OUT/synthbio/refine/s$SEED"
TOKENIZERDIR="$OUT/synthbio/delex_bart/s$SEED"

python ./eval_synthbio_templates.py \
    --model_name_or_path $CKPT_PATH \
    --dataset_name ./datasets/synthbio.py \
    --output_dir ${OUTPUT_DIR} \
    --tokenizer_name ${TOKENIZERDIR} \
    --per_device_eval_batch_size=10 \
    --inference_batch_size 100 \
    --eval_typed_templates \
    --length_penalty 1.0 \
    --evaluation_split test
done