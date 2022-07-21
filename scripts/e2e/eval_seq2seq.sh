for SEED in 1 2 3; do
  RUNNAME="d10s$SEED"
  CKPT_PATH="$OUT/e2e/$RUNNAME"
  OUT="$OUT/e2e/seq2seq_models/$RUNNAME"

  python ./eval_baselines.py \
      --model_name_or_path $CKPT_PATH \
      --dataset_name e2e_nlg \
      --output_dir $OUT \
      --tokenizer_name $CKPT_PATH \
      --per_device_eval_batch_size=10 \
      --inference_batch_size 100 \
      --num_beams 10 \
      --length_penalty 1.0 \
      --dedup_input False

  cd $DIR/e2e-metrics
  echo "testing"
  ./measure_scores.py $DIR/e2e_data/dev.txt $OUT/dev_output/baseline_output.txt > $OUT/dev_output/metrics.txt
  tail -n10 $OUT/dev_output/metrics.txt
  cd $DIR

  python ./eval_baselines.py \
      --model_name_or_path $CKPT_PATH \
      --dataset_name e2e_nlg \
      --output_dir $OUT \
      --tokenizer_name $CKPT_PATH \
      --per_device_eval_batch_size=10 \
      --inference_batch_size 100 \
      --num_beams 10 \
      --length_penalty 1.0 \
      --dedup_input False \
      --evaluation_split test

  cd $DIR/e2e-metrics
  echo "testing"
  ./measure_scores.py $DIR/e2e_data/test.txt $OUT/test_output/baseline_output.txt > $OUT/test_output/metrics.txt
  tail -n10 $OUT/test_output/metrics.txt
  cd $DIR
done