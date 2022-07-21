for SEED in 1 2 3; do
  RUNNAME="d10s${SEED}"
  OUTPUTDIR="$OUT/e2e/seq2seq_models/$RUNNAME"
  python fact_checker.py --output_dir $OUTPUTDIR/lenpen_1.0/dev_output --output_file_path $OUTPUTDIR/lenpen_1.0/dev_output/baseline_output.txt --split_name validation
done

for SEED in 1 2 3; do
  RUNNAME="d10s${SEED}"
  OUTPUTDIR="$OUT/e2e/refine/$RUNNAME"
  python fact_checker.py --output_dir $OUTPUTDIR/dev_output --output_file_path $OUTPUTDIR/dev_output/typed_best_outputs.txt --split_name validation
done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/delex_gt/$RUNNAME"
#  python fact_checker.py --output_dir $OUTPUTDIR/dev_output --output_file_path $OUTPUTDIR/dev_output/typed_best_outputs.txt --split_name validation
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/delex_gt/$RUNNAME/random_inference"
#  python fact_checker.py --output_dir $OUTPUTDIR/dev_output --output_file_path $OUTPUTDIR/dev_output/typed_best_outputs.txt --split_name validation
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/delex_bart/$RUNNAME"
#  python fact_checker.py --output_dir $OUTPUTDIR/dev_output --output_file_path $OUTPUTDIR/dev_output/typed_best_outputs.txt --split_name validation
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/delex_bart/$RUNNAME/prune5"
#  python fact_checker.py --output_dir $OUTPUTDIR/dev_output --output_file_path $OUTPUTDIR/dev_output/typed_best_outputs.txt --split_name validation
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/seq2seq_models/$RUNNAME"
#  python fact_checker.py --output_dir $OUTPUTDIR/lenpen_1.0/test_output --output_file_path $OUTPUTDIR/lenpen_1.0/test_output/baseline_output.txt --split_name test
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/refine/$RUNNAME"
#  python fact_checker.py --output_dir $OUTPUTDIR/test_output --output_file_path $OUTPUTDIR/test_output/typed_best_outputs.txt --split_name test
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/delex_gt/$RUNNAME"
#  python fact_checker.py --output_dir $OUTPUTDIR/test_output --output_file_path $OUTPUTDIR/test_output/typed_best_outputs.txt --split_name test
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/delex_gt/$RUNNAME/random_inference"
#  python fact_checker.py --output_dir $OUTPUTDIR/test_output --output_file_path $OUTPUTDIR/test_output/typed_best_outputs.txt --split_name test
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/delex_bart/$RUNNAME"
#  python fact_checker.py --output_dir $OUTPUTDIR/test_output --output_file_path $OUTPUTDIR/test_output/typed_best_outputs.txt --split_name test
#done
#
#for SEED in 1 2 3; do
#  RUNNAME="d10s${SEED}"
#  OUTPUTDIR="$OUT/e2e/delex_bart/$RUNNAME/prune5"
#  python fact_checker.py --output_dir $OUTPUTDIR/test_output --output_file_path $OUTPUTDIR/test_output/typed_best_outputs.txt --split_name test
#done
