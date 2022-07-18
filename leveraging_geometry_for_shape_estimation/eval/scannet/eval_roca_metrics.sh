python $2/eval/scannet/metrics_roca.py $1
python $2/eval/scannet/analyse_R_and_T.py $1
python $2/eval/scannet/combine_single_predictions.py $1
python $2/eval/scannet/transform_from_single_frame_to_raw.py $1
python $2/eval/scannet/split_prediction.py $1
python $2/eval/scannet/cluster_3d.py $1
python $2/eval/scannet/scan2cad_constraint.py $1
python $2/eval/scannet/EvaluateBenchmark_v5.py $1
python $2/visualise_3d/visualise_predictions_from_csvs_v2.py $1

