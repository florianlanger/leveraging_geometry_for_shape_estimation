python $2/line_detection/line_detection.py $1
python $2/line_detection/crop_lines.py $1
python $2/line_detection/filter_lines_v3.py $1
# python $2/line_detection/filter_lines_exp_T_classifier.py $1
# python $2/line_detection/convert_correct_lines.py $1