import json

in_path = '/scratch2/fml35/results/ROCA/per_frame_best_no_null.json'
out_path = '/scratch2/fml35/results/ROCA/per_frame_best_no_null_correct_category_names.json'

with open(in_path,'r') as f:
    roca_preds = json.load(f)

for img in roca_preds:
    for detection in roca_preds[img]:
        if detection['category'] == 'bookcase':
            detection['category'] = 'bookshelf'
        
with open(out_path,'w') as f:
    json.dump(roca_preds,f)