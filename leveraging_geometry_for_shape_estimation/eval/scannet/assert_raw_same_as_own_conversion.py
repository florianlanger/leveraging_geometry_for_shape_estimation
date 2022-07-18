import numpy as np

# with open('/scratch2/fml35/results/ROCA/own_raw_results_2.csv') as f:
with open('/scratch2/fml35/experiments/leveraging_geometry_for_shape/exp_154_eval_roca/global_stats/eval_scannet/raw_results.csv') as f:
    own = f.readlines()

with open('/scratch2/fml35/results/ROCA/raw_results.csv') as f:
    raw = f.readlines()

assert len(own) == len(raw)
assert own[0] == raw[0]

for i in range(1,len(own)):

    # THIS is because there are two predictions in the same scene with the same probability 0.9721203446388245 and they happen to be sorted in the reverse order
    if i == 9289:
        raw_line = raw[9290]
    elif i == 9290:
        raw_line = raw[9289] 
    else:
        raw_line = raw[i]

    split_own = own[i].replace('\n','').split(',')
    split_raw = raw_line.replace('\n','').split(',')

    assert split_own[1] == split_raw[1].zfill(8),(split_own[1],split_raw[1])
    for j in [0,2]:
        assert split_own[j] == split_raw[j],(split_own[j],split_raw[j])
    for j in range(3,14):
        if np.abs(float(split_own[j]) - float(split_raw[j])) > 0.00001:
            print(i,split_own[3:6])
            print(i,split_raw[3:6])
        #     print(split_own)
        #     print(split_raw)
        # assert np.abs(float(split_own[j]) - float(split_raw[j])) < 0.00001 ,(i,split_own[j],split_raw[j],split_own,split_raw)