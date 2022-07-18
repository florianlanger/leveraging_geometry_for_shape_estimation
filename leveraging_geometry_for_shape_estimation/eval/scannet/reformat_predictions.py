import json
import CSVHelper
import os
import glob
import numpy as np
import quaternion

def catid_to_name(catid):
    top = {}
    top["03211117"] = "display"
    top["04379243"] = "table"
    top["02808440"] = "bathtub"
    top["02747177"] = "bin"
    top["04256520"] = "sofa"
    top["03001627"] = "chair"
    top["02933112"] = "cabinet"
    top["02871439"] = "bookcase"
    top["02818832"] = "bed"

    # inv_map = {v: k for k, v in top.items()}
    return top[catid]

def main():
    dir_in = '/scratch2/fml35/results/ROCA/results_per_scene_own_2'
    dir_out = dir_in + '_reformated'

    # os.mkdir(dir_out)

    for file0 in glob.glob(dir_in + "/*.csv"):
        alignments = CSVHelper.read(file0)
        id_scan = os.path.basename(file0.rsplit(".", 1)[0])

        reformated_alignments = []

        for alignment in alignments:
            catid_cad = str(alignment[0]).zfill(8)
            id_cad = alignment[1]
            t = np.asarray(alignment[2:5], dtype=np.float64)
            q0 = np.asarray(alignment[5:9], dtype=np.float64)
            q = np.quaternion(q0[0], q0[1], q0[2], q0[3])
            s = np.asarray(alignment[9:12], dtype=np.float64)

            single_dict = {}
            single_dict["category"] = catid_to_name(catid_cad)
            single_dict["id_cad"] = id_cad
            single_dict["r"] = quaternion.as_rotation_matrix(q).tolist()
            single_dict["t"] = t.tolist()
            single_dict['s'] = s.tolist()
            reformated_alignments.append(single_dict)

        with open(file0.replace(dir_in,dir_out).replace('.csv','.json'),'w') as f:
            json.dump(reformated_alignments,f)



if __name__ == '__main__':
    main()