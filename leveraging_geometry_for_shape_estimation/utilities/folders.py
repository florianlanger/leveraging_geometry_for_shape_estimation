import os

def make_dir_save(out_dir,assert_not_exist=True):
    if os.path.exists(out_dir):
        if assert_not_exist == True:
            assert os.listdir(out_dir) == []
    else:
        os.mkdir(out_dir)


def make_empty_folder_structure(inputpath,outputpath):
    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder {} already exists!".format(structure))


if __name__ == '__main__':
    inputpath = '/scratch2/fml35/datasets/shapenet_v2/ShapeNetRenamed/model/'
    outputpath = '/scratch/fml35/datasets/shapenet_v2/ShapeNetRenamed/model_fixed/'
    make_empty_folder_structure(inputpath,outputpath)