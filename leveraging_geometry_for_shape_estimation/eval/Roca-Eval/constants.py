BENCHMARK_CLASSES = (
    'bathtub',
    'bin',
    'bookcase',
    'chair',
    'cabinet',
    'display',
    'sofa',
    'table',
)
ALL_CLASSES = (
    'bathtub',
    'bed',
    'bin',
    'bookcase',
    'chair',
    'cabinet',
    'display',
    'sofa',
    'table',
)

SYMMETRY_CLASS_IDS = {
    '__SYM_NONE': 0,
    '__SYM_ROTATE_UP_2': 1,
    '__SYM_ROTATE_UP_4': 2,
    '__SYM_ROTATE_UP_INF': 3
}
SYMMETRY_ID_CLASSES = {v: k for k, v in SYMMETRY_CLASS_IDS.items()}

CAD_TAXONOMY = {
    2747177: 'bin',
    2808440: 'bathtub',
    2818832: 'bed',
    2871439: 'bookcase',
    2933112: 'cabinet',
    3001627: 'chair',
    3211117: 'display',
    4256520: 'sofa',
    4379243: 'table'
}
CAD_TAXONOMY_REVERSE = {v: k for k, v in CAD_TAXONOMY.items()}

# IMAGE_SIZE = (480, 640)
IMAGE_SIZE = (360, 480)

VOXEL_RES = (32, 32, 32)
