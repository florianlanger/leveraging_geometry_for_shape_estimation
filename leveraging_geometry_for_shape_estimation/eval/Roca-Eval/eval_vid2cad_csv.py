import argparse
import sys

from category_manager import register_categories
from vid2cad_evaluation import eval_csv


def main(args):
    if not isinstance(args, argparse.Namespace):
        parser = argparse.ArgumentParser()
        parser.add_argument('--csv', required=True)
        parser.add_argument('--full_annot', required=True)
        parser.add_argument('--grid_file', default='none')
        parser.add_argument('--exact_ret', type=int, default=0)
        parser.add_argument('--info_file', default='')
        args = parser.parse_args(args)

    try:
        category_file = 'scan2cad_alignment_classes.json'
        register_categories('Scan2CAD', category_file)
    except AssertionError:  # To avoid duplicate registration
        pass

    if args.grid_file == 'none':
        grid_file = None
    else:
        grid_file = args.grid_file

    return eval_csv(
        dataset_name='Scan2CAD',
        csv_path=args.csv,
        full_annot=args.full_annot,
        grid_file=grid_file,
        exact_ret=args.exact_ret,
        info_file=args.info_file
    )


if __name__ == '__main__':
    main(sys.argv[1:])
