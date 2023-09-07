from src.datasets.dataset import Dataset
import argparse
import json
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        help='Dataset name',
        required=True
    )
    parser.add_argument(
        '--dataset_params',
        help='Dataset JSON configuration parameters',
        required=True,
        type=json.loads
    )
    args = parser.parse_args()
    
    ds = Dataset(name=args.dataset, dataset_attribute=args.dataset_params)
    ds.add_dataset()
    