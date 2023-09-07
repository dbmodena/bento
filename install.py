import os
import argparse
from src.algorithms.utils import install
from multiprocessing import Pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algorithm',
        metavar='NAME',
        help='build only the named algorithm image',
        default=None)
    parser.add_argument(
        '--build-arg',
        help='pass given args to all docker builds',
        nargs="+")
    args = parser.parse_args()

    install(args.algorithm, args.build_arg)