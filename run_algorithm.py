
import argparse
import sys
import os

if __name__ == "__main__":
    
    #Set up argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--algorithm',
        help='Algorithm name',
        required=True
    )
    parser.add_argument(
        '--algorithm_params',
        help='Algorithm configuration parameters',
        required=False
    )
    parser.add_argument(
        '--dataset',
        help='Dataset name',
        required=True
    )
    parser.add_argument(
        '--pipeline',
        help='flag for execute function in pipeline mode',
        action="store_true"
    )
    parser.add_argument(
        '--pipeline-step',
        help='flag for execute function i pipiple divided by step',
        action="store_true"
    )
    parser.add_argument(
        '--locally',
        help='Runs the algorithm locally instead of on docker',
        action="store_true"
    )
    parser.add_argument(
        '--requirements',
        help='Runs the algorithm locally instead of on docker',
        action="store_true"
    )
    parser.add_argument(
        '--mem-limit',
        help='Memory limit for docker container (default maximum available memory)',
        default=None
    )
    parser.add_argument(
        '--cpu-limit',
        help='CPU limit for docker container (number of CPUs)',
        default=1,
        type=int
    )
    args = parser.parse_args()

    if args.locally:
        print("Pipeline mode running locally")
                # check python version
        if sys.version_info[0] < 3:
            raise Exception("Must be using Python 3")

        # install requirements
        if args.requirements:
            os.system("pip install -r requirements.txt")

        from src.datasets.dataset import Dataset
        ds = Dataset().get_dataset_by_name(args.dataset)
        if ds is None:
            raise Exception(f"Dataset {args.dataset} not found")

        from src.algorithms.run import run_pipeline_locally
        run_pipeline_locally(args.algorithm, args.dataset, args.cpu_limit, args.mem_limit, args.algorithm_params, args.pipeline, args.pipeline_step)

    # if args.locally:
    #     # check python version
    #     if sys.version_info[0] < 3:
    #         raise Exception("Must be using Python 3")

    #     # install requirements
    #     if args.requirements:
    #         os.system("pip install -r requirements.txt")

    #     from src.algorithms.run import run_algo_locally
    #     from src.datasets.dataset import Dataset

    #     ds = Dataset().get_dataset_by_name(args.dataset)
    #     if ds is None:
    #         raise Exception(f"Dataset {args.dataset} not found")
    #     print("Running test functions")
    #     run_algo_locally(args.algorithm, args.dataset, args.algorithm_params, args.cpu_limit, args.mem_limit)


    else:
        try:
            import docker
        except Exception:
            os.system("pip install docker")
            import docker

        from src.algorithms.run import run_algo_docker
        from src.algorithms.utils import install
        # check return code
        ret = install(args.algorithm, args.algorithm_params)
        if ret != 0:
            raise Exception("Error building docker image")
        run_algo_docker(args.algorithm, args.dataset, args.algorithm_params, args.cpu_limit, args.mem_limit, args.pipeline, args.pipeline_step)