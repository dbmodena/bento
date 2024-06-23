from src.algorithms.run import run_algo_locally, run_algo_docker, run_pipeline_locally
import argparse

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
    #If locally is set to true the algorithm will run locally
    if args.locally:
        # algorithm, dataset, cpu_limit=None, mem_limit=None, algorithm_params=None, pipeline=False)
        run_pipeline_locally(args.algorithm, args.dataset, args.cpu_limit, args.mem_limit, args.algorithm_params, args.pipeline, args.pipeline_step)
    else:
        run_algo_docker(args.algorithm, args.dataset, args.cpu_limit, args.mem_limit, args.pipeline)