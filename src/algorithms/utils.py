from hashlib import algorithms_available
import os
import json
import resource
import tracemalloc
import importlib
import time
import csv
import colors
from src.algorithms.algorithm import AbstractAlgorithm
import psutil
from src.datasets.dataset import Dataset

timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")


def save_to_csv(
    function_name, algo, elapsed_time, memory_usage, ram, pipeline=False, step=False
):
    """Save the execution stats to csv file

    Args:
        function_name (function): Function name
        algo (AbrstractAlgorithm): Algorithm instance
        elapsed_time (int): Elapsed time
        memory_usage (int): Memory usage
        ram (int): RAM usage
        pipeline (bool, optional): The results are stored in pipeline folder . Defaults to False.
        step (bool, optional): The result are stored in pipeline step folder. Defaults to False.
    """
    if step:
        folder = f"{algo.ds_.name}_pipe_step/{algo.name}_mem{algo.mem_}_cpu{algo.cpu_}/"
    elif pipeline:
        folder = f"{algo.ds_.name}_pipe/{algo.name}_mem{algo.mem_}_cpu{algo.cpu_}"
    else:
        folder = f"{algo.ds_.name}/{algo.name}_mem{algo.mem_}_cpu{algo.cpu_}"

    path_name = "results"
    out_path = os.path.join(path_name, folder)
    if not os.path.exists(out_path):
        print(f"Creating folder: {out_path}")
        os.makedirs(out_path)

    # Create csv file for save results
    out_log = os.path.join(out_path, f"{algo.name}_run_{timestamp}.csv")

    if not os.path.exists(out_log):
        logger = csv.writer(open(out_log, "a"))
        logger.writerow(["method", "time", "memory", "ram"])
    else:
        logger = csv.writer(open(out_log, "a"))

    logger.writerow([function_name, elapsed_time, memory_usage, ram])


def timing(f):
    """
    Decorator that prints the execution time for the decorated function.
    Args:
        f (function): Function to be decorated

    Returns:
        function: Decorated function
    """
    process = psutil.Process(os.getpid())

    def wrap(*args, **kwargs):
        # Start the stopwatch
        if args[0].pipeline:
            return f(*args, **kwargs)
        start_time = time.time()
        memory_used_s = psutil.virtual_memory().used
        # Start tracking memory usage
        resource.setrlimit(
            resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        tracemalloc.start()

        # Call the function
        result = f(*args, **kwargs)
        args[0].force_execution()
        memory_used_e = psutil.virtual_memory().used
        print(f"Memory usage: {abs((memory_used_e - memory_used_s))}")
        memory_used = max((memory_used_e - memory_used_s), 0)
        # stopping the library
        tracemalloc.stop()

        # Calculate the difference
        ram = max(((tracemalloc.get_traced_memory()[1])), 0)

        # Stop the stopwatch and calculate the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        save_to_csv(f.__name__, args[0], elapsed_time, memory_used, ram)

        return result

    return wrap


def compile_json(data, algo: AbstractAlgorithm):
    """
    Given a configuration file compiles the values to include objects from libraries
    e.g.
    {"col1": "polars.Float32"}
    will be replaced with the object "polars.datatypes.Float32"
    """
    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = compile_json(data[i], algo)
        return data
    elif isinstance(data, dict):
        for k in data.keys():
            data[k] = compile_json(data[k], algo)
        return data
    else:
        return eval(data, globals())


def load_tests(path):
    """
    Load the tests for a specified dataset
    """
    print(f"Loading tests from {path}")
    f = open(path, "rt")
    return json.load(f)


def get_err_logger():
    """
    Get the error log file
    """
    out_path = os.path.join("results", "errors.log")
    return open(out_path, "at")


def set_algorithm_params(args: dict, mem_limit, cpu_limit) -> dict:
    """
    Get the parameters for the algorithm
    """
    args = json.dumps(args)
    return json.loads(
        args.replace("MEM_LIMIT", f"{str(mem_limit)}g")
        .replace("CPU_LIMIT", str(cpu_limit))
        .replace("PARALLELISM", str(cpu_limit))
    )


def execute_methods(methods: dict, ds: Dataset, algo: AbstractAlgorithm, step=False):
    """Execute the methods for a specified dataset and algorithm

    Args:
        methods (dict): dictionary containing the methods to execute
        ds (Dataset): Dataset to use
        algo (AbstractAlgorithm): Algorithm to use
        step (bool, optional): Execute pipeline by step, the execution will be forced at the end of every step.
                               Defaults to False.
    """
    err = get_err_logger()
    for test in methods:
        try:
            input = "input"
            start_time = time.time()
            if ("input" + "_" + algo.name) in test.keys():
                input = f"{input}_{algo.name}"
            if test["method"] == "load_dataset":
                # Load the data
                print("Running " + test["method"])
                algo.load_dataset(ds, **test[input])

            # skip the force execution step if we execute a pipeline by step
            elif (not step) and (test["method"] == "force_execution"):
                continue
            else:
                print("Running " + test["method"])
                input_cmd = test[input]

                # Checks if extra commands must be executed before
                # running the method (e.g. import extra libraries)
                if "pass" in input_cmd:
                    continue

                if "extra_commands" in input_cmd:
                    for cmd in input_cmd["extra_commands"]:
                        # print(f"Running extra command: {cmd}")
                        exec(cmd, globals())
                    input_cmd.pop("extra_commands")

                # Checks if some input requires to be compiled
                # i.e. replaced with objects from external libraries
                if "req_compile" in input_cmd:
                    for f in input_cmd["req_compile"]:
                        input_cmd[f] = compile_json(input_cmd[f], algo)
                    input_cmd.pop("req_compile")
                getattr(algo, test["method"])(**input_cmd)
                if "restore" in test:
                    algo.restore()
        except Exception as e:
            print(
                colors.color(
                    "Cannot execute the method: " + test["method"] + "because" + str(e),
                    fg="red",
                )
            )
            err.write(
                time.strftime("%Y-%m-%d %H:%M:%S")
                + " - ERROR - dataset: "
                + ds.name
                + ", algorithm: "
                + algo.name
                + ", method: "
                + test["method"]
                + " - "
                + str(e)
                + "\n"
            )
            err.close()
            exit(1)
    err.close()
