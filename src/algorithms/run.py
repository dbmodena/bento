import contextlib
import json
import os
import threading
import time
import traceback
import tracemalloc

import colors
import docker
import psutil

from src.algorithms.algorithms_factory import AlgorithmsFactory
from src.algorithms.utils import *
from src.datasets.dataset import Dataset


def run_algo_locally(
    algorithm, dataset, algorithm_params=None, cpu_limit="1", mem_limit=None
):
    """
    Runs an algorithm on local machine
    """
    factory = AlgorithmsFactory()

    log_mem_limit = mem_limit
    if mem_limit is None:
        mem_limit = int(psutil.virtual_memory().available / (1024.0**3))
        log_mem_limit = "max"

    if algorithm_params:
        algorithm_params = set_algorithm_params(algorithm_params, mem_limit, cpu_limit)

    algo = factory.build_algorithm(
        algorithm, mem_limit, cpu_limit, conf=algorithm_params, pipeline=False
    )

    # Create log file
    ds = Dataset().get_dataset_by_name(dataset)

    # Load the tests to perform on the dataset
    tests = load_tests(ds.dataset_attribute.test)
    # Execute each test on the dataset
    execute_methods(tests, ds, algo)


def run_algo_docker(
    algorithm,
    dataset,
    algorithm_params=None,
    cpu_limit=None,
    mem_limit=None,
    pipeline=False,
    step=False,
):
    """
    Runs an algorithm inside a docker container
    """

    cmd = ["--algorithm", algorithm, "--dataset", dataset, "--locally"]
    if pipeline:
        cmd.append("--pipeline")
    elif step:
        cmd.append("--pipeline-step")

    if mem_limit is not None:
        cmd.append("--mem-limit")
        cmd.append(f"{mem_limit}g")

    if cpu_limit is not None:
        cmd.append("--cpu-limit")
        cmd.append(str(cpu_limit))

    docker_tag = f"df-benchmarks-{algorithm}"
    if mem_limit is None:
        mem_limit = int(psutil.virtual_memory().available / (1024.0**3))

    if algorithm_params:
        if cpu_limit is None:
            cpu_limit = 1
        algorithm_params = set_algorithm_params(
            algorithm_params, mem_limit, str(cpu_limit)
        )
        cmd.insert(2, "--algorithm_params")
        cmd.insert(3, json.dumps(algorithm_params))
    shm = int(mem_limit * 0.1)
    mem_limit -= shm
    shm = f"{int(mem_limit * 0.1)}g"
    mem_limit = f"{mem_limit}g"
    """
    cpu_limit is the number of cpu (cores) that docker can use.
    The parameter "cpuset_cpus" takes the cpus that docker is allowed to use.
    E.g. 0 means the core 0, 0-10 means the cores: 0, 1, 2, ... 10.
    For example, if we want to allow docker to use 4 cores, we have to set the input 0-3,
    this means that is allowed to use the cores 0, 1, 2, and 3.
    {"spark.default.parallelism": "111", "spark.driver.maxResultSize": "-1", "spark.driver.memory": "170g", "spark.executor.memory": "170g", "spark.executor.cores": "1"}
    {{'spark.default.parallelism': '111', 'spark.driver.maxResultSize': '-1', 'spark.driver.memory': '170g', 'spark.executor.memory': '170g', 'spark.executor.cores': '1'}
    """

    cpu_limit = "0" if cpu_limit is None else f"0-{cpu_limit - 1}"

    client = docker.from_env()
    container = client.containers.run(
        docker_tag,
        runtime="nvidia",
        command=cmd,
        # user="$(id -u):$(id -g)",
        volumes={
            os.path.abspath("."): {"bind": "/app/", "mode": "rw"},
            os.path.abspath("src"): {"bind": "/app/src", "mode": "ro"},
            os.path.abspath("datasets"): {"bind": "/app/datasets", "mode": "ro"},
            os.path.abspath("results"): {"bind": "/app/results", "mode": "rw"},
        },
        cpuset_cpus=str(cpu_limit),
        mem_limit=mem_limit,
        shm_size=shm,
        mem_swappiness=0,
        detach=True,
    )  # ,

    # show gpu configuration
    def stream_logs():
        for line in container.logs(stream=True):
            print(colors.color(line.decode().rstrip(), fg="white"))

    t = threading.Thread(target=stream_logs, daemon=True)
    t.start()

    try:
        exit_code = container.wait(timeout=20000)

        # Exit if exit code
        with contextlib.suppress(AttributeError):
            exit_code = exit_code["StatusCode"]
        if exit_code not in [0, None]:
            print(colors.color(container.logs().decode(), fg="red"))
            print(
                "Child process for container",
                container.short_id,
                "raised exception",
                exit_code,
            )
    except Exception:
        print(
            "Container.wait for container ",
            container.short_id,
            " failed with exception",
        )
        traceback.print_exc()
    finally:
        container.remove(force=True)


def run_pipeline_locally(
    algorithm,
    dataset,
    cpu_limit=None,
    mem_limit=None,
    algorithm_params=None,
    pipeline=False,
    step=False,
):
    """run the pipeline for the givrne algorithm and dataset

    Args:
        algorithm (AbstractAlgorithm): Algorithm to run
        dataset (Dataset): Dataset to run the algorithm on
        cpu_limit (int, optional): cpu limit. Defaults to None.
        mem_limit (int, optional): ram memory limits. Defaults to None.
        algorithm_params (dict, optional): configuration parameter to use. Defaults to None.
        pipeline (bool, optional): If True will be executed the full pipeline. Defaults to False.
        step (bool, optional): if True the pipeline will be executed by step. Defaults to False.
    """

    factory = AlgorithmsFactory()

    if step:
        pipeline = True

    log_mem_limit = mem_limit
    if mem_limit is None:
        mem_limit = int(psutil.virtual_memory().available / (1024.0**3))
        log_mem_limit = "max"

    if algorithm_params:
        algorithm_params = set_algorithm_params(algorithm_params, mem_limit, cpu_limit)

    algo = factory.build_algorithm(
        algorithm, mem_limit, cpu_limit, conf=algorithm_params, pipeline=pipeline
    )

    # Create log file
    ds = Dataset().get_dataset_by_name(dataset)

    # Load the tests to perform on the dataset
    pipeline_step = load_tests(ds.dataset_attribute.pipe)
    if (not step) and pipeline:
        print("Running pipeline")
        process = psutil.Process(os.getpid())
        start_time = time.time()
        memory_used_s = psutil.virtual_memory().used
        # Start tracking memory usage
        resource.setrlimit(
            resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        tracemalloc.start()

        # Get the current memory usage
        before = tracemalloc.get_traced_memory()

        for key, tests in pipeline_step.items():
            execute_methods(tests, ds, algo, step)

        algo.force_execution()

        # Get the current memory usage
        after = tracemalloc.get_traced_memory()

        # Stop tracing memory allocations
        tracemalloc.stop()
        ram = max(((tracemalloc.get_traced_memory()[1])), 0)

        # Stop the stopwatch and calculate the elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Stop tracking memory usage and get the memory usage
        memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        memory_used_e = psutil.virtual_memory().used

        print(f"Memory usage: {abs((memory_used_e - memory_used_s))}")
        memory_used = max((memory_used_e - memory_used_s), 0)

        save_to_csv("pipe", algo, elapsed_time, memory_used, ram, pipeline, step)

    elif step:
        process = psutil.Process(os.getpid())
        # Get initial CPU usage
        initial_cpu_percent = psutil.cpu_percent()
        # get the initial memory usage

        # Start tracking memory usage
        resource.setrlimit(
            resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
        memory_used_s = psutil.virtual_memory().used
        before_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print(before_memory // 1024**3)
        ram = 0
        tracemalloc.start()

        for key, tests in pipeline_step.items():
            print("Running pipeline step: ", key)
            # execute_methods(tests, ds, algo)
            start_time = time.time()
            # Start tracking memory usage
            resource.setrlimit(
                resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
            )

            # Call the function
            execute_methods(tests, ds, algo, step)
            # get the final memory usage
            tracemalloc.stop()
            ram = max(((tracemalloc.get_traced_memory()[1])), 0)
            after_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            # print(after_memory // 1024**3)
            # Get final CPU usage
            final_cpu_percent = psutil.cpu_percent()

            # Stop the stopwatch and calculate the elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Print results

            # Stop tracking memory usage and get the memory usage
            memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

            memory_used_e = psutil.virtual_memory().used
            print(f"Memory usage: {abs((memory_used_e - memory_used_s))} GB")
            memory_used = max((memory_used_e - memory_used_s), 0)

            save_to_csv(key, algo, elapsed_time, memory_used, ram, pipeline, step)
    else:
        print("Running core functions")
        for key, tests in pipeline_step.items():
            execute_methods(tests, ds, algo, step)
