from hashlib import algorithms_available
import os
import json
import resource
import tracemalloc
import subprocess
import time
import csv
import colors
from src.algorithms.algorithm import AbstractAlgorithm
import psutil
from src.datasets.dataset import Dataset

timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")


def save_to_csv(function_name, algo, elapsed_time, memory_usage, ram, pipeline=False, step = False):
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
    #print("Saving results...")
    if step:
        #print("Saving Step...")
        folder=f"{algo.ds_.name}_pipe_step/{algo.name}_mem{algo.mem_}_cpu{algo.cpu_}/"
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
        # print("Forcing execution...")
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

        # Run your method here

        # Stop tracking memory usage and get the memory usage
        #memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


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
    # check if the results folder exists
    if not os.path.exists("results"):
        os.makedirs("results")
    
    out_path = os.path.join("results", "errors.log")
    return open(out_path, "at")


def set_algorithm_params(args: dict, mem_limit, cpu_limit) -> dict:
    """
    Get the parameters for the algorithm
    """
    #print(mem_limit, cpu_limit)
    args = json.dumps(args)
    return json.loads(
        args.replace("MEM_LIMIT", f'{str(mem_limit)}g')
        .replace("CPU_LIMIT", str(cpu_limit))
        .replace("PARALLELISM", str(cpu_limit))
    )


def execute_methods(methods: dict, ds: Dataset, algo: AbstractAlgorithm, step = False):
    """Execute the methods for a specified dataset and algorithm

    Args:
        methods (dict): dictionary containing the methods to execute
        ds (Dataset): Dataset to use
        algo (AbstractAlgorithm): Algorithm to use
        step (bool, optional): Execute pipeline by step, the execution will be forced at the end of every step. 
                               Defaults to False.
    """
    # Open the error log file
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
                #print(f"Settings: {str(test[input])}")
                algo.load_dataset(ds, **test[input])
                algo.backup()
                # <#if (not step) and (not algo.pipeline):
                #     print("Forcing execution...")
                #     algo.force_execution()
                #     end = time.time()
                #     save_to_csv(test["method"], algo, end - start_time, 0, 0)
                
            # skip the force execution step if we execute a pipeline by step
            elif (not step) and (test["method"]=="force_execution"):
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
                        #print(f"Running extra command: {cmd}")
                        exec(cmd, globals())
                    input_cmd.pop("extra_commands")

                # Checks if some input requires to be compiled
                # i.e. replaced with objects from external libraries
                if "req_compile" in input_cmd:
                    for f in input_cmd["req_compile"]:
                        #print(f"Running req_compile: {input_cmd[f]}")
                        input_cmd[f] = compile_json(input_cmd[f], algo)
                    input_cmd.pop("req_compile")

                #print(f"Method input: {str(input_cmd)}")
                getattr(algo, test["method"])(**input_cmd)
                # if (not step) and (not algo.pipeline):
                #     print("Forcing execution...")
                #     algo.force_execution()
                #     end = time.time()
                #     save_to_csv(test["method"], algo, end - start_time, 0, 0)
                if ("restore" in test):
                    algo.restore()
        except Exception as e:
            print(
                colors.color("Cannot execute the method: " + test["method"] + "because" + str(e), fg="red")
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
            
    #print("DONE", algo.name)
    algo.done()
    err.close()
    
def build(library, args):
    """
    Install a docker container
    """
    print(f'Building {library}...')
    if args is not None and len(args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in args])
    else:
        q = ""

    try:
        subprocess.check_call(
            'docker build %s --rm -t df-benchmarks-%s -f'
            ' install/Dockerfile.%s .' % (q, library, library), shell=True)
        return {library: 'success'}
    except subprocess.CalledProcessError:
        return {library: 'fail'}

def install(algorithm, build_arg):
    """
    Install a docker container
    """
    if subprocess.run(["docker", "image", "inspect", "df-benchmarks"], stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode != 0:
        print('Downloading ubuntu container...')
        subprocess.check_call('docker pull ubuntu:18.04', shell=True)
        print('Building base image...')
        subprocess.check_call('docker build --rm -t df-benchmarks -f install/Dockerfile .', shell=True)
    else: 
        print('Base image already builded')

    if algorithm:
        tags = [algorithm]
    else:
        tags = [fn.split('.')[-1] for fn in os.listdir('install') if fn.startswith('Dockerfile.')]

    # check docker images that starts with df-benchmarks
    # get list
    images = [image.split(' ')[0] for image in os.popen("docker images | grep df-benchmarks").read().split('\n') if image != '']

    dockerfiles = [f.split('.')[1] for f in os.listdir('install') if f.startswith('Dockerfile.')]
    if algorithm not in dockerfiles:
        print('Image name not found in dockerfiles')
        # stop execution
        return 1

    if f'df-benchmarks-{algorithm}' in images:
        print('Algorithm image already builded')
        
    else:
        try: 
            install_status = [build(tag, build_arg) for tag in tags]
        except Exception as e:
            print(e)
            return 1
        print('\n\nInstall Status:\n' + '\n'.join(str(algo) for algo in install_status))
    return 0