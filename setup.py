import itertools
import os
from src.algorithms.run import run_algo_docker, run_algo_locally, run_pipeline_locally
from src.algorithms.algorithms_factory import AlgorithmsFactory
from src.datasets.dataset import Dataset, DatasetAttribute

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

            
import signal

def handler(signum, frame):
    raise Exception("Execution Time exceeded 1 hour")


if __name__ == "__main__":
    #factory = AlgorithmsFactory()
    #dask = factory.build_algorithm("dask")
    #dataset = Dataset(name="nyc_taxi_big", dataset_attribute=DatasetAttribute(path="datasets/nyc_taxi_big/yellow_taxi_2009_2015_f32.hdf5", test= "datasets/nyc_taxi_big/nyc_taxi_test.json",type="hdf5"))
    #dataset = Dataset(name="test_hdf5", dataset_attribute=DatasetAttribute(path="datasets/test_hdf5/test_hdf5.hdf5", test= "datasets/test_hdf5/test_hdf5.json",type="hdf5"))
    #dataset.add_dataset()
    #print(Dataset().get_dataset_by_name("test_hdf5"))
    #dask.load_dataset("/home/angelo/df-benchmark-priv/datasets/census/census.csv", "csv")
    framework = ["spark", "pandas20"] # "dask", "vaex", "polars", "pandas", "modin_dask", "modin_ray", "spark", "rapids"
    datasets = ["loan_parquet"] # , "athlete", "loan", "state_patrol", "nyc_taxi", "nyc_taxi_big"
    conf_cpu=[24]
    conf_mem=[180]

    for f, d in itertools.product(framework, datasets):
        if d in ["state_patrol", "nyc_taxi", "nyc_taxi_big"] and f in ["koalas"]:
            continue
        for i, _ in itertools.product(range(1), range(3)):
            try:
                print(_)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(7200)

                # execute your script here
                print('-----------------{f}---------------------'.format(f=f))
                print('-----------------{d}---------------------'.format(d=d))
                print('-----------------{mem}----{cpu}---------------------'.format(mem = conf_mem[i], cpu = conf_cpu[i]))
                run_algo_docker(f, d,
                                algorithm_params={"spark.default.parallelism":"PARALLELISM", 
                                                "spark.driver.maxResultSize": "-1", 
                                                "spark.driver.memory":"MEM_LIMIT", 
                                                "spark.executor.memory": "MEM_LIMIT", 
                                                "spark.executor.cores": "CPU_LIMIT" },
                                mem_limit=conf_mem[i], cpu_limit=conf_cpu[i], pipeline=False)

                signal.alarm(0)
            except Exception as e:
                print("Execution Time exceeded 2 hour")
