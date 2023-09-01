import itertools
import os
import psutil
from src.algorithms.run import run_algo_docker, run_algo_locally, run_pipeline_locally
from src.algorithms.algorithms_factory import AlgorithmsFactory
from src.datasets.dataset import Dataset, DatasetAttribute

os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

            
import signal

def handler(signum, frame):
    raise Exception("Execution Time exceeded 1 hour")

if __name__ == "__main__":
    framework = ["rapids"]#"pyspark_pandas", "modin_ray", "modin_dask", "pandas20"] # "vaex", "polars", "pandas", "modin_dask", "modin_ray", "spark", "rapids", "pandas20", "datatable", "pyspark_pandas"
    datasets = [ "nyc_taxi_load"]#, "loan_parq","state_patrol_parq", "nyc_taxi_parq"] # , "athlete", "loan", "state_patrol", "nyc_taxi", "nyc_taxi_big"
    conf_cpu=[24]
    conf_mem=[128]
    step = [(False, False)]#, (True, False), (False, True)]#, (True, False), (False, True)] # False, False -> Core

    for d, f in itertools.product(datasets, framework):
        for i, _, item in itertools.product(range(1), range(1), step):
            #try:
            print(_)
            #signal.signal(signal.SIGALRM, handler)
            #signal.alarm(7200)
            if (f == "datatable") & (d == "nyc_100"):
                continue 

            # execute your script here
            print('-----------------{f}-------------------'.format(f=f))
            print('-----------------{d}---------------------'.format(d=d))
            print('-----------------{mem}----{cpu}----------------'.format(mem = conf_mem[i], cpu = conf_cpu[i]))
            run_algo_docker(
                f,
                d,
                mem_limit=conf_mem[i],
                cpu_limit=conf_cpu[i],
                step=item[0],
                pipeline=item[1],
            )

                                    #signal.alarm(0)
            # except Exception as e:
            #     print(e)
            #     print("Execution Time exceeded 2 hour")
                


#factory = AlgorithmsFactory()
#dask = factory.build_algorithm("dask")
#dataset = Dataset(name="nyc_taxi_big", dataset_attribute=DatasetAttribute(path="datasets/nyc_taxi_big/yellow_taxi_2009_2015_f32.hdf5", test= "datasets/nyc_taxi_big/nyc_taxi_test.json",type="hdf5"))
#dataset = Dataset(name="test_hdf5", dataset_attribute=DatasetAttribute(path="datasets/test_hdf5/test_hdf5.hdf5", test= "datasets/test_hdf5/test_hdf5.json",type="hdf5"))
#dataset.add_dataset()
#print(Dataset().get_dataset_by_name("test_hdf5"))
#dask.load_dataset("/home/angelo/df-benchmark-priv/datasets/census/census.csv", "csv")

# algorithm_params={
#     "spark.driver.memory": "96g",
#     "spark.executor.memory": "96g",
#     "spark.executor.memoryOverhead": "48g"
# },
    
# #     'spark.sql.warehouse.dir': 'file:///tmp/spark-warehouse',
# #     'spark.rdd.compress': 'True',
# #     'spark.serializer.objectStreamReset': '100',
# #     'spark.master': 'local[*]',
# #     'spark.submit.pyFiles': '',
# #     'spark.executor.id': 'driver',
# #     'spark.submit.deployMode': 'client',
# #     'spark.ui.showConsoleProgress': 'true'},
# {
#     "spark.default.parallelism": "24",
#     "spark.driver.maxResultSize": "-1",
#     "spark.driver.memory": "128g",
#     "spark.executor.memory": "128g",
#     #"spark.sql.execution.arrow.pyspark.enabled": True,
#     #"spark.master": "local[24]",
#     #"spark.executor.instances": "1",
# },
