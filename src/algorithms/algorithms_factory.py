from pydantic import BaseModel
from typing import Any
from pydantic_factories import ModelFactory
from src.algorithms.algorithm import AbstractAlgorithm









class AlgorithmsFactory(ModelFactory[Any]):
    @staticmethod
    def build_algorithm(algorithm_name:str, mem:str = None, cpu: int = None, conf=None, pipeline=False) -> AbstractAlgorithm:
        if conf is None:
            conf = {}
        try:
            if algorithm_name == "dask":
                from src.algorithms.modules.dask_bench import DaskBench
                return DaskBench(mem, cpu, pipeline)
            elif algorithm_name == "koalas":
                from src.algorithms.modules.koalas_bench import KoalasBench
                return KoalasBench(conf)

            elif algorithm_name == "modin_dask":
                from src.algorithms.modules.modin_bench import ModinBench
                return ModinBench(mem, cpu, type="dask", pipeline = pipeline)

            elif algorithm_name == "modin_hdk":
                from src.algorithms.modules.modin_bench import ModinBench
                return ModinBench(mem, cpu, type="hdk", pipeline=pipeline)

            elif algorithm_name == "modin_ray":
                from src.algorithms.modules.modin_bench import ModinBench
                return ModinBench(mem, cpu, type="ray", pipeline = pipeline)

            elif algorithm_name == "modin_unidist":
                from src.algorithms.modules.modin_bench import ModinBench
                return ModinBench(mem, cpu, type="unidist", pipeline=pipeline)

            elif algorithm_name in {"pandas", "pandas20"}:
                from src.algorithms.modules.pandas_bench import PandasBench
                return PandasBench(algorithm_name, mem, cpu, pipeline)
            
            elif algorithm_name == "pyspark_pandas":
                from src.algorithms.modules.pandas_pyspark_bench import PandasPysparkBench
                return PandasPysparkBench(algorithm_name, mem, cpu, pipeline)
            
            elif algorithm_name == "datatable":
                from src.algorithms.modules.datatable_bench import DataTableBench
                return DataTableBench(algorithm_name, mem, cpu, pipeline)


            elif algorithm_name == "polars":
                from src.algorithms.modules.polars_bench import PolarsBench
                return PolarsBench(mem, cpu, pipeline)
            
            elif algorithm_name == "polars_big":
                from src.algorithms.modules.polars_bench import PolarsBench
                return PolarsBench(mem, cpu, pipeline)
            
            elif algorithm_name == "rapids":
                from src.algorithms.modules.rapids_bench import RapidsBench
                return RapidsBench(mem, cpu, pipeline)          

            elif algorithm_name == "spark":
                from src.algorithms.modules.spark_bench import SparkBench
                return SparkBench(mem = mem, cpu = cpu, conf = conf, pipeline = pipeline)

            elif algorithm_name == "vaex":
                from src.algorithms.modules.vaex_bench import VaexBench
                return VaexBench(mem, cpu, pipeline)

            raise AssertionError("Algorithm type is not valid.")
        except AssertionError as e:
            print(e)
 