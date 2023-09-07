# Dataframes benchmark
The aim of this benchmark is to compare several frameworks who manage DataFrames on common operations of data preparation.

## Install the benchmark
1. Clone this github repository on your machine;
2. Run `pip install -r requirements.txt`;
3. Run `python install.py` to build all the algorithms inside Docker containers\*.

\***Note**: you will need Docker installed on your machine. If you want to run the algorithms locally, avoid this step.

## Run an Algorithm

The command `python run_algorithm.py --algorithm <algorithm_name> --dataset <dataset_name>` will run an algorithm on the specified dataset. By default, an algorithm runs inside its Docker container. If you want to run it locally, add the parameter `--locally`.

The results of a run are stored in `results/<dataset_name>/<algorithm_name>.csv`.

`run_algorithm.py` takes as input the following parameters:

- `--algorithm <algorithm_name>`, mandatory, the name of the algorithm to run.
- `--algorithm_params`, optional, algorithm configuration parameters.
- `--dataset <dataset_name>`, mandatory, the dataset on which to run the algorithm.
- `--pipeline`, optional, flag for executing in pipeline mode.
- `--pipeline-step`, optional, flag for executing in pipeline divided by step.
- `--locally`, optional, runs the algorithm locally instead of in a Docker container.
- `--requirements`, optional, flag to install requirements.
- `--mem-limit`, optional, memory limit for the Docker container (default maximum available memory).
- `--cpu-limit <cpu_number>`, optional, maximum number of CPUs that the Docker container can use (default is 1).

## Add a new dataset
1. Create a new folder named as the dataset name inside the `dataset` folder;
2. Place the new dataset file inside your folder;
3. Create the file `dataset/<your_dataset_name>_template.json` and edit it;
4. The command `python add_dataset.py --dataset <dataset_name> --dataset_params <dataset_params>` will create the dataset entry in `src/data/datasets.json`.

`add_dataset.py` takes as input the following parameters:
- `--dataset <dataset_name>`, mandatory, the name of the dataset to add.
- `--dataset_params`, mandatory, dataset configuration parameters. The parameters must be provided as a json string, for example `--dataset_params '{"path": "dataset/<dataset_name>/<dataset_name>.csv", "pipe": "dataset/<dataset_name>/<dataset_name>_template.json", "type": "csv"}'`.


## Add a new algorithm
1. Create a docker file for your algorithm named `Dockerfile.your_algo` inside the `install` folder. It must contain all the instructions needed to install the required libraries (see as example `Dockerfile.pandas`);
2. Create a python class named `your_algo.py` inside the folder `src/algorithms/modules`. The class must extend and implement all the methods of the base class contained in `src/algorithms/algorithm.py`;
3. Add your algorithm definition in the build_algorithm function in the factory class `src/algorithms/algorithms_factory.py`.
```python
if algorithm_name == "dask":
   from src.algorithms.modules.dask_bench import DaskBench
   return DaskBench(mem, cpu, pipeline)
```

## Write a benchmark file

You can take as example the file `dataset/tests_template.json`.
For each method you want to execute you have to add a new object as the following one to the list:
```
{
   "method": "sort",
   "input": {"columns": ["Attr1", "Attr2"], "ascending": false}
}
```
In the previous example the method `sort` is called with the following input parameters `"columns": ["Attr1", "Attr2"], "ascending": false`.

If you need to differentiate the input of a method based on the algorithm, you can add a keyword named as `input_<algorithm_name>`, for example:
```
{
   "method": "cast_columns_types",
   "input": {"dtypes": {"id": "float32"}},
   "input_spark": {"dtypes": {"id": "float"}}
}
```
In the previous example all the algorithms will use as input for the method `cast_columns_types` `{"dtypes": {"id": "float32"}}`, while the `spark` algorithm will use `{"dtypes": {"id": "float"}}`.

If you need to include extra libraries or to execute extra commands before running the method the `extra_commands` list can be used. For example:
```
{
   "method": "cast_columns_types",
   "input": {"dtypes": {"id": "float32"}},
   "input_spark": {"dtypes": {"id": "float"}},
   "input_polars": {"dtypes": {"id": "polars.Float32"}, "req_compile": ["dtypes"], "extra_commands": ["import polars"]}
}
```
In the previous example, when the method `cast_columns_types` is called for the `polars` algorithm, first the `extra_commands` are executed: each command is parsed by using the python function `exec`, thus in this specific example the Polars library is imported.

If you need to pass an object to a function, you can put the code of the object inside the parameter that will provided as method input, then you have to include the parameter name inside the `req_compile` array, as shown in the previous example.
Every value of the parameters inserted in the `req_compile` array is parsed through the `eval` function of python.

For example, the value `polars.Float32` is replaced with the object `polars.datatypes.Float32` of the Polars library.

Here's an example including various methods and their inputs for different algorithms for pipeline execution:
```
{
    "Input": [
        {
            "method": "load_dataset",
            "input": {
                "sep": ","
            },
            "input_dask": {
                "sep": ",",
                "assume_missing": "True",
                "dtype": "object"
            },
            "input_koalas": {
                "sep": ",",
                "assume_missing": "True"
            },
            "input_vaex1": {
                "sep": ",",
                "low_memory": "False"
            },
            "input_vaex": {
                "lazy": true
            }
        },
        {
            "method": "force_execution",
            "input": {}
        }
    ],
    "EDA": [
        {
            "method": "get_columns",
            "input": {}
        },
        {
            "method": "locate_null_values",
            "input": {
                "column": "all"
            }
        },
        {
            "method": "sort",
            "input": {
                "columns": [
                    "Year"
                ]
            },
            "input_dask": {
                "columns": [
                    "Year"
                ],
                "cast": {
                    "Year": "int64"
                }
            }
        },
        {
            "method": "query",
            "input": {
                "query": "Year >= 1960 & Season == 'Summer'"
            },
            "input_rapids": {
                "query": "(Year >= 1960 and Season == 'Summer')"
            },
            "input_datatable": {
                "query": "((dt.f.Year >= 1960) & (dt.f.Season == 'Summer'))"
            },
            "input_polars": {
                "query": "(pl.col('Year') >= 1960) & (pl.col('Season') == 'Summer')",
                "req_compile": [
                    "query"
                ],
                "extra_commands": [
                    "import polars as pl"
                ]
            },
            "input_spark": {
                "query": "(fn.col('Year') >= 1960) & (fn.col('Season') == 'Summer')",
                "req_compile": [
                    "query"
                ],
                "extra_commands": [
                    "import pyspark.sql.functions as fn"
                ]
            },
            "input_pyspark_pandas": {
                "query": "('Year' >= 1960) and ('Season' == 'Summer')"
            },            
            "input_vaex":{
                "query": "Year >= 1960 and Season == 'Summer'"
            }
        },
        {
            "method": "force_execution",
            "input": {}
        }
    ]
}
```

## Plot the results
The script `make_charts.py` can be used to plot the results obtained on a dataset.