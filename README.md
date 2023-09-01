# Dataframes benchmark
The aim of this benchmark is to compare several frameworks who manage DataFrames on common operations of data preparation.

## Install the benchmark
1. Clone this github repository on your machine;
2. Run `pip install -r requirements.txt`;
3. Run `python install.py` to build all the algorithms inside Docker containers\*.

\***Note**: you will need Docker installed on your machine. If you want to run the algorithms locally, avoid this step.

## Run an algorithm
The command `python run_algorithm.py --algorithm <algorithm_name> --dataset <dataset_name>` will run an algorithm on the specified dataset.
By default an algorithm running inside its Docker container, if you want to run it locally add the parameter `--locally`.

The results of a run are stored in `results/<dataset_name>/<algorithm_name>.csv`.

*run_algorithm.py* takes as input the following parameters:
* --algorithm <algorithm_name>, mandatory, the name of the algorithm to run.
* --dataset <dataset_name>, mandatory, the dataset on which run the algorithm.
* --locally, optional, if set the algorithm will run locally, otherwise it will run inside its Docker container.
* --cpu_limit <cpu_number>, optional, maximum number of CPUs that the Docker container can use.
* --mem_limit <memory_limit>, optional, maximum memory that the Docker container can use.


## Add a new dataset
1. Create a new folder named as the dataset name inside the `dataset` folder;
2. Place the new dataset file inside your folder;
3. Copy the file `dataset/tests_template.json` inside your folder renaming it as `<your_dataset_name>_template.json` and edit it;
4. Edit the file `dataset/datasets.json` by adding the new dataset.

## Add a new algorithm
1. Create a docker file for your algorithm named `Dockerfile.your_algo` inside the `install` folder. It must contain all the instructions needed to install the required libraries (see as example `Dockerfile.pandas`);
2. Create a python class named `your_algo.py` inside the folder `df_benchmark/algorithms`. The class must extend and implement all the methods of the base class contained in `df_benchmark/algorithms/base.py`;
3. Add your algorithm definition in `df_benchmark/algorithms/algorithms.json` by using the following pattern
```
{
   "name": "algorithm_name",
   "module": "df_benchmark.algorithms.algorithm_name",
   "constructor": "className",
   "constructor_args": []
}
```
* name: the name of your algorithm.
* module: the name of the module which contains your class
* constructor: name name of your class
* constructor_args: arguments that have to be passed to the constructor when the class is instantiated

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

## Plot the results
The script `make_charts.py` can be used to plot the results obtained on a dataset.
The usage is `python make_charts.py --dataset <dataset_name> --var <variable to plot>`, the variable to plot can be `mem` to plot the memory usage or `time` to plot the time.
The generated figures will be placed in the folder `plots/<dataset name>`.