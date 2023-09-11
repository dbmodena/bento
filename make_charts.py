import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import math
import itertools


step_dict = {
    "i/o": ["read_csv", "read_json", "read_xml", "read_excel", "read_parquet", "read_sql", "load_from_pandas", "to_csv", "load_dataset", "get_pandas_df", "Input", "output"],
    "eda": ["locate_null_values", "locate_outliers", "search_by_pattern", "sort", "get_columns", "get_columns_types", "get_stats", "is_unique", "check_allowed_char", "sample_rows", "query", "find_mismatched_dtypes", "EDA"],
    "data_transformation": ["data_transformation", "cast_columns_types", "delete_columns", "rename_columns", "split", "merge_columns", "pivot", "unpivot", "calc_column", "duplicate_columns", "set_index", "join", "append", "min_max_scaler", "one_hot_encoding", "categorical_encoding", "groupby"],
    "data_cleaning": ["data_cleaning", "change_date_time_format", "delete_empty_rows", "set_header_case", "set_content_case", "change_num_format", "round", "drop_duplicates", "get_duplicate_columns", "drop_by_pattern", "fill_nan", "replace", "edit", "set_value", "strip", "remove_diacritics"],
}




def reduce_dataset_name(df):
    print(df.dataset.unique())
    datasets_letter = [d[:4] for d in df.dataset.unique()]
    df['dataset'] = df['dataset'].map(dict(zip(df.dataset.unique(), datasets_letter)))
    return df

def concatenate_csv(path, niter=3):
    """given a path, it returns a dataframe with the last niter csv files

    Args:
        path (str): path of the folder
        niter (int, optional): number of csv files to concatenate. Defaults to 3.

    Returns:
        DataFrame: dataframe with the last niter csv files
    """
    all_files = os.listdir(path)
    csv_files = [file for file in all_files if file.endswith(".csv")]
    csv_files = sorted(
        csv_files, key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True
    )[:3]
    dfs = [pd.read_csv(os.path.join(path, file)) for file in csv_files]
    return pd.concat(dfs)


def get_dataframe_avg_time(framework, datasets, pipe, step, conf_cpu, conf_mem):
    """given a dataframe, it returns a dataframe with the average time for each algorithm method

    Args:
        df (DataFrame): dataframe with the time for each algorithm
        framework (str): framework name
        dataset (str): dataset name
        pipe (bool): if the dataframe is for the pipeline
        step (bool): if the dataframe is for the pipeline step

    Returns:
        DataFrame: dataframe with the average time for each algorithm
    """

    avg_time_cpu = pd.DataFrame(
        columns=[
            "dataset",
            "size",
            "framework",
            "method",
            "cpu",
            "mem",
            "avg_time",
            "avg_memory",
        ]
    )

    for f, (d, size) in itertools.product(framework, datasets.items()):
        for cpu, mem in itertools.product(conf_cpu, conf_mem):
            d_name = d
            if step:
                d_name = f"{d}_pipe_step"
            elif pipe:
                d_name = f"{d}_pipe"

            path = f"./results/{d_name}/{f}_mem{mem}g_cpu{cpu}/"

            # try to concatenate the last niter csv files if there is an error, it means that the algorithm is out of memory
            df = pd.DataFrame()  # commodity dataframe for concatenation of csv files
            try:
                df = concatenate_csv(path, niter=3)
            except Exception as e:
                print(f"Out of memory {f} {d}")
                continue

            # for each method, it calculates the average time and memory
            for m in df["method"].unique():
                # insert the row in the dataframe
                avg_time_cpu.loc[len(avg_time_cpu)] = [
                    d,
                    size,
                    f,
                    m,
                    cpu,
                    mem,
                    df[df["method"] == m]["time"].mean(),
                    df[df["method"] == m]["ram"].mean(),
                ]

    return avg_time_cpu


# open json file create a dataframe with indicates the key and the number of elements for each key
import json


def count_methods(df, path="./", datasets=["athlete", "loan", "state_patrol"]):
    count = {}
    for d in datasets:
        file = json.load(open(f"{path}{d}_pipe.json"))
        # for every key count the number of "method"
        count[d] = {}
        for k, e in file.items():
            # print unique methods
            for c in e:
                if c["method"] not in count:
                    if c["method"] == "force_execution":
                        continue
                    if k in ["Input", "output"]:
                        k = "i/o"
                    if k == "EDA":
                        k = "eda"
                    count[d][k, c["method"]] = 1
    # count the number of methods for each key
    count_keys = {}
    for c in count:
        count_keys[c] = {}
        for k in count[c].keys():
            if k[0] not in count_keys[c]:
                count_keys[c][k[0]] = 1
            else:
                count_keys[c][k[0]] += 1
    json.dump(count_keys, open("count_keys.json", "w"))
    for d in df["dataset"].unique():
        for k in df[df["dataset"] == d]["method"].unique():
            df.loc[(df["dataset"] == d) & (df["method"] == k), "count"] = count_keys[d][
                k
            ]
    return df


def normalize_time_memory(df):
    df = count_methods(df)
    # normlize avg_time and avg_memory

    # calculate the tot method for every dataset
    values = df.groupby(["dataset", "framework"])["count"].sum().reset_index()
    # map the value of the tot method for every dataset in the dataframe
    df["totm"] = df.apply(
        lambda x: values[
            (values["dataset"] == x["dataset"])
            & (values["framework"] == x["framework"])
        ]["count"].values[0],
        axis=1,
    )

    df["norm_avg_time"] = df["avg_time"] / df["totm"]

    return df


def framework_names_remapping(df):
    names_dict = {
        "pandas": "pandas",
        "pandas20": "pandas2.0",
        "rapids": "cuDF",
        "spark": "pyspark-sql",
        "pyspark_pandas": "pyspark-pandas",
        "vaex": "vaex",
        "datatable": "datatable",
        "modin_dask": "modin-dask",
        "modin_ray": "modin-ray",
        "polars": "polars",
    }
    df["framework"] = df["framework"].map(names_dict)
    return df

def short_framework_names_remapping(df):
    names_dict = {
        "pandas": "Pandas",
        "pandas20": "Pandas2",
        "rapids": "cuDF",
        "spark": "SparkSQL",
        "pyspark_pandas": "SparkPD",
        "vaex": "Vaex",
        "datatable": "Datatable",
        "modin_dask": "ModinD",
        "modin_ray": "ModinR",
        "polars": "aPolars",
    }
    df["framework"] = df["framework"].map(names_dict)
    return df

def map_step_key(method):
    return next(
        (key for key, values in step_dict.items() if method in values), None
    )
    
    












def load_df(framework, dataset, cpu, memory):
    folder = f"{dataset}_mem_{memory}_cpu_{cpu}"
    files = glob.glob(os.path.join("results", folder)+"/*.csv")
    
    paths = list(filter(lambda x: framework in x, files))
    
    df = pd.DataFrame(columns=["operation", framework+"_time", framework+"_mem"])
    
    if len(paths) > 0:    
        for path in paths:
            try:
                df1 = pd.read_csv(path)
                
                if list(df1.columns.values) != ['operation', 'time', 'mem']:
                    raise Exception("The schema format is not valid, it must be (operation, time, mem): "+path)
            
                if not df1['time'].dtype.kind in 'iufc':
                    raise Exception("The column 'time' must be numeric: "+path)
                
                if not df1['mem'].dtype.kind in 'iufc':
                    raise Exception("The column 'mem' must be numeric: "+path)
                
                for c in df1.columns:
                    if c != "operation":
                        df1 = df1.rename({c: framework+"_"+c}, axis=1)
                df = df.append(df1)
            except:
                print("The CSV format is not valid: "+path)
                continue
                #raise Exception("The CSV format is not valid: "+path)
        
        if len(df) > 0:
            agg = df.groupby('operation')
            return agg.mean().reset_index(), agg.std().reset_index()
        else:
            return df, df
    else:
        return df, df

def make_df(frameworks, dataset, cpu, memory):
    df_avg, df_std = load_df(frameworks[0], dataset, cpu, memory)
    
    frameworks1 = []
    if len(df_avg) > 0:
        frameworks1.append(frameworks[0])
    
    for i in range(1, len(frameworks)):
        df_avg1, df_std1 = load_df(frameworks[i], dataset, cpu, memory)
        if len(df_avg1) > 0:
            frameworks1.append(frameworks[i])
            df_avg = df_avg.merge(df_avg1, how='outer', on='operation')
            df_std = df_std.merge(df_std1, how='outer', on='operation')
    
    df_avg = df_avg[['operation'] + sorted([x for x in df_avg.columns if x != "operation"])]
    df_std = df_std[['operation'] + sorted([x for x in df_std.columns if x != "operation"])]
    frameworks1 = sorted(frameworks1)
    
    return frameworks1, df_avg, df_std
    
def plot(ax, var, row_avg, row_std, frameworks):
    ax.set_title(row_avg['operation'])
    x = np.arange(0, len(frameworks)/10, 0.1)
    y = row_avg[filter(lambda x: x.endswith(var), row_avg.keys())].values
    y_err = row_std[filter(lambda x: x.endswith(var), row_std.keys())].values
    
    ax.bar(x, np.nan_to_num(y), yerr=np.nan_to_num(y_err), capsize=5.0, width=0.08, edgecolor='black', color='#CCCCCC')
    for i in range(0, len(y)):
        if math.isnan(y[i]):
            ax.text(x[i]-0.015, 0, 'X', color='red', fontsize=40)
    ax.set_xticks(x)
    ax.set_xticklabels(frameworks, rotation=10)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.set_ylabel(var)
    ax.set_xlim([min(x)-0.045, max(x)+0.045])
    pass

def make_chart(dataset, cpu, memory, var="time", n_cols=10, width=8, height=5):
    frameworks = list(load_algorithms().keys())
    
    frameworks, df_avg, df_std = make_df(frameworks, dataset, cpu, memory)
    
    n_rows = math.ceil(len(df_avg)/n_cols)

    fig, ax = plt.subplots(n_rows, n_cols)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=0.85, wspace=0.5, hspace=None)
    fig.set_size_inches(width*n_cols, height*n_rows)

    x = 0
    y = 0

    for index, row in df_avg.iterrows():
        if n_rows < 2:
            plot(ax[y], var, row, df_std.loc[index], frameworks)
        else:
            plot(ax[x][y], var, row, df_std.loc[index], frameworks)
        
        y = (y+1)%n_cols
        
        if y == 0:
            x = x+1
            
    
    fig_name = f"{dataset}_mem_{memory}_cpu_{cpu}_{var}.pdf"
    out_path = os.path.join('plots', dataset)
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    print(os.path.join(out_path, dataset+"_"+var+".pdf"))
    fig.savefig(os.path.join(out_path, fig_name), format='pdf', dpi=1200, pad_inches=.05, bbox_inches="tight")
    pass



if __name__ == "__main__":  
    #Load available datasets
    datasets = load_datasets()
    
    #Set up argument parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        choices=datasets.keys(),
        help='Dataset name',
        required=True
    )
    parser.add_argument(
        '--memory',
        help="Memory to plot",
        required=True
    )
    parser.add_argument(
        '--cpu',
        help="CPU to plot",
        required=True
    )
    parser.add_argument(
        '--var',
        help='Variable to plot, memory usage (mem) or execution time (time)',
        choices=["time", "mem"],
        default="time"
    )
    parser.add_argument(
        '--n-cols',
        help='Number of columns in the figure',
        default=10,
        type=int
    )
    parser.add_argument(
        '--plot-width',
        help='Width of each plot',
        default=8,
        type=int
    )
    parser.add_argument(
        '--plot-height',
        help='Height of each plot',
        default=5,
        type=int
    )
    args = parser.parse_args()
    
    make_chart(args.dataset, args.cpu, args.memory, args.var, args.n_cols, args.plot_width, args.plot_height)
    print("DONE")