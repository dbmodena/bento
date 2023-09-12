import argparse
import glob
import itertools
import math
import os
import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set matplotlib config
# fontsize and fontfamily
plt.rcParams["font.size"] = 25
plt.rcParams["font.family"] = "Times New Roman"

# set figure size
figure_size = (10, 6)

# column order
column_order = [
    "pandas",
    "pandas2.0",
    "pyspark-pandas",
    "pyspark-sql",
    "modin-dask",
    "modin-ray",
    "polars",
    "cuDF",
    "vaex",
    "datatable",
]

ncols = 5
nrows = len(column_order) // ncols

# set tight_layout as default
plt.rcParams["figure.autolayout"] = True

# set default legend position
plt.rcParams["legend.loc"] = "upper center"
plt.rcParams["legend.borderaxespad"] = 0.0
plt.rcParams["legend.fancybox"] = True


def reduce_dataset_name(df):
    """given a dataframe, it returns a dataframe with the dataset name reduced to 4 letters

    Keyword arguments:
    df -- dataframe
    Return: dataframe with the dataset name reduced to 4 letters
    """

    print(df.dataset.unique())
    datasets_letter = [d[:4] for d in df.dataset.unique()]
    df["dataset"] = df["dataset"].map(dict(zip(df.dataset.unique(), datasets_letter)))
    return df


def count_methods(datasets, path="./dataset/"):
    count = {}
    for d in datasets:
        file = json.load(open(f"{path}{d}/{d}_pipe.json"))
        # for every key count the number of "method"
        count[d] = {}
        print(d)
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
    return count_keys


def normalize_time_memory(df):
    count_keys = count_methods()
    tot_methods = {d: sum(count_keys[d].values()) for d in count_keys}
    grouped = df.groupby(["framework", "dataset"])["method"].count().reset_index()

    oom = {}
    for f, d in itertools.product(df["framework"].unique(), df["dataset"].unique()):
        if oom.get(d) is None:
            oom[d] = []
        tmp = grouped[(grouped["framework"] == f) & (grouped["dataset"] == d)]
        if tmp["method"].values[0] != tot_methods[d]:
            oom[d].append(f)

    print(oom)

    # remove the framework for each dataset that is out of memory
    for d, value in oom.items():
        for f in value:
            df = df[(df["dataset"] != d) | (df["framework"] != f)]

    return df


def framework_names_remapping(df):
    """given a dataframe, it returns a dataframe with the framework names remapped

    Keyword arguments:
    df -- dataframe
    Return: dataframe with the framework names remapped
    """

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
    """given a dataframe, it returns a dataframe with the framework names remapped

    Keyword arguments:
    df -- dataframe
    Return: dataframe with the framework names remapped
    """

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
    """given a method, it returns the step key

    Keyword arguments:
    method -- method name
    Return: step key
    """

    step_dict = {
        "i/o": [
            "read_csv",
            "read_json",
            "read_xml",
            "read_excel",
            "read_parquet",
            "read_sql",
            "load_from_pandas",
            "to_csv",
            "load_dataset",
            "get_pandas_df",
            "Input",
            "output",
        ],
        "eda": [
            "locate_null_values",
            "locate_outliers",
            "search_by_pattern",
            "sort",
            "get_columns",
            "get_columns_types",
            "get_stats",
            "is_unique",
            "check_allowed_char",
            "sample_rows",
            "query",
            "find_mismatched_dtypes",
            "EDA",
        ],
        "data_transformation": [
            "data_transformation",
            "cast_columns_types",
            "delete_columns",
            "rename_columns",
            "split",
            "merge_columns",
            "pivot",
            "unpivot",
            "calc_column",
            "duplicate_columns",
            "set_index",
            "join",
            "append",
            "min_max_scaler",
            "one_hot_encoding",
            "categorical_encoding",
            "groupby",
        ],
        "data_cleaning": [
            "data_cleaning",
            "change_date_time_format",
            "delete_empty_rows",
            "set_header_case",
            "set_content_case",
            "change_num_format",
            "round",
            "drop_duplicates",
            "get_duplicate_columns",
            "drop_by_pattern",
            "fill_nan",
            "replace",
            "edit",
            "set_value",
            "strip",
            "remove_diacritics",
        ],
    }

    return next((key for key, values in step_dict.items() if method in values), None)


def table_core_results_respect_pandas(core):
    """given a dataframe, it returns a dataframe with the results respect pandas

    Keyword arguments:
    core -- dataframe
    Return: dataframe with the results respect pandas and the rows to remove
    """

    core["step"] = core["method"].apply(lambda x: map_step_key(x))
    core = short_framework_names_remapping(core)
    pivoted_core = core.pivot_table(
        index=["dataset", "step", "method"], columns="framework", values="avg_time"
    )
    cols_order = [
        "dataset",
        "step",
        "method",
        "Pandas",
        "Pandas2",
        "SparkPD",
        "SparkSQL",
        "ModinD",
        "ModinR",
        "Polars",
        "cuDF",
        "Vaex",
        "Datatable",
    ]

    pivoted_core = core.pivot_table(
        index=["dataset", "step", "method"], columns="framework", values="avg_time"
    )
    pivoted_core = pivoted_core.sort_index(
        level=[0, 1], ascending=[True, False]
    ).reset_index()

    pivoted_core = pivoted_core[cols_order]
    rows_to_remove = pivoted_core[pivoted_core["Pandas"].isna()]
    pivoted_core = pivoted_core.drop(rows_to_remove.index)

    subset = [
        "Pandas2",
        "SparkPD",
        "SparkSQL",
        "ModinD",
        "ModinR",
        "Polars",
        "cuDF",
        "Vaex",
        "Datatable",
    ]

    for f in subset:
        pivoted_core[f] = pivoted_core["Pandas"] / pivoted_core[f]

    pivoted_core["Pandas"] = pivoted_core["Pandas"]
    # add s to pandas column
    pivoted_core.loc[pivoted_core["Pandas"].notna(), "Pandas"] = (
        pivoted_core.loc[pivoted_core["Pandas"].notna(), "Pandas"].astype(str) + "s"
    )

    pivoted_core.fillna("OoM", inplace=True)
    for c in ["dataset", "step", "method"]:
        pivoted_core[c] = pivoted_core[c].apply(lambda x: x.replace("_", "\_"))

    def bold_math_mode(df):
        # get max value in row
        # remove $ from the row
        # format to 10^
        # calc max value ignoring OoM
        df = df.replace("OoM", np.nan)
        max_val = df.max()
        max_val = f"{max_val:.1e}"
        return df.apply(
            lambda x: "\\textbf{" + str(max_val) + "}" if x == df.max() else x
        )

    pivoted_core[subset] = pivoted_core[subset].apply(bold_math_mode, axis=1)
    pivoted_core[subset].applymap(
        lambda x: "{:.1e}".format(x) if isinstance(x, float) else x
    )
    pivoted_core[subset] = pivoted_core[subset].applymap(lambda x: f"${str(x)}$")

    def text_colormap_text(row):
        row = row.str.replace("$", "")
        row = pd.to_numeric(row, errors="coerce")
        sorted_row = row.sort_values()

        min_values = sorted_row.iloc[:1].values
        # Create a list to store text colors
        text_colors = []
        for i in range(len(row)):
            if row.iloc[i] in min_values:
                # Set the color of the two minimum values to red
                text_colors.append("color: red")
                # format the values to .1e
                row.iloc[i] = f"{row.iloc[i]:.1e}"
            else:
                # Set the color of other values based on the colormap
                text_colors.append("")
                row.iloc[i] = f"{row.iloc[i]:.1e}"

        return text_colors

    pivoted_core = pivoted_core[pivoted_core["step"] != "i/o"]

    # make output charts dir if not exists
    if not os.path.exists("charts"):
        os.makedirs("charts")

    pivoted_core.style.apply(text_colormap_text, axis=1, subset=subset).to_latex(
        "charts/pivoted_core.tex", convert_css=True
    )
    pivoted_core = pivoted_core.style.apply(text_colormap_text, axis=1, subset=subset)

    rows_to_remove.reset_index(drop=True, inplace=True)
    for c in ["dataset", "step", "method"]:
        rows_to_remove[c] = rows_to_remove[c].apply(lambda x: x.replace("_", "\_"))

    rows_to_remove.fillna("OoM", inplace=True)
    rows_to_remove[subset] = rows_to_remove[subset].applymap(
        lambda x: "{:.1e}".format(x) if isinstance(x, float) else x
    )

    rows_to_remove.to_csv("charts/rows_to_remove.csv", index=False)

    return pivoted_core, rows_to_remove


def csv_parquet_io(load_csv, load_parq):
    # if Unnamed: 0 in load_csv.columns or Unnamed: 0 in load_parq.columns remove it
    if "Unnamed: 0" in load_csv.columns:
        load_csv.drop(columns=["Unnamed: 0"], inplace=True)
    if "Unnamed: 0" in load_parq.columns:
        load_parq.drop(columns=["Unnamed: 0"], inplace=True)

    load_parq["dataset"].replace(
        regex=True, inplace=True, to_replace="_parq", value=r""
    )
    load_csv = load_csv[load_csv["method"].isin(["load_dataset", "to_csv"])]
    load_parq = load_parq[load_parq["method"].isin(["load_dataset", "to_parquet"])]
    load_parq = short_framework_names_remapping(load_parq)
    load_csv = short_framework_names_remapping(load_csv)
    load_parq["type"] = "Parquet"
    load_csv["type"] = "CSV"

    map_dict = {"to_csv": "Write", "to_parquet": "Write", "load_dataset": "Read"}

    load_parq["method"] = load_parq["method"].map(map_dict)
    load_csv["method"] = load_csv["method"].map(map_dict)

    load = pd.concat([load_parq, load_csv], axis=0)

    load = load[["dataset", "framework", "method", "avg_time", "type"]]

    datasets = {
        "athlete": "Athlete",
        "loan": "Loan",
        "state_patrol": "Patrol",
        "nyc_taxi_load": "Taxi",
    }
    load["dataset"] = load["dataset"].map(datasets)

    fig, axes = plt.subplots(
        nrows=2, ncols=4, figsize=(24, 12), sharey=True, sharex=False
    )

    y_min = load[load["avg_time"] > 0]["avg_time"].min() * 0.5
    y_max = load[load["avg_time"] > 0]["avg_time"].max() * 1.8

    # colors for frameworks
    colors = ["#083d77", "#ee964b"]
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    for (s, d), i in zip(
        itertools.product(["Read", "Write"], ["Athlete", "Loan", "Patrol", "Taxi"]),
        enumerate(axes.flatten()),
    ):
        n = i[0]
        ax = i[1]
        l = letters[n]

        tmp = load[(load["dataset"] == d) & (load["method"] == s)]

        # set hatches

        tmp = tmp.pivot(
            index=["framework"], columns=["type", "method"], values="avg_time"
        ).sort_index(axis=1, ascending=True)
        tmp = tmp.reindex(
            [
                "Pandas",
                "Pandas2",
                "SparkPD",
                "SparkSQL",
                "ModinD",
                "ModinR",
                "Polars",
                "cuDF",
                "Vaex",
                "Datatable",
            ]
        )
        tmp.plot.bar(
            legend=False,
            rot=75,
            logy=True,
            linewidth=1.5,
            edgecolor="black",
            ax=ax,
            width=0.85,
            ylim=(y_min, y_max),
            color=colors,
        )

        ax.set_xlabel(f"({l}) {d}, {s}", fontsize=35)

        # set grid
        ax.grid(axis="y", linestyle="-", alpha=0.25, linewidth=0.3, color="grey")

        # add red cross when height is 0
        for i, row in tmp.iterrows():
            for j, v in enumerate(row):
                if str(v) == "nan":
                    # Get type
                    type_file = row.index[j][0].lower()
                    # get patches by dataset
                    x_v = tmp.index.get_loc(i)
                    p = ax.patches[x_v]

                    ax.text(
                        p.get_x() + p.get_width() / 2 + 0.3,
                        6,
                        "parquet not supported",
                        rotation=90,
                        color="red",
                        fontsize=20,
                        va="center",
                        weight="bold",
                        fontname="Times New Roman",
                    )
                elif v <= 0:
                    x_v = tmp.index.get_loc(i)
                    p = ax.patches[x_v]
                    ax.text(
                        p.get_x() + p.get_width() / 2 - 0.24,
                        0.4,
                        "out of mem",
                        rotation=90,
                        color="red",
                        fontsize=20,
                        va="center",
                        weight="bold",
                        fontname="Times New Roman",
                    )

    for a in [0, 4]:
        axes.flatten()[a].set_ylabel("Average time (s)", fontsize=35)

    handles = []
    labels = []
    hatches = ["", ".."]
    for i, f in enumerate(["CSV", "Parquet"]):
        handles.append(
            mpatches.Rectangle(
                (0, 0), 1, 1, facecolor=colors[i], edgecolor="black", hatch=hatches[i]
            )
        )
        labels.append(f)

    fig.legend(handles, labels, ncol=3, bbox_to_anchor=(0.52, 1.05))

    fig.tight_layout()
    # save figure
    fig.savefig("charts/csv_parquet_io.pdf", bbox_inches="tight")


def plot_step_datasets(step):
    step["step"] = step["method"].apply(lambda x: map_step_key(x))
    step.drop(columns=["method"], inplace=True)
    step.rename(columns={"step": "method"}, inplace=True)
    step = step[["dataset", "framework", "avg_time", "method"]]
    step = (
        step.groupby(["dataset", "framework", "method"])["avg_time"].sum().reset_index()
    )
    step = short_framework_names_remapping(step)
    steps = {
        "i/o": "I/O",
        "eda": "EDA",
        "data_transformation": "DT",
        "data_cleaning": "DC",
    }

    datasets = {
        "athlete": "Athlete",
        "loan": "Loan",
        "state_patrol": "Patrol",
        "nyc_taxi": "Taxi",
    }

    step["method"] = step["method"].map(steps)
    step["dataset"] = step["dataset"].map(datasets)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(24, 18), sharey=True)

    y_min = step[step["avg_time"] > 0]["avg_time"].min() * 0.5
    y_max = step[step["avg_time"] > 0]["avg_time"].max() * 1.8

    order = [
        "Pandas",
        "Pandas2",
        "SparkPD",
        "SparkSQL",
        "ModinD",
        "ModinR",
        "Polars",
        "cuDF",
        "Vaex",
        "Datatable",
    ]
    # colors for frameworks
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    # associate color to order
    color_order = dict(zip(order, colors))
    print(color_order)

    # letter in range axes
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"]

    for (s, d), i in zip(
        itertools.product(["EDA", "DT", "DC"], ["Athlete", "Loan", "Patrol", "Taxi"]),
        enumerate(axes.flatten()),
    ):
        n = i[0]
        ax = i[1]
        l = letters[n]
        tmp = step[(step["dataset"] == d) & (step["method"] == s)]
        tmp = tmp.pivot(index="framework", columns="method", values="avg_time")
        tmp = tmp.reindex(order)

        pl = tmp.plot.bar(
            legend=False,
            rot=75,
            logy=True,
            linewidth=1.5,
            edgecolor="black",
            ax=ax,
            width=0.8,
            ylim=(y_min, y_max),
        )
        ax.set_xlabel(f"({l}) {d}, {s}", fontsize=35, fontname="Times New Roman")

        xticklabels = ax.get_xticklabels()
        hatch = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
        for b in ax.containers:
            for i, p in enumerate(b.patches):
                # set color on the base of the framework
                # get xtik label
                xticklabel = xticklabels[i].get_text()
                p.set_color(color_order[xticklabel])
        for bars in pl.patches:
            bars.set_edgecolor("black")
            bars.set_hatch(hatch.pop(0))

        # add red cross if the height is 0
        for p in ax.patches:
            if p.get_height() == 0:
                ax.text(
                    p.get_x() + p.get_width() / 2 - 0.26,
                    1.9,
                    "out of mem",
                    rotation=90,
                    color="red",
                    fontsize=25,
                    va="center",
                    weight="bold",
                    fontname="Times New Roman",
                )

        ax.grid(axis="y", linestyle="-", alpha=0.25, linewidth=0.3, color="grey")

    for a in [0, 4, 8]:
        axes.flatten()[a].set_ylabel("Average time (s)", fontsize=35)

    for a in [3, 6, 9]:
        ax = axes.flatten()[a]
        # reduce width

    handles = []
    labels = []
    for i, f in enumerate(column_order):
        handles.append(mpatches.Patch(color=colors[i]))
        labels.append(f)

    fig.tight_layout()
    fig.savefig("step_avg.pdf", bbox_inches="tight")


def plot_eager_lazy(core, pipe, lazy_framework=["SparkSQL", "SparkPD", "Polars"]):
    core = short_framework_names_remapping(core)
    pipe = short_framework_names_remapping(pipe)
    core["step"] = core["method"].apply(lambda x: map_step_key(x))
    core = normalize_time_memory(core)
    core.drop(columns=["method"], inplace=True)
    core.rename(columns={"step": "method"}, inplace=True)
    core = (
        core.groupby(["framework", "dataset", "method"])["avg_time"].sum().reset_index()
    )
    core["type"] = "core"
    pipe["type"] = "pipe"
    pipe.loc[~pipe["framework"].isin(lazy_framework), "avg_time"] = 0.0

    pipe.drop(["avg_memory", "cpu", "mem"], axis=1, inplace=True)
    df_merged = pd.concat([core, pipe], ignore_index=True)
    df_merged = (
        df_merged.groupby(["dataset", "framework", "type"])
        .agg({"avg_time": "sum"})
        .reset_index()
    )

    datasets = {
        "athlete": "Athlete",
        "loan": "Loan",
        "state_patrol": "Patrol",
        "nyc_taxi": "Taxi",
    }
    df_merged["dataset"] = df_merged["dataset"].map(datasets)

    test = df_merged.copy()
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(24, 7), sharey=True)

    # get y_max for log scale
    y_min = test[test["avg_time"] > 0]["avg_time"].min() * 0.5
    y_max = test[test["avg_time"] > 0]["avg_time"].max() * 1.8

    colors = ["#1b9aaa", "#06d6a0"]
    hatches = {
        "core": "//",
        "pipe": "",
    }

    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    order = [
        "Pandas",
        "Pandas2",
        "SparkPD",
        "SparkSQL",
        "ModinD",
        "ModinR",
        "Polars",
        "cuDF",
        "Vaex",
        "Datatable",
    ]

    oom = {
        "Patrol": ["Pandas", "Pandas2"],
        "Taxi": ["Pandas", "Pandas2", "ModinR", "ModinD", "Polars", "SparkPD"],
    }

    for i, d in zip(enumerate(axes.flatten()), ["Athlete", "Loan", "Patrol", "Taxi"]):
        n = i[0]
        ax = i[1]
        letter = letters[n]

        tmp = (
            test[test["dataset"] == d]
            .pivot(index=["framework"], columns=["type"], values=["avg_time"])
            .sort_index(axis=1, level=1, ascending=True)
        )
        # index order by list
        tmp = tmp.reindex(order, axis=0, level=0)
        tmp.plot.bar(
            rot=75,
            legend=False,
            logy=True,
            fontsize=25,
            ax=ax,
            ylim=(y_min, y_max),
            width=0.85,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
        )

        bars = ax.patches
        patterns = ("//", "")
        hatches = [p for p in patterns for i in range(len(tmp))]
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_xlabel(f"({letter}) {d}", fontsize=40)

        # if dataset and framework are in oom_f add red cross
        for ds, fm in oom.items():
            if ds == d:
                # get tick label
                for t in ax.get_xticklabels():
                    if t.get_text() in fm:
                        ax.text(
                            t.get_position()[0] - 0.24,
                            21,
                            "out of mem",
                            rotation=90,
                            color="red",
                            fontsize=25,
                            va="center",
                            weight="bold",
                            fontname="Times New Roman",
                        )

        # add grid on the background
        ax.grid(axis="y", linestyle="-", alpha=0.25, linewidth=0.3, color="grey")

    # Add y label only on the first plot
    axes[0].set_ylabel("Average time (s)", fontsize=40)

    fig.tight_layout()

    labels = ["Eager", "Lazy"]

    handles = [
        mpatches.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=colors[0],
            hatch="//",
            edgecolor="black",
            linewidth=1,
        ),
        mpatches.Rectangle(
            (0, 0), 1, 1, facecolor=colors[1], hatch="", edgecolor="black", linewidth=1
        ),
    ]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        borderaxespad=0.0,
        fancybox=True,
        ncol=2,
        fontsize=30,
        bbox_to_anchor=(0.52, 1.1),
    )
    fig.savefig("charts/eager_lazy.pdf", bbox_inches="tight")


def plot_scalability(
    tot_scalable,
    legend_order=[
        "Pandas",
        "Pandas2",
        "SparkPD",
        "SparkSQL",
        "ModinD",
        "ModinR",
        "Polars",
        "cuDF",
        "Vaex",
        "Datatable",
    ],
):
    tot_scalable = short_framework_names_remapping(tot_scalable)
    tot_scalable["avg_time"] = tot_scalable["avg_time"] / 60 / 60
    tot_scalable["avg_time"] = tot_scalable["avg_time"].round(2)
    fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 4))
    config = [(8, 16), (16, 64), (24, 128)]

    machine = ["Laptop", "WorkStation", "Server"]
    letter = ["a", "b", "c"]

    # associate color to each framework
    color_map = plt.cm.get_cmap("Paired")
    colors = {
        f: color_map(tot_scalable.framework.unique().tolist().index(f))
        for f in tot_scalable.framework.unique()
    }
    for ax, i in zip(axes.flatten(), range(len(config))):
        df = tot_scalable[
            (tot_scalable.cpu == config[i][0]) & (tot_scalable.mem == config[i][1])
        ]
        pivoted = df.pivot(index=["dataset"], columns="framework", values="avg_time")

        pivoted.plot.line(
            ax=ax, lw=4.5, color=[colors[x] for x in pivoted.columns.to_list()]
        )
        # add marker when the line is not continuous
        columns = pivoted.columns.to_list()
        indices = pivoted.index.to_list()

        # remove legend
        ax.get_legend().remove()

        handles, labels = plt.gca().get_legend_handles_labels()
        # add first x label
        ax.set_xlabel("# Rows (M)\n", rotation=0, fontsize=20)

        # add text under x axis
        ax.text(
            0.47,
            -0.5,
            f"({letter[i]}) {machine[i]}",
            transform=ax.transAxes,
            ha="center",
            fontsize=25,
        )

        ax.set_ylabel("Time (h)", fontsize=25)

        # add more x ticks
        ax.set_xticks(np.arange(0, 100, 20))
        ax.tick_params(axis="both", which="major", labelsize=18)

        # add grid to the ax
        ax.grid(True, which="major", linestyle="--")

        # add scatter plot when y is nan
        for i in range(pivoted.index.size):
            row = pivoted.loc[indices[i]]
            for c in columns:
                if np.isnan(row[c]) & (not np.isnan(pivoted.loc[indices[i - 1]][c])):
                    ax.plot(
                        indices[i - 1],
                        pivoted.iloc[i - 1][c],
                        marker="x",
                        color="red",
                        ms=12,
                        mew=3,
                    )

    fig.tight_layout()

    # set legend order
    handles = [handles[labels.index(label)] for label in legend_order]
    labels = legend_order
    # add legend into ax 0
    fig.legend(
        handles,
        labels,
        loc="upper center",
        borderaxespad=0.0,
        fancybox=True,
        ncol=5,
        fontsize=20,
        bbox_to_anchor=(0.53, 1.15),
    )

    fig.savefig("charts/scalable.pdf", bbox_inches="tight")


def concatenate_csv(path, niter):
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
    )[:niter]
    dfs = [pd.read_csv(os.path.join(path, file)) for file in csv_files]
    print(f"Concatenating {csv_files}")
    return pd.concat(dfs)


def get_dataframe_avg_time(framework, datasets, pipe, step, niter=3):
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
            "framework",
            "method",
            "cpu",
            "mem",
            "avg_time",
            "avg_memory",
        ]
    )

    for f, d in itertools.product(framework, datasets):
        for cpu, mem in itertools.product(conf_cpu, conf_mem):
            d_name = d
            if step:
                d_name = f"{d}_pipe_step"
            elif pipe:
                d_name = f"{d}_pipe"

            path = f"./results/{d_name}/{f}_mem{mem}_cpu{cpu}/"

            # try to concatenate the last niter csv files if there is an error, it means that the algorithm is out of memory
            df = pd.DataFrame()  # commodity dataframe for concatenation of csv files
            out_of_memory = {}
            print(path)
            try:
                df = concatenate_csv(path, niter)
            except Exception as e:
                print(f"Out of memory {f} {d}")
                out_of_memory[f, d] = 1
                continue

            # for each method, it calculates the average time and memory
            for m in df["method"].unique():
                # insert the row in the dataframe
                avg_time_cpu.loc[len(avg_time_cpu)] = [
                    d,
                    f,
                    m,
                    cpu,
                    mem,
                    df[df["method"] == m]["time"].mean(),
                    df[df["method"] == m]["ram"].mean(),
                ]

    for k, v in out_of_memory.items():
        methods = avg_time_cpu.loc[
            (avg_time_cpu["dataset"] == k[1]) & (avg_time_cpu["framework"] == k[0])
        ]["method"].unique()
        for m in methods:
            row_to_add = [k[1], k[0], m, 0, 0, 0, 0]
            print(row_to_add)
            avg_time_cpu.loc[len(avg_time_cpu)] = [k[1], k[0], m, 0, 0, 0, 0]

        # avg_time_cpu.loc[len(avg_time_cpu)] = [k[1], datasets[k[1]], k[0], "out_of_memory", 0, 0, 0, 0]

    return avg_time_cpu


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", help="Datasets list", required=True)
    parser.add_argument("--framework", help="Frameworks list", required=True)
    parser.add_argument("--memory", help="Memory to plot", required=True)
    parser.add_argument("--cpu", help="CPU to plot", required=True)
    parser.add_argument("--core-results", help="Core results", action="store_true")
    parser.add_argument("--io-plot", help="IO plot", action="store_true")
    parser.add_argument("--step-plot", help="Step plot", action="store_true")
    parser.add_argument("--eager-lazy", help="Eager lazy plot", action="store_true")
    parser.add_argument("--scalability", help="Scalability plot", action="store_true")

    args = parser.parse_args()
    print(args)

    datasets = [*args.dataset.split(",")]
    frameworks = [*args.framework.split(",")]
    conf_mem = [*args.memory.split(",")]
    conf_cpu = [*args.cpu.split(",")]

    core = get_dataframe_avg_time(frameworks, datasets, False, False)
    step = get_dataframe_avg_time(frameworks, datasets, False, True)
    pipe = get_dataframe_avg_time(frameworks, datasets, True, False)

    if args.core_results:
        pivoted_core, rows_to_remove = table_core_results_respect_pandas(core)

    if args.io_plot:
        datasets_parq = [d + "_parq" for d in datasets]
        core_parquet = get_dataframe_avg_time(frameworks, datasets_parq, False, False)
        csv_parquet_io(core, core_parquet)

    if args.step_plot:
        plot_step_datasets(step)

    if args.eager_lazy:
        plot_eager_lazy(core, pipe)

    if args.scalability:
        plot_scalability(pipe)

    print("DONE")
