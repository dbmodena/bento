from df_benchmark.run import load_datasets, load_algorithms
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import math

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