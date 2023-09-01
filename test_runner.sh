#!/bin/bash
    #framework = ["pandas", "dask", "vaex", "polars", "spark", "koalas", "modin_dask", "modin_ray"]
    #datasets = ["census", "athlete", "loan", "state_patrol", "nyc_taxi"]
algo=dask
n_conf=(0 1 2)
conf_cpu=(8 16 24)
conf_mem=(16G 64G 128G)
dataset="census"

# For every algorithm
for al in ${algo[@]}; do
    # For every configuration
    for i in ${n_conf[@]}; do
        # Repeat the test 10 times
        for run in {1..10}; do
            python run_algorithm.py --algorithm $al --dataset $dataset --mem-limit ${conf_mem[$i]} --cpu-limit ${conf_cpu[$i]}
        done
    done
done
