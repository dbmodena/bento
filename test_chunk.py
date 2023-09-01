import vaex
df = vaex.open('s3://vaex/taxi/yellow_taxi_2009_2015_f32.hdf5?anon=true')
print("DF shape:", df.shape)
# Sample n rows from the dataframe

sample = {
    "25": 120000000,
    "50": 240000000,
    "75": 360000000,
    "100": 480000000,
}


for s, v in sample.items():
    print(f"Sampling {s}% of the dataset")
    sampled_df = df.sample(n=v, random_state=42)  # Sampling 10% of the dataframe

    # Specify the output path for the sampled CSV file
    output_path = f"datasets/sampled_nyc_taxi/nyc_taxi_{s}.csv"
    sampled_df.export(output_path, progress=True)
    print(f"Exported {s}% of the dataset to {output_path}")