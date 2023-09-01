import pandas as pd

df = pd.read_csv("datasets/state_patrol/state_patrol.csv")

# get length of dataframe
df_len = 27426840
# divide dataframe into 3 sample 25%, 50%, 75%
samples = {
    "75%": df_len * 0.75
}

seed = 42

for s, size in samples.items():
    sampled_df  = df.sample(int(size), random_state=seed)
    sampled_df.to_csv(f"datasets/state_patrol/state_patrol_{s}.csv", index=False)
    print(f"Exported {s} sample")
