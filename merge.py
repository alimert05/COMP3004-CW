import pandas as pd, glob

dfs = [pd.read_csv(f) for f in sorted(glob.glob("results/map_*.csv"))]
pd.concat(dfs, ignore_index=True).to_csv("results/experiment_results.csv", index=False)
