"""Verify that running the same experiment twice produces identical data."""
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df1 = pd.read_csv("results/reproducibility_run1.csv")
df2 = pd.read_csv("results/reproducibility_run2.csv")

# Sort both by the same deterministic key
sort_cols = ["map", "fire_position", "policy", "n_civilians", "trial"]
df1_sorted = df1.sort_values(sort_cols).reset_index(drop=True)
df2_sorted = df2.sort_values(sort_cols).reset_index(drop=True)

print("Run 1 shape:", df1.shape)
print("Run 2 shape:", df2.shape)

# Compare
if df1_sorted.equals(df2_sorted):
    print("\nRESULT: IDENTICAL DATA (reproducibility confirmed)")
    print("\nTo verify: row-by-row comparison")
    print(df1_sorted.to_string())
else:
    print("\nRESULT: DIFFER — NON-REPRODUCIBLE (bug!)")
    print("\nDifferences:")
    # Find columns that differ
    for col in df1_sorted.columns:
        if not (df1_sorted[col] == df2_sorted[col]).all():
            print(f"  Column '{col}' differs")
    # Show side-by-side
    cmp = pd.concat({"run1": df1_sorted, "run2": df2_sorted}, axis=1)
    print(cmp.to_string())
