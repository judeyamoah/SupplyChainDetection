import pandas as pd

f = pd.read_csv("data_v2/sample_frame.csv")
bad = {l.split("\t")[0] for l in open("data_v2/failed.txt").read().splitlines() if l}
f["failed"] = f.package_name.isin(bad)
print(f.groupby("role").failed.agg(["sum", "count"]))
