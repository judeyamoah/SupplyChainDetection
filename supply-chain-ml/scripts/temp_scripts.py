# import pandas as pd

# fi = pd.read_csv(
#     "supply-chain-ml/results/feature_importance_random_forest.csv"
# )

# print(fi.head(20))

import json

with open(
    "supply-chain-ml/data/raw/npm_metadata_raw.json"
) as f:
    bg = {
        x["name"]
        for x in json.load(f)
    }

with open(
    "supply-chain-ml/data/raw/npm_vulnerable_metadata_raw.json"
) as f:
    vul = {
        x["name"]
        for x in json.load(f)
    }

print("Background:", len(bg))
print("Vulnerable:", len(vul))
print("Overlap:", len(bg.intersection(vul)))