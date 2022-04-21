import json
import pandas as pd

datapath = "./data/reddit_data.csv"
labelpath = "./data/reddit_labels.json"
outputpath = "./data/short_reddit_data.csv"
NUM_ROWS_PER_SUB_PER_PERCENT = 50
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1


with open(datapath, "r") as f:
    data = pd.read_csv(f)
with open(labelpath, "r") as f:
    labels = json.load(f)

sr_list = labels["subreddit"]
percentile_bins = labels["percentile_bin"]
train_amt = int(TRAIN_FRAC * NUM_ROWS_PER_SUB_PER_PERCENT)
val_amt = int(VAL_FRAC * NUM_ROWS_PER_SUB_PER_PERCENT)
test_amt = NUM_ROWS_PER_SUB_PER_PERCENT - train_amt - val_amt
filtered_rows = []
for sr in sr_list:
    for percent in percentile_bins:
        subdata = data[(data["SUBREDDIT"] == sr) & (data["PERCENTILE BIN"] == percent)]
        filtered_rows.extend(subdata[subdata["SPLIT"] == "train"].head(train_amt).values.tolist())
        filtered_rows.extend(subdata[subdata["SPLIT"] == "val"].head(val_amt).values.tolist())
        filtered_rows.extend(subdata[subdata["SPLIT"] == "test"].head(test_amt).values.tolist())
output = pd.DataFrame(filtered_rows, columns=data.columns)
with open(outputpath, "w") as f:
    output.to_csv(f, index=False)
