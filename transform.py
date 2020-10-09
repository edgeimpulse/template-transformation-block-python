import pyarrow.parquet as pq
import numpy as np
import math, os, sys, argparse, json, hmac, hashlib, time
import pandas as pd

# these are the three arguments that we get in
parser = argparse.ArgumentParser(description='Organization transformation block')
parser.add_argument('--in-file', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args, unknown = parser.parse_known_args()

# verify that the input file exists and create the output directory if needed
if not os.path.exists(args.in_file):
    print('--in-file argument', args.in_file, 'does not exist', flush=True)
    exit(1)

if not os.path.exists(args.out_directory):
    os.makedirs(args.out_directory)

# load and parse the input file
print('Loading parquet file', args.in_file, flush=True)
table = pq.read_table(args.in_file)
data = table.to_pandas()

features = []

# we group by label and then extract some metrics
for label in data.label.unique():
    data_per_label = data[data.label == label]

    # calculate the RMS per axis
    features.append({
        'label': label,
        'rmsX': np.sqrt(np.mean(data_per_label.accX**2)),
        'rmsY': np.sqrt(np.mean(data_per_label.accY**2)),
        'rmsZ': np.sqrt(np.mean(data_per_label.accZ**2))
    })

# and store as new file in the output directory
out_file = os.path.join(args.out_directory, os.path.splitext(os.path.basename(args.in_file))[0] + '_features.parquet')
pd.DataFrame(features).to_parquet(out_file)

print('Written features file', out_file, flush=True)